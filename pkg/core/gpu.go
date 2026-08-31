package core

import (
	"context"
	"encoding/binary"
	"fmt"
	"math"

	"github.com/gogpu/gputypes"
	"github.com/gogpu/wgpu"
	_ "github.com/gogpu/wgpu/hal/allbackends"
)

func dotIntoGPU(a, b, result []float32, m, n, p int) {
	out := getGPU().matmulTiled(a, b, m, n, p)
	copy(result, out)
}

func dotTInto(a, b, result []float32, m, n, q int) {
	out := getGPU().matmulTransposeB(a, b, m, n, q)
	copy(result, out)
}

func tDotAccInto(a, b, c []float32, m, n, p int) {
	out := getGPU().matmulTransposeAAcc(a, b, c, m, n, p)
	copy(c, out)
}

const gpuMatmulTiledShader = `
struct Params {
    M: u32,
    K: u32,
    N: u32,
    _pad: u32,
};

const TILE: u32 = 16u;

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

var<workgroup> tileA: array<f32, 256>;
var<workgroup> tileB: array<f32, 256>;

@compute @workgroup_size(16, 16, 1)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let row = gid.y;
    let col = gid.x;
    var sum: f32 = 0.0;

    let numTiles = (params.K + TILE - 1u) / TILE;
    for (var t: u32 = 0u; t < numTiles; t = t + 1u) {
        let aCol = t * TILE + lid.x;
        let bRow = t * TILE + lid.y;

        if (row < params.M && aCol < params.K) {
            tileA[lid.y * TILE + lid.x] = a[row * params.K + aCol];
        } else {
            tileA[lid.y * TILE + lid.x] = 0.0;
        }
        if (bRow < params.K && col < params.N) {
            tileB[lid.y * TILE + lid.x] = b[bRow * params.N + col];
        } else {
            tileB[lid.y * TILE + lid.x] = 0.0;
        }
        workgroupBarrier();

        for (var k: u32 = 0u; k < TILE; k = k + 1u) {
            sum = sum + tileA[lid.y * TILE + k] * tileB[k * TILE + lid.x];
        }
        workgroupBarrier();
    }

    if (row < params.M && col < params.N) {
        c[row * params.N + col] = sum;
    }
}
`

const gpuMatmulTransposeBShader = `
struct Params {
    M: u32,
    N: u32,
    Q: u32,
    _pad: u32,
};

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x;
    let row = gid.y;
    if (row >= params.M || col >= params.Q) {
        return;
    }
    var sum: f32 = 0.0;
    for (var k: u32 = 0u; k < params.N; k = k + 1u) {
        sum = sum + a[row * params.N + k] * b[col * params.N + k];
    }
    c[row * params.Q + col] = sum;
}
`

const gpuMatmulTransposeAAccShader = `
struct Params {
    M: u32,
    N: u32,
    P: u32,
    _pad: u32,
};

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x;
    let row = gid.y;
    if (row >= params.N || col >= params.P) {
        return;
    }
    var sum: f32 = 0.0;
    for (var k: u32 = 0u; k < params.M; k = k + 1u) {
        sum = sum + a[k * params.N + row] * b[k * params.P + col];
    }
    c[row * params.P + col] = c[row * params.P + col] + sum;
}
`

func (g *gpuOps) matmulTiled(a, b []float32, M, K, N int) []float32 {
	p := g.pipeline("matmul_tiled", gpuMatmulTiledShader)
	return g.run(p, a, b, nil, uint32(M), uint32(K), uint32(N), M, N)
}

func (g *gpuOps) matmulTransposeB(a, b []float32, M, N, Q int) []float32 {
	p := g.pipeline("matmul_transpose_b", gpuMatmulTransposeBShader)
	return g.run(p, a, b, nil, uint32(M), uint32(N), uint32(Q), M, Q)
}

func (g *gpuOps) matmulTransposeAAcc(a, b, cInit []float32, M, N, P int) []float32 {
	p := g.pipeline("matmul_transpose_a_acc", gpuMatmulTransposeAAccShader)
	return g.run(p, a, b, cInit, uint32(M), uint32(N), uint32(P), N, P)
}

func (g *gpuOps) run(pipeline *wgpu.ComputePipeline, a, b, cInit []float32, p0, p1, p2 uint32, outRows, outCols int) []float32 {
	g.mu.Lock()
	defer g.mu.Unlock()

	aBytes := gpuFloat32sToBytes(a)
	bBytes := gpuFloat32sToBytes(b)
	resultSize := uint64(outRows * outCols * 4)

	bufferA := g.uploadBuffer(aBytes, wgpu.BufferUsageStorage|wgpu.BufferUsageCopyDst)
	defer bufferA.Release()

	bufferB := g.uploadBuffer(bBytes, wgpu.BufferUsageStorage|wgpu.BufferUsageCopyDst)
	defer bufferB.Release()

	var bufferC *wgpu.Buffer
	if cInit != nil {
		bufferC = g.uploadBuffer(gpuFloat32sToBytes(cInit), wgpu.BufferUsageStorage|wgpu.BufferUsageCopySrc|wgpu.BufferUsageCopyDst)
	} else {
		buf, err := g.device.CreateBuffer(&wgpu.BufferDescriptor{
			Size:  resultSize,
			Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc | wgpu.BufferUsageCopyDst,
		})
		gpuMust(err)
		bufferC = buf
	}
	defer bufferC.Release()

	params := make([]byte, 16)
	binary.LittleEndian.PutUint32(params[0:4], p0)
	binary.LittleEndian.PutUint32(params[4:8], p1)
	binary.LittleEndian.PutUint32(params[8:12], p2)
	bufferParams := g.uploadBuffer(params, wgpu.BufferUsageUniform|wgpu.BufferUsageCopyDst)
	defer bufferParams.Release()

	bindGroup, err := g.device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Layout: g.binaryBGL,
		Entries: []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: bufferA, Offset: 0, Size: uint64(len(aBytes))},
			{Binding: 1, Buffer: bufferB, Offset: 0, Size: uint64(len(bBytes))},
			{Binding: 2, Buffer: bufferC, Offset: 0, Size: resultSize},
			{Binding: 3, Buffer: bufferParams, Offset: 0, Size: 16},
		},
	})
	gpuMust(err)
	defer bindGroup.Release()

	staging, err := g.device.CreateBuffer(&wgpu.BufferDescriptor{
		Size:  resultSize,
		Usage: wgpu.BufferUsageMapRead | wgpu.BufferUsageCopyDst,
	})
	gpuMust(err)
	defer staging.Release()

	encoder, err := g.device.CreateCommandEncoder(nil)
	gpuMust(err)

	pass, err := encoder.BeginComputePass(nil)
	gpuMust(err)
	pass.SetPipeline(pipeline)
	pass.SetBindGroup(0, bindGroup, nil)
	pass.Dispatch(uint32(gpuCeilDiv(outCols, 16)), uint32(gpuCeilDiv(outRows, 16)), 1)
	gpuMust(pass.End())

	encoder.CopyBufferToBuffer(bufferC, 0, staging, 0, resultSize)

	cmd, err := encoder.Finish()
	gpuMust(err)
	_, err = g.queue.Submit(cmd)
	gpuMust(err)

	gpuMust(staging.Map(context.Background(), wgpu.MapModeRead, 0, resultSize))
	defer func() { _ = staging.Unmap() }()

	mapped, err := staging.MappedRange(0, resultSize)
	gpuMust(err)
	defer mapped.Release()

	return gpuBytesToFloat32s(mapped.Bytes())
}

func (g *gpuOps) uploadBuffer(data []byte, usage gputypes.BufferUsage) *wgpu.Buffer {
	buf, err := g.device.CreateBuffer(&wgpu.BufferDescriptor{Size: uint64(len(data)), Usage: usage})
	gpuMust(err)
	gpuMust(g.queue.WriteBuffer(buf, 0, data))
	return buf
}

func gpuStorageEntry(binding uint32, readOnly bool) gputypes.BindGroupLayoutEntry {
	bufType := gputypes.BufferBindingTypeStorage
	if readOnly {
		bufType = gputypes.BufferBindingTypeReadOnlyStorage
	}
	return gputypes.BindGroupLayoutEntry{
		Binding:    binding,
		Visibility: gputypes.ShaderStageCompute,
		Buffer:     &gputypes.BufferBindingLayout{Type: bufType},
	}
}

func gpuUniformEntry(binding uint32) gputypes.BindGroupLayoutEntry {
	return gputypes.BindGroupLayoutEntry{
		Binding:    binding,
		Visibility: gputypes.ShaderStageCompute,
		Buffer:     &gputypes.BufferBindingLayout{Type: gputypes.BufferBindingTypeUniform},
	}
}

func gpuCeilDiv(a, b int) int { return (a + b - 1) / b }

func gpuFloat32sToBytes(fs []float32) []byte {
	out := make([]byte, len(fs)*4)
	for i, f := range fs {
		binary.LittleEndian.PutUint32(out[i*4:], math.Float32bits(f))
	}
	return out
}

func gpuBytesToFloat32s(b []byte) []float32 {
	out := make([]float32, len(b)/4)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[i*4:]))
	}
	return out
}

func gpuMust(err error) {
	if err != nil {
		panic(fmt.Sprintf("core: gpu op failed: %v", err))
	}
}
