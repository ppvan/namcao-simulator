package core

import (
	"fmt"
	"sync"

	"github.com/gogpu/gputypes"
	"github.com/gogpu/wgpu"
	_ "github.com/gogpu/wgpu/hal/allbackends"
)

func InitGPU() error {
	_, err := getGPUOrErr()
	return err
}

var (
	gpuOnce sync.Once
	gpuCtx  *gpuOps
	gpuErr  error
)

func getGPUOrErr() (*gpuOps, error) {
	gpuOnce.Do(func() {
		gpuCtx, gpuErr = newGPUOps()
	})
	return gpuCtx, gpuErr
}

func getGPU() *gpuOps {
	g, err := getGPUOrErr()
	if err != nil {
		panic(fmt.Sprintf("core: GPU required but unavailable: %v", err))
	}
	return g
}

type gpuOps struct {
	instance *wgpu.Instance
	device   *wgpu.Device
	queue    *wgpu.Queue

	binaryBGL    *wgpu.BindGroupLayout
	binaryLayout *wgpu.PipelineLayout

	mu        sync.Mutex
	pipelines map[string]*wgpu.ComputePipeline
}

func newGPUOps() (*gpuOps, error) {
	instance, err := wgpu.CreateInstance(nil)
	if err != nil {
		return nil, fmt.Errorf("create instance: %w", err)
	}

	adapter, err := instance.RequestAdapter(&wgpu.RequestAdapterOptions{
		PowerPreference: wgpu.PowerPreferenceHighPerformance,
	})
	if err != nil {
		instance.Release()
		return nil, fmt.Errorf("request adapter: %w", err)
	}
	defer adapter.Release()

	device, err := adapter.RequestDevice(nil)
	if err != nil {
		instance.Release()
		return nil, fmt.Errorf("request device: %w", err)
	}

	bgl, err := device.CreateBindGroupLayout(&wgpu.BindGroupLayoutDescriptor{
		Label: "core binary BGL",
		Entries: []gputypes.BindGroupLayoutEntry{
			gpuStorageEntry(0, true),
			gpuStorageEntry(1, true),
			gpuStorageEntry(2, false),
			gpuUniformEntry(3),
		},
	})
	if err != nil {
		device.Release()
		instance.Release()
		return nil, fmt.Errorf("create bind group layout: %w", err)
	}

	layout, err := device.CreatePipelineLayout(&wgpu.PipelineLayoutDescriptor{
		Label:            "core binary Layout",
		BindGroupLayouts: []*wgpu.BindGroupLayout{bgl},
	})
	if err != nil {
		bgl.Release()
		device.Release()
		instance.Release()
		return nil, fmt.Errorf("create pipeline layout: %w", err)
	}

	return &gpuOps{
		instance:     instance,
		device:       device,
		queue:        device.Queue(),
		binaryBGL:    bgl,
		binaryLayout: layout,
		pipelines:    make(map[string]*wgpu.ComputePipeline),
	}, nil
}

func (g *gpuOps) pipeline(name, code string) *wgpu.ComputePipeline {
	g.mu.Lock()
	defer g.mu.Unlock()

	if p, ok := g.pipelines[name]; ok {
		return p
	}

	shader, err := g.device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{Label: name, WGSL: code})
	gpuMust(err)
	defer shader.Release()

	p, err := g.device.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Label: name, Layout: g.binaryLayout, Module: shader, EntryPoint: "main",
	})
	gpuMust(err)

	g.pipelines[name] = p
	return p
}
