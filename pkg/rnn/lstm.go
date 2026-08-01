package rnn

import (
	"github.com/ppvan/gpt/pkg/core"
)

// LSTMCell with fused gates: instead of 4 separate weight matrices per
// input, the gates live side by side in single matrices, in the column
// order f | i | o | g:
//
//	W (inputSize, 4h), U (hiddenSize, 4h), B (1, 4h)
//
// Each Step then runs 2 large matmuls instead of 8 small ones. This is
// the key change for CPU utilization: BPTT is sequential over time
// steps, so the only parallelism available is *inside* each matmul, and
// a (batch x in)·(in x 4h) product gives the column-splitting Dot four
// times more width to spread across cores, with a quarter of the
// goroutine spawn/join cycles.
type LSTMCell struct {
	W core.Mat // input weights,  (inputSize, 4*hiddenSize)
	U core.Mat // hidden weights, (hiddenSize, 4*hiddenSize)
	B core.Mat // biases,         (1, 4*hiddenSize)

	hiddenSize int
}

type lstmCellCache struct {
	x     core.Mat
	hPrev core.Mat
	cPrev core.Mat

	// z is the fused post-activation matrix (batch, 4h), column layout f|i|o|g.
	// Storing the whole matrix instead of four SliceCols copies means the
	// backward pass can write dz directly into a pre-allocated (batch,4h) buffer
	// without HStack or any intermediate allocation.
	z core.Mat

	tanhC core.Mat
}

func NewLSTMCell(inputSize, hiddenSize int) *LSTMCell {
	// initialize each gate block independently so the per-gate Xavier
	// scale matches the previous separate-matrix layout; forget-gate
	// bias starts at 1 as before
	ones := core.NewZeroMat(1, hiddenSize).Apply(func(float32) float32 { return 1 })
	zeros := core.NewZeroMat(1, hiddenSize)
	return &LSTMCell{
		W: core.HStack(
			core.XavierMat(inputSize, hiddenSize),
			core.XavierMat(inputSize, hiddenSize),
			core.XavierMat(inputSize, hiddenSize),
			core.XavierMat(inputSize, hiddenSize),
		),
		U: core.HStack(
			core.XavierMat(hiddenSize, hiddenSize),
			core.XavierMat(hiddenSize, hiddenSize),
			core.XavierMat(hiddenSize, hiddenSize),
			core.XavierMat(hiddenSize, hiddenSize),
		),
		B: core.HStack(ones, zeros, zeros, zeros),

		hiddenSize: hiddenSize,
	}
}

func (c *LSTMCell) InitState(batch int) State {
	return State{
		core.NewZeroMat(batch, c.hiddenSize),
		core.NewZeroMat(batch, c.hiddenSize),
	}
}

func (c *LSTMCell) Step(x core.Mat, prev State) (next State, cache core.Cache) {
	hPrev, cPrev := prev[0], prev[1]
	h := c.hiddenSize

	// all four gate pre-activations in two matmuls
	z := x.Dot(c.W).AddInPlace(hPrev.Dot(c.U)).AddBiasInPlace(c.B)

	// sigmoid on the f|i|o blocks, tanh on the g block, in place
	z.ApplyColsInPlace(0, 3*h, core.FastSigmoid32)
	z.ApplyColsInPlace(3*h, 4*h, core.FastTanh32)

	f := z.SliceCols(0, h)
	i := z.SliceCols(h, 2*h)
	o := z.SliceCols(2*h, 3*h)
	g := z.SliceCols(3*h, 4*h)

	cNext := f.MulAdd(cPrev, i, g) // f⊙cPrev + i⊙g in one pass
	tanhC := cNext.Apply(core.FastTanh32)
	hNext := o.Hadamard(tanhC)

	cache = lstmCellCache{
		x:     x,
		hPrev: hPrev,
		cPrev: cPrev,
		z:     z, // full (batch,4h) post-activation matrix
		tanhC: tanhC,
	}

	return State{hNext, cNext}, cache
}

func (c *LSTMCell) BackwardStep(cache core.Cache, dNext State) (dx core.Mat, dPrev State, grads core.Grads) {
	cc := cache.(lstmCellCache)
	dh, dcNext := dNext[0], dNext[1]
	h := c.hiddenSize

	f := cc.z.SliceCols(0, h)
	i := cc.z.SliceCols(h, 2*h)
	o := cc.z.SliceCols(2*h, 3*h)
	g := cc.z.SliceCols(3*h, 4*h)

	do := dh.Hadamard(cc.tanhC)
	dc := dh.Hadamard(o).HadamardApply(cc.tanhC, core.FastDTanh32).AddInPlace(dcNext)

	df := dc.Hadamard(cc.cPrev)
	di := dc.Hadamard(g)
	dg := dc.Hadamard(i)
	dcPrev := dc.Hadamard(f)

	dz := core.NewZeroMat(dh.Rows, 4*h)
	dz.HadamardApplyWriteCols(0, df, f, core.FastDSigmoid32)
	dz.HadamardApplyWriteCols(h, di, i, core.FastDSigmoid32)
	dz.HadamardApplyWriteCols(2*h, do, o, core.FastDSigmoid32)
	dz.HadamardApplyWriteCols(3*h, dg, g, core.FastDTanh32)

	dx = dz.DotT(c.W)
	dhPrev := dz.DotT(c.U)

	grads = core.Grads{
		"W": cc.x.TDot(dz),
		"U": cc.hPrev.TDot(dz),
		"b": dz.SumRows(),
	}

	return dx, State{dhPrev, dcPrev}, grads
}

// BackwardStepInto is the hot training path: weight gradients are accumulated
// into caller-owned buffers (acc) rather than being returned as fresh matrices.
func (c *LSTMCell) BackwardStepInto(cache core.Cache, dNext State, acc core.Grads) (dx core.Mat, dPrev State) {
	cc := cache.(lstmCellCache)
	dh, dcNext := dNext[0], dNext[1]
	h := c.hiddenSize

	f := cc.z.SliceCols(0, h)
	i := cc.z.SliceCols(h, 2*h)
	o := cc.z.SliceCols(2*h, 3*h)
	g := cc.z.SliceCols(3*h, 4*h)

	do := dh.Hadamard(cc.tanhC)
	dc := dh.Hadamard(o).HadamardApply(cc.tanhC, core.FastDTanh32).AddInPlace(dcNext)

	df := dc.Hadamard(cc.cPrev)
	di := dc.Hadamard(g)
	dg := dc.Hadamard(i)
	dcPrev := dc.Hadamard(f)

	dz := core.NewZeroMat(dh.Rows, 4*h)
	dz.HadamardApplyWriteCols(0, df, f, core.FastDSigmoid32)
	dz.HadamardApplyWriteCols(h, di, i, core.FastDSigmoid32)
	dz.HadamardApplyWriteCols(2*h, do, o, core.FastDSigmoid32)
	dz.HadamardApplyWriteCols(3*h, dg, g, core.FastDTanh32)

	dx = dz.DotT(c.W)
	dhPrev := dz.DotT(c.U)

	cc.x.TDotAccInto(dz, acc["W"])
	cc.hPrev.TDotAccInto(dz, acc["U"])
	dz.SumRowsInto(acc["b"])

	return dx, State{dhPrev, dcPrev}
}

func (c *LSTMCell) Params() core.Params {
	return core.Params{"W": c.W, "U": c.U, "b": c.B}
}

// SetParams accepts either the fused layout ("W", "U", "b") or the
// legacy per-gate layout ("Wf".."Wg", "Uf".."Ug", "bf".."bg"), so old
// checkpoints keep loading; they are fused on the way in.
func (c *LSTMCell) SetParams(p core.Params) {
	if w, ok := p["W"]; ok {
		c.W, c.U, c.B = w, p["U"], p["b"]
		return
	}
	c.W = core.HStack(p["Wf"], p["Wi"], p["Wo"], p["Wg"])
	c.U = core.HStack(p["Uf"], p["Ui"], p["Uo"], p["Ug"])
	c.B = core.HStack(p["bf"], p["bi"], p["bo"], p["bg"])
}

func NewCharLSTM(vocabSize, embedSize, hiddenSize int) *CharRNN {
	return &CharRNN{
		Embed:      NewEmbedding(vocabSize, embedSize),
		Cell:       NewLSTMCell(embedSize, hiddenSize),
		Output:     newRnnLinear(hiddenSize, vocabSize),
		hiddenSize: hiddenSize,
		vocabSize:  vocabSize,
	}
}
