package rnn

import (
	"math"

	"github.com/ppvan/gpt/pkg/core"
)

func sigmoid(v float64) float64 {
	return 1.0 / (1.0 + math.Exp(-v))
}

func dSigmoid(y float64) float64 {
	return y * (1 - y)
}

func dTanh(y float64) float64 {
	return 1 - y*y
}

type LSTMCell struct {
	Wf, Wi, Wo, Wg core.Mat
	Uf, Ui, Uo, Ug core.Mat
	Bf, Bi, Bo, Bg core.Mat

	hiddenSize int
}

type lstmCellCache struct {
	x     core.Mat
	hPrev core.Mat
	cPrev core.Mat

	f core.Mat
	i core.Mat
	o core.Mat
	g core.Mat

	tanhC core.Mat
}

func NewLSTMCell(inputSize, hiddenSize int) *LSTMCell {
	return &LSTMCell{
		Wf: core.XavierMat(inputSize, hiddenSize),
		Wi: core.XavierMat(inputSize, hiddenSize),
		Wo: core.XavierMat(inputSize, hiddenSize),
		Wg: core.XavierMat(inputSize, hiddenSize),

		Uf: core.XavierMat(hiddenSize, hiddenSize),
		Ui: core.XavierMat(hiddenSize, hiddenSize),
		Uo: core.XavierMat(hiddenSize, hiddenSize),
		Ug: core.XavierMat(hiddenSize, hiddenSize),

		Bf: core.NewZeroMat(1, hiddenSize).Apply(func(float64) float64 { return 1 }),
		Bi: core.NewZeroMat(1, hiddenSize),
		Bo: core.NewZeroMat(1, hiddenSize),
		Bg: core.NewZeroMat(1, hiddenSize),

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

	// The results of Dot are fresh allocations, so it is safe to
	// accumulate into them in place: each gate costs 2 matmuls plus a
	// single pre-activation buffer instead of 4 allocations.
	zf := x.Dot(c.Wf).AddInPlace(hPrev.Dot(c.Uf)).AddBiasInPlace(c.Bf)
	zi := x.Dot(c.Wi).AddInPlace(hPrev.Dot(c.Ui)).AddBiasInPlace(c.Bi)
	zo := x.Dot(c.Wo).AddInPlace(hPrev.Dot(c.Uo)).AddBiasInPlace(c.Bo)
	zg := x.Dot(c.Wg).AddInPlace(hPrev.Dot(c.Ug)).AddBiasInPlace(c.Bg)

	// z buffers are not needed after activation, so activate in place.
	f := zf.ApplyInPlace(sigmoid)
	i := zi.ApplyInPlace(sigmoid)
	o := zo.ApplyInPlace(sigmoid)
	g := zg.ApplyInPlace(math.Tanh)

	// cNext = f⊙cPrev + i⊙g in one fused pass
	cNext := f.MulAdd(cPrev, i, g)
	tanhC := cNext.Apply(math.Tanh)
	hNext := o.Hadamard(tanhC)

	cache = lstmCellCache{
		x:     x,
		hPrev: hPrev,
		cPrev: cPrev,
		f:     f,
		i:     i,
		o:     o,
		g:     g,
		tanhC: tanhC,
	}

	return State{hNext, cNext}, cache
}

func (c *LSTMCell) BackwardStep(cache core.Cache, dNext State) (dx core.Mat, dPrev State, grads core.Grads) {
	cc := cache.(lstmCellCache)
	dh, dcNext := dNext[0], dNext[1]

	do := dh.Hadamard(cc.tanhC)
	// dc = dcNext + dh⊙o⊙dTanh(tanhC); the Hadamard result is a fresh
	// buffer, so fuse the dTanh factor and accumulate dcNext in place.
	dc := dh.Hadamard(cc.o).HadamardApply(cc.tanhC, dTanh).AddInPlace(dcNext)

	df := dc.Hadamard(cc.cPrev)
	di := dc.Hadamard(cc.g)
	dg := dc.Hadamard(cc.i)
	dcPrev := dc.Hadamard(cc.f)

	// dz = dGate ⊙ activation'(gate), fused into one pass each
	dzf := df.HadamardApply(cc.f, dSigmoid)
	dzi := di.HadamardApply(cc.i, dSigmoid)
	dzo := do.HadamardApply(cc.o, dSigmoid)
	dzg := dg.HadamardApply(cc.g, dTanh)

	// DotT computes dz·Wᵀ and TDot computes xᵀ·dz directly, so the
	// eight per-step weight transposes are gone from the call sites,
	// and the running sums accumulate in place.
	dx = dzf.DotT(c.Wf).
		AddInPlace(dzi.DotT(c.Wi)).
		AddInPlace(dzo.DotT(c.Wo)).
		AddInPlace(dzg.DotT(c.Wg))

	dhPrev := dzf.DotT(c.Uf).
		AddInPlace(dzi.DotT(c.Ui)).
		AddInPlace(dzo.DotT(c.Uo)).
		AddInPlace(dzg.DotT(c.Ug))

	grads = core.Grads{
		"Wf": cc.x.TDot(dzf),
		"Wi": cc.x.TDot(dzi),
		"Wo": cc.x.TDot(dzo),
		"Wg": cc.x.TDot(dzg),

		"Uf": cc.hPrev.TDot(dzf),
		"Ui": cc.hPrev.TDot(dzi),
		"Uo": cc.hPrev.TDot(dzo),
		"Ug": cc.hPrev.TDot(dzg),

		"bf": sumRows(dzf),
		"bi": sumRows(dzi),
		"bo": sumRows(dzo),
		"bg": sumRows(dzg),
	}

	return dx, State{dhPrev, dcPrev}, grads
}

func (c *LSTMCell) Params() core.Params {
	return core.Params{
		"Wf": c.Wf, "Wi": c.Wi, "Wo": c.Wo, "Wg": c.Wg,
		"Uf": c.Uf, "Ui": c.Ui, "Uo": c.Uo, "Ug": c.Ug,
		"bf": c.Bf, "bi": c.Bi, "bo": c.Bo, "bg": c.Bg,
	}
}

func (c *LSTMCell) SetParams(p core.Params) {
	c.Wf, c.Wi, c.Wo, c.Wg = p["Wf"], p["Wi"], p["Wo"], p["Wg"]
	c.Uf, c.Ui, c.Uo, c.Ug = p["Uf"], p["Ui"], p["Uo"], p["Ug"]
	c.Bf, c.Bi, c.Bo, c.Bg = p["bf"], p["bi"], p["bo"], p["bg"]
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
