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

	zf := x.Dot(c.Wf).Add(hPrev.Dot(c.Uf)).AddBias(c.Bf)
	zi := x.Dot(c.Wi).Add(hPrev.Dot(c.Ui)).AddBias(c.Bi)
	zo := x.Dot(c.Wo).Add(hPrev.Dot(c.Uo)).AddBias(c.Bo)
	zg := x.Dot(c.Wg).Add(hPrev.Dot(c.Ug)).AddBias(c.Bg)

	f := zf.Apply(sigmoid)
	i := zi.Apply(sigmoid)
	o := zo.Apply(sigmoid)
	g := zg.Apply(math.Tanh)

	cNext := f.Hadamard(cPrev).Add(i.Hadamard(g))
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
	dc := dcNext.Add(dh.Hadamard(cc.o).Hadamard(cc.tanhC.Apply(dTanh)))

	df := dc.Hadamard(cc.cPrev)
	di := dc.Hadamard(cc.g)
	dg := dc.Hadamard(cc.i)
	dcPrev := dc.Hadamard(cc.f)

	dzf := df.Hadamard(cc.f.Apply(dSigmoid))
	dzi := di.Hadamard(cc.i.Apply(dSigmoid))
	dzo := do.Hadamard(cc.o.Apply(dSigmoid))
	dzg := dg.Hadamard(cc.g.Apply(dTanh))

	xT := cc.x.Transpose()
	hT := cc.hPrev.Transpose()

	dx = dzf.Dot(c.Wf.Transpose()).
		Add(dzi.Dot(c.Wi.Transpose())).
		Add(dzo.Dot(c.Wo.Transpose())).
		Add(dzg.Dot(c.Wg.Transpose()))

	dhPrev := dzf.Dot(c.Uf.Transpose()).
		Add(dzi.Dot(c.Ui.Transpose())).
		Add(dzo.Dot(c.Uo.Transpose())).
		Add(dzg.Dot(c.Ug.Transpose()))

	grads = core.Grads{
		"Wf": xT.Dot(dzf),
		"Wi": xT.Dot(dzi),
		"Wo": xT.Dot(dzo),
		"Wg": xT.Dot(dzg),

		"Uf": hT.Dot(dzf),
		"Ui": hT.Dot(dzi),
		"Uo": hT.Dot(dzo),
		"Ug": hT.Dot(dzg),

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
