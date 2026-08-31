package rnn

import (
	"github.com/ppvan/gpt/pkg/core"
)

// GRUCell implements the "linear before reset" GRU variant:
//
//	r = sigmoid(x Wr + hPrev Ur + br)
//	z = sigmoid(x Wz + hPrev Uz + bz)
//	n = tanh(x Wn + r * (hPrev Un + bn))
//	h' = (1-z)*n + z*hPrev
type GRUCell struct {
	Wr, Ur, Br core.Mat // reset gate
	Wz, Uz, Bz core.Mat // update gate
	Wn, Un, Bn core.Mat // candidate state

	hiddenSize int
}

type gruCellCache struct {
	x, hPrev core.Mat

	r, z, n core.Mat // post-activation gate values
	hPrevUn core.Mat // hPrev.Dot(Un) + Bn, i.e. the un-gated hidden contribution to n
}

func NewGRUCell(inputSize, hiddenSize int) *GRUCell {
	return &GRUCell{
		Wr: core.XavierMat(inputSize, hiddenSize),
		Ur: core.XavierMat(hiddenSize, hiddenSize),
		Br: core.NewZeroMat(1, hiddenSize),

		Wz: core.XavierMat(inputSize, hiddenSize),
		Uz: core.XavierMat(hiddenSize, hiddenSize),
		Bz: core.NewZeroMat(1, hiddenSize),

		Wn: core.XavierMat(inputSize, hiddenSize),
		Un: core.XavierMat(hiddenSize, hiddenSize),
		Bn: core.NewZeroMat(1, hiddenSize),

		hiddenSize: hiddenSize,
	}
}

func (c *GRUCell) InitState(batch int) State {
	return State{core.NewZeroMat(batch, c.hiddenSize)}
}

func (c *GRUCell) Step(x core.Mat, prev State) (next State, cache core.Cache) {
	hPrev := prev[0]

	r := x.Dot(c.Wr).AddInPlace(hPrev.Dot(c.Ur)).AddBiasInPlace(c.Br).ApplyInPlace(core.FastSigmoid32)
	z := x.Dot(c.Wz).AddInPlace(hPrev.Dot(c.Uz)).AddBiasInPlace(c.Bz).ApplyInPlace(core.FastSigmoid32)

	hPrevUn := hPrev.Dot(c.Un).AddBiasInPlace(c.Bn)
	n := x.Dot(c.Wn).AddInPlace(r.Hadamard(hPrevUn)).ApplyInPlace(core.FastTanh32)

	oneMinusZ := z.Apply(func(v float32) float32 { return 1 - v })
	hNext := oneMinusZ.Hadamard(n).AddInPlace(z.Hadamard(hPrev))

	cache = gruCellCache{
		x: x, hPrev: hPrev,
		r: r, z: z, n: n,
		hPrevUn: hPrevUn,
	}
	return State{hNext}, cache
}

func (c *GRUCell) BackwardStep(cache core.Cache, dNext State) (dx core.Mat, dPrev State, grads core.Grads) {
	cc := cache.(gruCellCache)
	dh := dNext[0]

	// dz = dh * (hPrev - n)
	dz := dh.Hadamard(cc.hPrev).AddInPlace(
		dh.Hadamard(cc.n).Apply(func(v float32) float32 { return -v }),
	)
	// dn = dh * (1-z)
	dn := dh.Hadamard(cc.z.Apply(func(v float32) float32 { return 1 - v }))

	dnPre := dn.HadamardApply(cc.n, core.FastDTanh32)
	dzPre := dz.HadamardApply(cc.z, core.FastDSigmoid32)

	dr := dnPre.Hadamard(cc.hPrevUn)
	drPre := dr.HadamardApply(cc.r, core.FastDSigmoid32)

	dHPrevUn := dnPre.Hadamard(cc.r)
	dhPrevDirect := dh.Hadamard(cc.z)

	dx = drPre.DotT(c.Wr).AddInPlace(dzPre.DotT(c.Wz)).AddInPlace(dnPre.DotT(c.Wn))

	dhPrev := dhPrevDirect.
		AddInPlace(drPre.DotT(c.Ur)).
		AddInPlace(dzPre.DotT(c.Uz)).
		AddInPlace(dHPrevUn.DotT(c.Un))

	grads = core.Grads{
		"Wr": cc.x.TDot(drPre), "Ur": cc.hPrev.TDot(drPre), "br": drPre.SumRows(),
		"Wz": cc.x.TDot(dzPre), "Uz": cc.hPrev.TDot(dzPre), "bz": dzPre.SumRows(),
		"Wn": cc.x.TDot(dnPre), "Un": cc.hPrev.TDot(dHPrevUn), "bn": dHPrevUn.SumRows(),
	}

	return dx, State{dhPrev}, grads
}

func (c *GRUCell) BackwardStepInto(cache core.Cache, dNext State, acc core.Grads) (dx core.Mat, dPrev State) {
	cc := cache.(gruCellCache)
	dh := dNext[0]

	dz := dh.Hadamard(cc.hPrev).AddInPlace(
		dh.Hadamard(cc.n).Apply(func(v float32) float32 { return -v }),
	)
	dn := dh.Hadamard(cc.z.Apply(func(v float32) float32 { return 1 - v }))

	dnPre := dn.HadamardApply(cc.n, core.FastDTanh32)
	dzPre := dz.HadamardApply(cc.z, core.FastDSigmoid32)

	dr := dnPre.Hadamard(cc.hPrevUn)
	drPre := dr.HadamardApply(cc.r, core.FastDSigmoid32)

	dHPrevUn := dnPre.Hadamard(cc.r)
	dhPrevDirect := dh.Hadamard(cc.z)

	cc.x.TDotAccInto(drPre, acc["Wr"])
	cc.hPrev.TDotAccInto(drPre, acc["Ur"])
	drPre.SumRowsInto(acc["br"])

	cc.x.TDotAccInto(dzPre, acc["Wz"])
	cc.hPrev.TDotAccInto(dzPre, acc["Uz"])
	dzPre.SumRowsInto(acc["bz"])

	cc.x.TDotAccInto(dnPre, acc["Wn"])
	cc.hPrev.TDotAccInto(dHPrevUn, acc["Un"])
	dHPrevUn.SumRowsInto(acc["bn"])

	dx = drPre.DotT(c.Wr).AddInPlace(dzPre.DotT(c.Wz)).AddInPlace(dnPre.DotT(c.Wn))

	dhPrev := dhPrevDirect.
		AddInPlace(drPre.DotT(c.Ur)).
		AddInPlace(dzPre.DotT(c.Uz)).
		AddInPlace(dHPrevUn.DotT(c.Un))

	return dx, State{dhPrev}
}

func (c *GRUCell) Params() core.Params {
	return core.Params{
		"Wr": c.Wr, "Ur": c.Ur, "br": c.Br,
		"Wz": c.Wz, "Uz": c.Uz, "bz": c.Bz,
		"Wn": c.Wn, "Un": c.Un, "bn": c.Bn,
	}
}

func (c *GRUCell) SetParams(p core.Params) {
	c.Wr, c.Ur, c.Br = p["Wr"], p["Ur"], p["br"]
	c.Wz, c.Uz, c.Bz = p["Wz"], p["Uz"], p["bz"]
	c.Wn, c.Un, c.Bn = p["Wn"], p["Un"], p["bn"]
}

func NewCharGRU(vocabSize, embedSize, hiddenSize int) *CharRNN {
	return &CharRNN{
		Embed:      NewEmbedding(vocabSize, embedSize),
		Cell:       NewGRUCell(embedSize, hiddenSize),
		Output:     newRnnLinear(hiddenSize, vocabSize),
		hiddenSize: hiddenSize,
		vocabSize:  vocabSize,
	}
}
