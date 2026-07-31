package rnn

import (
	"github.com/ppvan/gpt/pkg/core"
)

type State []core.Mat

type Cell interface {
	InitState(batch int) State
	Step(x core.Mat, prev State) (next State, cache core.Cache)
	BackwardStep(cache core.Cache, dNext State) (dx core.Mat, dPrev State, grads core.Grads)
	BackwardStepInto(cache core.Cache, dNext State, acc core.Grads) (dx core.Mat, dPrev State)
	Params() core.Params
	SetParams(core.Params)
}

type RecurentCell struct {
	Wx, Wh, Bh core.Mat

	hiddenSize int
}

type rnnCellCache struct {
	x, hPrev, hNext core.Mat
}

func NewRecurentCell(inputSize, hiddenSize int) *RecurentCell {
	return &RecurentCell{
		Wx: core.XavierMat(inputSize, hiddenSize),
		Wh: core.XavierMat(hiddenSize, hiddenSize),
		Bh: core.NewZeroMat(1, hiddenSize),

		hiddenSize: hiddenSize,
	}
}

func (c *RecurentCell) InitState(batch int) State {
	return State{core.NewZeroMat(batch, c.hiddenSize)}
}

func (c *RecurentCell) Step(x core.Mat, prev State) (next State, cache core.Cache) {
	hPrev := prev[0]
	// x.Dot(Wx) is a fresh buffer, so accumulate and activate in place:
	// 4 allocations per step become 2.
	z := x.Dot(c.Wx).AddInPlace(hPrev.Dot(c.Wh)).AddBiasInPlace(c.Bh)
	hNext := z.ApplyInPlace(core.FastTanh32)
	return State{hNext}, rnnCellCache{x: x, hPrev: hPrev, hNext: hNext}
}

func (c *RecurentCell) BackwardStep(cache core.Cache, dNext State) (dx core.Mat, dPrev State, grads core.Grads) {
	cc := cache.(rnnCellCache)

	// dz = dNext ⊙ (1 - hNext²) fused into one pass, without
	// materializing the derivative matrix
	dz := dNext[0].HadamardApply(cc.hNext, dTanh)

	// TDot / DotT avoid materializing the transposed operands
	dWx := cc.x.TDot(dz)
	dWh := cc.hPrev.TDot(dz)
	dBh := dz.SumRows()

	dx = dz.DotT(c.Wx)
	dhPrev := dz.DotT(c.Wh)

	return dx, State{dhPrev}, core.Grads{"Wx": dWx, "Wh": dWh, "bh": dBh}
}

func (c *RecurentCell) Params() core.Params {
	return core.Params{"Wx": c.Wx, "Wh": c.Wh, "bh": c.Bh}
}

func (c *RecurentCell) SetParams(p core.Params) {
	c.Wx = p["Wx"]
	c.Wh = p["Wh"]
	c.Bh = p["bh"]
}

type rnnLinear struct {
	weights core.Mat
	biases  core.Mat
}

type rnnLinearCache struct {
	input core.Mat
}

func newRnnLinear(inputs, outputs int) *rnnLinear {
	return &rnnLinear{
		weights: core.HeMat(inputs, outputs),
		biases:  core.NewZeroMat(1, outputs),
	}
}

func (l *rnnLinear) Forward(x core.Mat) (out core.Mat, cache core.Cache) {
	// Dot result is fresh, so the bias add can mutate it in place.
	// For a char model this is the big (batch x hidden)·(hidden x vocab)
	// projection, so skipping the broadcast+add copies matters.
	out = x.Dot(l.weights).AddBiasInPlace(l.biases)
	return out, rnnLinearCache{input: x}
}

func (l *rnnLinear) Backward(cache core.Cache, dOut core.Mat) (dIn core.Mat, grads core.Grads) {
	c := cache.(rnnLinearCache)

	dW := c.input.TDot(dOut)
	dB := dOut.SumRows()
	dIn = dOut.DotT(l.weights)

	return dIn, core.Grads{"W": dW, "b": dB}
}

func (l *rnnLinear) Params() core.Params {
	return core.Params{"W": l.weights, "b": l.biases}
}

func (l *rnnLinear) SetParams(p core.Params) {
	l.weights = p["W"]
	l.biases = p["b"]
}

func (c *RecurentCell) BackwardStepInto(cache core.Cache, dNext State, acc core.Grads) (dx core.Mat, dPrev State) {
	cc := cache.(rnnCellCache)
	dz := dNext[0].HadamardApply(cc.hNext, dTanh)
	cc.x.TDotAccInto(dz, acc["Wx"])
	cc.hPrev.TDotAccInto(dz, acc["Wh"])
	dz.SumRowsInto(acc["bh"])
	dx = dz.DotT(c.Wx)
	return dx, State{dz.DotT(c.Wh)}
}

func (l *rnnLinear) BackwardInto(cache core.Cache, dOut core.Mat, dW, dB core.Mat) core.Mat {
	c := cache.(rnnLinearCache)
	c.input.TDotAccInto(dOut, dW)
	dOut.SumRowsInto(dB)
	return dOut.DotT(l.weights)
}

type CharRNN struct {
	Embed  *Embedding
	Cell   Cell
	Output *rnnLinear

	hiddenSize int
	vocabSize  int
}

func NewCharRNN(vocabSize, embedSize, hiddenSize int) *CharRNN {
	return &CharRNN{
		Embed:      NewEmbedding(vocabSize, embedSize),
		Cell:       NewRecurentCell(embedSize, hiddenSize),
		Output:     newRnnLinear(hiddenSize, vocabSize),
		hiddenSize: hiddenSize,
		vocabSize:  vocabSize,
	}
}

func (m *CharRNN) HiddenSize() int { return m.hiddenSize }

func (m *CharRNN) VocabSize() int { return m.vocabSize }

func (m *CharRNN) InitState(batch int) State {
	return m.Cell.InitState(batch)
}

type charRNNSeqCache struct {
	embedCaches []core.Cache
	cellCaches  []core.Cache
	outCaches   []core.Cache
}

func (m *CharRNN) ForwardSeq(xsIDs [][]int, s0 State) (logits []core.Mat, sFinal State, cache core.Cache) {
	T := len(xsIDs)
	s := s0
	logits = make([]core.Mat, T)
	sc := charRNNSeqCache{
		embedCaches: make([]core.Cache, T),
		cellCaches:  make([]core.Cache, T),
		outCaches:   make([]core.Cache, T),
	}

	for t, ids := range xsIDs {
		emb, ec := m.Embed.Forward(ids)
		sNext, cc := m.Cell.Step(emb, s)
		y, oc := m.Output.Forward(sNext[0])

		logits[t] = y
		sc.embedCaches[t] = ec
		sc.cellCaches[t] = cc
		sc.outCaches[t] = oc
		s = sNext
	}
	return logits, s, sc
}

func (m *CharRNN) StepOnce(ids []int, s State) (logits core.Mat, sNext State) {
	emb, _ := m.Embed.Forward(ids)
	sNext, _ = m.Cell.Step(emb, s)
	logits, _ = m.Output.Forward(sNext[0])
	return logits, sNext
}

func (m *CharRNN) BackwardSeq(cache core.Cache, dLogits []core.Mat) (grads core.Grads, dState0 State) {
	sc := cache.(charRNNSeqCache)
	T := len(dLogits)

	dWemb := core.NewZeroMat(m.Embed.Weights.Rows, m.Embed.Weights.Columns)
	dWy := core.NewZeroMat(m.Output.weights.Rows, m.Output.weights.Columns)
	dBy := core.NewZeroMat(m.Output.biases.Rows, m.Output.biases.Columns)

	cellAcc := core.Grads{}
	for name, p := range m.Cell.Params() {
		cellAcc[name] = core.NewZeroMat(p.Rows, p.Columns)
	}

	batch := 1
	if T > 0 {
		batch = dLogits[0].Rows
	}
	dState := m.Cell.InitState(batch)

	for t := T - 1; t >= 0; t-- {
		dh := m.Output.BackwardInto(sc.outCaches[t], dLogits[t], dWy, dBy)

		// dState[0] is either the fresh InitState buffer (first
		// iteration) or the fresh dPrev returned by BackwardStep, so
		// mutating it in place is safe and avoids a per-step copy.
		dState[0] = dState[0].AddInPlace(dh)

		dEmb, dPrev := m.Cell.BackwardStepInto(sc.cellCaches[t], dState, cellAcc)

		m.Embed.AccumulateInto(dWemb, sc.embedCaches[t], dEmb)

		dState = dPrev
	}

	grads = core.Grads{
		"embed.W": dWemb,
		"out.W":   dWy,
		"out.b":   dBy,
	}
	for name, g := range cellAcc {
		grads["cell."+name] = g
	}

	return grads, dState
}

func (m *CharRNN) Params() core.Params {
	p := core.Params{}
	for k, v := range m.Embed.Params() {
		p["embed."+k] = v
	}
	for k, v := range m.Cell.Params() {
		p["cell."+k] = v
	}
	for k, v := range m.Output.Params() {
		p["out."+k] = v
	}
	return p
}

func (m *CharRNN) SetParams(p core.Params) {
	m.Embed.SetParams(core.Params{"W": p["embed.W"]})

	cellParams := core.Params{}
	for name := range m.Cell.Params() {
		cellParams[name] = p["cell."+name]
	}
	m.Cell.SetParams(cellParams)

	m.Output.SetParams(core.Params{"W": p["out.W"], "b": p["out.b"]})
}
