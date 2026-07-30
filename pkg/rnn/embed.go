package rnn

import "github.com/ppvan/gpt/pkg/core"

type Embedding struct {
	Weights core.Mat
}

type embeddingCache struct {
	ids []int
}

func NewEmbedding(vocabSize, embedSize int) *Embedding {
	return &Embedding{Weights: core.XavierMat(vocabSize, embedSize)}
}

func (e *Embedding) Forward(ids []int) (out core.Mat, cache core.Cache) {
	embedSize := e.Weights.Columns
	out = core.NewZeroMat(len(ids), embedSize)

	for i, id := range ids {
		for j := 0; j < embedSize; j++ {
			out.Set(i, j, e.Weights.Get(id, j))
		}
	}
	return out, embeddingCache{ids: ids}
}

func (e *Embedding) AccumulateInto(dW core.Mat, cache core.Cache, dOut core.Mat) {
	c := cache.(embeddingCache)
	for i, id := range c.ids {
		for j := 0; j < dOut.Columns; j++ {
			dW.Set(id, j, dW.Get(id, j)+dOut.Get(i, j))
		}
	}
}

func (e *Embedding) Backward(cache core.Cache, dOut core.Mat) core.Grads {
	dW := core.NewZeroMat(e.Weights.Rows, e.Weights.Columns)
	e.AccumulateInto(dW, cache, dOut)
	return core.Grads{"W": dW}
}

func (e *Embedding) Params() core.Params     { return core.Params{"W": e.Weights} }
func (e *Embedding) SetParams(p core.Params) { e.Weights = p["W"] }
