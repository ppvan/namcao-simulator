package mlp

import "github.com/ppvan/gpt/pkg/core"

type Linear struct {
	weights core.Mat
	biases  core.Mat
}

type linearCache struct {
	input core.Mat // x, needed for dW = x^T * dOut
}

func NewLinear(inputs, outputs int) *Linear {
	return &Linear{
		weights: core.HeMat(inputs, outputs),
		biases:  core.HeMat(1, outputs),
	}
}

func (l *Linear) Forward(x core.Mat) (out core.Mat, cache core.Cache) {
	one := core.NewZeroMat(x.Rows, 1).Apply(func(f float32) float32 { return 1 })
	b := one.Dot(l.biases)
	out = x.Dot(l.weights).Add(b)
	return out, linearCache{input: x}
}

func (l *Linear) Backward(cache core.Cache, dOut core.Mat) (dIn core.Mat, grads core.Grads) {
	c := cache.(linearCache)

	oneT := core.NewZeroMat(1, dOut.Rows).Apply(func(f float32) float32 { return 1 })

	dW := c.input.Transpose().Dot(dOut)
	dB := oneT.Dot(dOut)
	dIn = dOut.Dot(l.weights.Transpose())

	grads = core.Grads{
		"W": dW,
		"b": dB,
	}
	return dIn, grads
}

func (l *Linear) Params() core.Params {
	return core.Params{
		"W": l.weights,
		"b": l.biases,
	}
}

func (l *Linear) SetParams(p core.Params) {
	l.weights = p["W"]
	l.biases = p["b"]
}
