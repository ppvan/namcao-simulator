package mlp

import "github.com/ppvan/gpt/pkg/core"

type Layer interface {
	Forward(x core.Mat) (out core.Mat, cache core.Cache)
	Backward(cache core.Cache, dOut core.Mat) (dIn core.Mat, grads core.Grads)
}

type ParamLayer interface {
	Layer
	core.Parameter
}

type Sequential interface {
	Layer
	ParamLayer
	Add(l Layer)
}

type Prediction struct {
	Class int
	Probs []float64
}

type TrainMetrics struct {
	Epoch int
	Loss  float64
}
