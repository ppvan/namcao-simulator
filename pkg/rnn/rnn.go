package rnn

import "github.com/ppvan/gpt/pkg/core"

// RNN "cell" to abstract RNN, LSTM and GRU
type Cell interface {
	core.Parameter
	InitState(batch int) State
	Step(x core.Mat, prev State) (next State, cache core.Cache)
	BackwardStep(cache core.Cache, dNext State) (dx core.Mat, dPrev State, grads core.Grads)
	BackwardStepInto(cache core.Cache, dNext State, acc core.Grads) (dx core.Mat, dPrev State)
}
