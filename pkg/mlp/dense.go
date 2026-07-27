package mlp

import (
	"fmt"

	"github.com/ppvan/gpt/pkg/core"
)

type Dense struct {
	layers []Layer
}

func NewDense(layers ...Layer) *Dense {
	return &Dense{layers: layers}
}

func (s *Dense) Add(l Layer) {
	s.layers = append(s.layers, l)
}

func (s *Dense) Forward(x core.Mat) (core.Mat, core.Cache) {
	caches := make([]core.Cache, len(s.layers))
	out := x
	for i, l := range s.layers {
		var c core.Cache
		out, c = l.Forward(out)
		caches[i] = c
	}
	return out, caches
}

func (s *Dense) Backward(cache core.Cache, dOut core.Mat) (core.Mat, core.Grads) {
	caches := cache.([]core.Cache)
	grads := core.Grads{}
	dIn := dOut

	for i := len(s.layers) - 1; i >= 0; i-- {
		var layerGrads core.Grads
		dIn, layerGrads = s.layers[i].Backward(caches[i], dIn)
		for name, g := range layerGrads {
			grads[layerKey(i, name)] = g
		}
	}
	return dIn, grads
}

func (s *Dense) Params() core.Params {
	params := core.Params{}
	for i, l := range s.layers {
		pl, ok := l.(core.Parameter)
		if !ok {
			continue
		}
		for name, p := range pl.Params() {
			params[layerKey(i, name)] = p
		}
	}
	return params
}

func (s *Dense) SetParams(p core.Params) {
	for i, l := range s.layers {
		pl, ok := l.(core.Parameter)
		if !ok {
			continue
		}
		layerParams := core.Params{}
		for name := range pl.Params() {
			layerParams[name] = p[layerKey(i, name)]
		}
		pl.SetParams(layerParams)
	}
}

func layerKey(index int, name string) string {
	return fmt.Sprintf("%d:%s", index, name)
}
