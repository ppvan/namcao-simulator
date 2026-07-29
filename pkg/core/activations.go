package core

import "math"

type ReLU struct{}
type reluCache struct {
	x Mat
}

func NewReLU() *ReLU {
	return &ReLU{}
}

func (r *ReLU) Forward(x Mat) (out Mat, cache Cache) {
	out = x.Apply(func(v float64) float64 {
		if v > 0 {
			return v
		}
		return 0
	})
	return out, reluCache{x: x}
}

func (r *ReLU) Backward(cache Cache, dOut Mat) (dIn Mat, grads Grads) {
	c := cache.(reluCache)
	dIn = dOut.Hadamard(
		c.x.Apply(func(v float64) float64 {
			if v > 0 {
				return 1
			}
			return 0
		}),
	)
	return dIn, Grads{}
}

type LeakyReLU struct {
	Alpha float64
}
type leakyReluCache struct {
	x     Mat
	alpha float64
}

func NewLeakyReLU(alpha float64) *LeakyReLU {
	return &LeakyReLU{Alpha: alpha}
}

func (l *LeakyReLU) Forward(x Mat) (out Mat, cache Cache) {
	out = x.Apply(func(v float64) float64 {
		if v > 0 {
			return v
		}
		return l.Alpha * v
	})
	return out, leakyReluCache{x: x, alpha: l.Alpha}
}

func (l *LeakyReLU) Backward(cache Cache, dOut Mat) (dIn Mat, grads Grads) {
	c := cache.(leakyReluCache)
	dIn = dOut.Hadamard(
		c.x.Apply(func(v float64) float64 {
			if v > 0 {
				return 1
			}
			return c.alpha
		}),
	)
	return dIn, Grads{}
}

type Sigmoid struct{}

type sigmoidCache struct {
	out Mat
}

func NewSigmoid() *Sigmoid {
	return &Sigmoid{}
}

func (s *Sigmoid) Forward(x Mat) (out Mat, cache Cache) {
	out = x.Apply(func(v float64) float64 {
		return 1.0 / (1.0 + math.Exp(-v))
	})
	return out, sigmoidCache{out: out}
}

func (s *Sigmoid) Backward(cache Cache, dOut Mat) (dIn Mat, grads Grads) {
	c := cache.(sigmoidCache)

	dIn = dOut.Hadamard(
		c.out.Apply(func(y float64) float64 {
			return y * (1 - y)
		}),
	)

	return dIn, Grads{}
}
