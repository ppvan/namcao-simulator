package core

import "math"

// A opaque type saved in forward and useful for backward pass. Extact type depend on the operation
type Cache any

// Gradient map return by backward pass
type Grads map[string]Mat

// Learnable param of a layer, exactly like Grads, just easy to reason in optimizer context.
type Params map[string]Mat

// A interface so Optimizer could update learn-able parameters without knowing about the layer.
type Parameter interface {
	Params() Params   // read current parameter values
	SetParams(Params) // write updated parameter values back
}

type Optimizer interface {
	Update(name string, param, grad Mat) Mat
}

type Gradient struct {
	Rate float64
}

func NewGradient(rate float64) *Gradient {
	return &Gradient{Rate: rate}
}

func (g *Gradient) Update(name string, param Mat, grad Mat) Mat {
	return param.Sub(grad.Scale(g.Rate))
}

type Momentum struct {
	gradAvg map[string]Mat

	Rate float64
	Beta float64
}

func NewMomentum(rate float64, beta float64) *Momentum {
	return &Momentum{Rate: rate, Beta: beta, gradAvg: make(map[string]Mat)}
}

func (g *Momentum) Update(name string, param Mat, grad Mat) Mat {
	v := g.gradAvg[name]
	if v.Rows == 0 {
		v = NewZeroMat(grad.Rows, grad.Columns)
	}

	v = v.Scale(g.Beta).
		Add(grad.Scale(1 - g.Beta))

	g.gradAvg[name] = v

	return param.Sub(v.Scale(g.Rate))
}

type RMSProp struct {
	avgSq map[string]Mat

	Rate    float64
	Beta    float64
	Epsilon float64
}

func NewRMSProp(rate, beta, eps float64) *RMSProp {
	return &RMSProp{
		avgSq:   make(map[string]Mat),
		Rate:    rate,
		Beta:    beta,
		Epsilon: eps,
	}
}

func (r *RMSProp) Update(name string, param, grad Mat) Mat {
	s, ok := r.avgSq[name]
	if !ok {
		s = NewZeroMat(grad.Rows, grad.Columns)
	}

	// grad²
	grad2 := grad.Hadamard(grad)

	// running average
	s = s.Scale(r.Beta).
		Add(grad2.Scale(1 - r.Beta))

	r.avgSq[name] = s

	// sqrt(s) + eps
	term := s.
		Apply(math.Sqrt).
		Apply(func(v float64) float64 {
			return 1 / (v + r.Epsilon)
		})

	update := grad.Hadamard(term)

	return param.Sub(update.Scale(r.Rate))
}

type Adam struct {
	m       map[string]Mat
	v       map[string]Mat
	t       map[string]int
	Rate    float64
	Beta1   float64
	Beta2   float64
	Epsilon float64
}

func NewAdam(rate, beta1, beta2, eps float64) *Adam {
	return &Adam{
		m:       make(map[string]Mat),
		v:       make(map[string]Mat),
		t:       make(map[string]int),
		Rate:    rate,
		Beta1:   beta1,
		Beta2:   beta2,
		Epsilon: eps,
	}
}

func (a *Adam) Update(name string, param, grad Mat) Mat {
	m, ok := a.m[name]
	if !ok {
		m = NewZeroMat(grad.Rows, grad.Columns)
	}
	v, ok := a.v[name]
	if !ok {
		v = NewZeroMat(grad.Rows, grad.Columns)
	}

	a.t[name]++
	t := a.t[name]

	// first moment (mean of gradient)
	m = m.Scale(a.Beta1).
		Add(grad.Scale(1 - a.Beta1))

	// second moment (uncentered variance of gradient)
	grad2 := grad.Hadamard(grad)
	v = v.Scale(a.Beta2).
		Add(grad2.Scale(1 - a.Beta2))

	a.m[name] = m
	a.v[name] = v

	// bias correction
	biasCorr1 := 1 - math.Pow(a.Beta1, float64(t))
	biasCorr2 := 1 - math.Pow(a.Beta2, float64(t))

	mHat := m.Scale(1 / biasCorr1)
	vHat := v.Scale(1 / biasCorr2)

	// sqrt(vHat) + eps, inverted
	denom := vHat.Apply(func(x float64) float64 {
		return 1 / (math.Sqrt(x) + a.Epsilon)
	})

	update := mHat.Hadamard(denom)
	return param.Sub(update.Scale(a.Rate))
}
