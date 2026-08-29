package core

type Cache any

type Grads map[string]Mat

type Params map[string]Mat

type Parameter interface {
	Params() Params   // read current parameter values
	SetParams(Params) // write updated parameter values back
}

type Optimizer interface {
	Update(name string, param, grad Mat) Mat
}

type Gradient struct {
	Rate float32
}

func NewGradient(rate float32) *Gradient {
	return &Gradient{Rate: rate}
}

func (g *Gradient) Update(name string, param Mat, grad Mat) Mat {
	return param.Sub(grad.Scale(g.Rate))
}

type Momentum struct {
	gradAvg map[string]Mat

	Rate float32
	Beta float32
}

func NewMomentum(rate float32, beta float32) *Momentum {
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

	Rate    float32
	Beta    float32
	Epsilon float32
}

func NewRMSProp(rate, beta, eps float32) *RMSProp {
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
		Apply(FastSqrt32).
		Apply(func(v float32) float32 {
			return 1 / (v + r.Epsilon)
		})

	update := grad.Hadamard(term)

	return param.Sub(update.Scale(r.Rate))
}

type Adam struct {
	m       map[string]Mat
	v       map[string]Mat
	t       map[string]int
	Rate    float32
	Beta1   float32
	Beta2   float32
	Epsilon float32
}

func NewAdam(rate, beta1, beta2, eps float32) *Adam {
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

func (a *Adam) M() map[string]Mat { return a.m }

func (a *Adam) V() map[string]Mat { return a.v }

func (a *Adam) Steps() map[string]int { return a.t }

func (a *Adam) SetState(m map[string]Mat, v map[string]Mat, t map[string]int) {
	a.m = m
	a.v = v
	a.t = t
}

func (a *Adam) Update(name string, param, grad Mat) Mat {
	m, ok := a.m[name]
	if !ok {
		m = NewZeroMat(grad.Rows, grad.Columns)
		a.m[name] = m
	}
	v, ok := a.v[name]
	if !ok {
		v = NewZeroMat(grad.Rows, grad.Columns)
		a.v[name] = v
	}

	a.t[name]++
	t := a.t[name]

	corr1 := 1 - PowIntApprox32(a.Beta1, t)
	corr2 := 1 - PowIntApprox32(a.Beta2, t)

	// One fused pass over the parameter instead of twelve allocating ones.
	// m and v are updated in place and kept across steps.
	AdamStep(param, grad, m, v, a.Rate, a.Beta1, a.Beta2, a.Epsilon, corr1, corr2)
	return param
}
