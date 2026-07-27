package nn

type Grads map[string]Mat
type Params map[string]Mat

// Cache holds whatever a Forward pass needs to remember for its
// matching Backward pass
type Cache any

type Layer interface {
	Forward(x Mat) (out Mat, cache Cache)
	Backward(cache Cache, dOut Mat) (dIn Mat, grads Grads)
}

type ParamLayer interface {
	Layer
	Params() Params   // read current parameter values
	SetParams(Params) // write updated parameter values back
}

type Sequential interface {
	Layer
	ParamLayer
	Add(l Layer)
}

type RecurrentCell interface {
	Step(x, hPrev Mat) (hNext Mat, cache Cache)
	BackwardStep(cache Cache, dhNext Mat) (dx, dhPrev Mat, grads Grads)
	ParamLayer
}

type Sequence interface {
	ForwardSequence(xs []Mat, h0 Mat) (outs []Mat, hFinal Mat)
	BackwardSequence(dOuts []Mat) Grads
	ParamLayer
}

type Optimizer interface {
	Update(name string, param, grad Mat) Mat
}

type GradClipper interface {
	Clip(grads Grads) Grads
}

type LossFunction interface {
	Forward(pred, target Mat) (Mat, error)
	Backward(pred, target Mat) (Mat, error)
}

type Trainable interface {
	ParamLayer
}

type Data struct {
	X, Y Mat
}

type TrainMetrics struct {
	Epoch int
	Loss  float64
}

type Prediction struct {
	Class int
	Probs []float64
}
