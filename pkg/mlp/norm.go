package mlp

import (
	"github.com/ppvan/gpt/pkg/core"
)

type BatchNorm struct {
	gamma core.Mat // 1 x d
	beta  core.Mat // 1 x d
}

const epsilon = 0.001

type batchNormCache struct {
	input     core.Mat
	xHat      core.Mat
	xCentered core.Mat
	invStd    core.Mat
}

func NewBatchNorm(features int) *BatchNorm {
	return &BatchNorm{
		gamma: core.NewZeroMat(1, features).Apply(func(float32) float32 {
			return 1
		}),
		beta: core.NewZeroMat(1, features),
	}
}

func (l *BatchNorm) Forward(x core.Mat) (out core.Mat, cache core.Cache) {
	one := core.NewZeroMat(x.Rows, 1).Apply(func(float32) float32 { return 1 })
	oneT := one.Transpose()

	// mean: 1 x d
	mean := oneT.Dot(x).Scale(1 / float32(x.Rows))

	// broadcast mean
	meanBatch := one.Dot(mean)

	// centered input
	xCentered := x.Sub(meanBatch)

	// variance: 1 x d
	variance := oneT.
		Dot(xCentered.Apply(func(v float32) float32 { return v * v })).
		Scale(1 / float32(x.Rows))

	// invStd: 1 x d  <-- DO NOT broadcast here
	invStd := variance.Apply(func(v float32) float32 {
		return 1.0 / core.FastSqrt32(v+epsilon)
	})

	// Broadcast only when needed
	invStdBatch := one.Dot(invStd)

	// normalized input
	xHat := xCentered.Hadamard(invStdBatch)

	gammaBatch := one.Dot(l.gamma)
	betaBatch := one.Dot(l.beta)

	out = gammaBatch.Hadamard(xHat).Add(betaBatch)

	return out, batchNormCache{
		input:     x,
		xHat:      xHat,
		xCentered: xCentered,
		invStd:    invStd, // <-- store 1 x d
	}
}

func (l *BatchNorm) Backward(cache core.Cache, dOut core.Mat) (dIn core.Mat, grads core.Grads) {
	c := cache.(batchNormCache)
	one := core.NewZeroMat(dOut.Rows, 1).Apply(func(f float32) float32 { return 1 })
	oneT := one.Transpose()

	dbeta := oneT.Dot(dOut)
	dgamma := oneT.Dot(dOut.Hadamard(c.xHat))
	dXHat := dOut.Hadamard(one.Dot(l.gamma))

	invStd3 := c.invStd.Apply(func(v float32) float32 {
		return v * v * v
	})
	dVariance := oneT.
		Dot(dXHat.Hadamard(c.xCentered)).
		Hadamard(invStd3).
		Scale(-0.5)

	invStdBatch := one.Dot(c.invStd) // broadcast 1 x d -> N x d
	term1 := dXHat.Hadamard(invStdBatch)

	dMean := oneT.
		Dot(dXHat.Hadamard(invStdBatch)).
		Scale(-1)
	term2 := c.xCentered.
		Hadamard(one.Dot(dVariance)).
		Scale(2.0 / float32(dOut.Rows))
	term3 := one.
		Dot(dMean).
		Scale(1.0 / float32(dOut.Rows))

	dIn = term1.Add(term2).Add(term3)

	grads = core.Grads{
		"gamma": dgamma,
		"beta":  dbeta,
	}
	return dIn, grads
}

func (l *BatchNorm) Params() core.Params {
	return core.Params{
		"gamma": l.gamma,
		"beta":  l.beta,
	}
}

func (l *BatchNorm) SetParams(p core.Params) {
	l.gamma = p["gamma"]
	l.beta = p["beta"]
}
