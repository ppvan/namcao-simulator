package core

import (
	"math"
	"math/rand"
)

func randomUniform(min, max float64) float64 {
	return min + rand.Float64()*(max-min)
}

func XavierMat(row, column int) Mat {
	fanIn := float64(row)
	fanOut := float64(column)

	a := math.Sqrt(6.0 / (fanIn + fanOut))

	return NewZeroMat(row, column).Apply(func(f float64) float64 {
		return randomUniform(-a, a)
	})
}

func HeMat(row, column int) Mat {
	fanIn := float64(row)

	a := math.Sqrt(2.0 / fanIn)

	return NewZeroMat(row, column).Apply(func(f float64) float64 {
		return randomUniform(-a, a)
	})
}
