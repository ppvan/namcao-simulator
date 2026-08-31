package core

import "math/rand/v2"

func randomUniform(min, max float32) float32 {
	return min + rand.Float32()*(max-min)
}

func XavierMat(row, column int) Mat {
	fanIn := float32(row)
	fanOut := float32(column)

	a := FastSqrt32(6.0 / (fanIn + fanOut))

	return NewZeroMat(row, column).Apply(func(f float32) float32 {
		return randomUniform(-a, a)
	})
}

func HeMat(row, column int) Mat {
	fanIn := float32(row)

	a := FastSqrt32(2.0 / fanIn)

	return NewZeroMat(row, column).Apply(func(f float32) float32 {
		return randomUniform(-a, a)
	})
}
