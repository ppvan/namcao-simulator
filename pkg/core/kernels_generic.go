//go:build !goexperiment.simd || !amd64

package core

func dotTInto(a, b, c []float32, m, n, q int) { dotTIntoScalar(a, b, c, m, n, q) }

func tDotAccInto(a, b, c []float32, m, n, p int) { tDotAccIntoScalar(a, b, c, m, n, p) }

func dotTiled(a, b, result []float32, rowStart, rowEnd, n, p, colStart, colEnd int) {
	dotTiledScalar(a, b, result, rowStart, rowEnd, n, p, colStart, colEnd)
}
