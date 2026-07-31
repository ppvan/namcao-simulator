//go:build !goexperiment.simd || !amd64

package core

// Fallback dispatch: without GOEXPERIMENT=simd (or off amd64) the kernels
// are the scalar ones. Keeping the indirection here rather than in mat.go
// means callers never need a build tag.

func dotTInto(a, b, c []float32, m, n, q int) { dotTIntoScalar(a, b, c, m, n, q) }

func tDotAccInto(a, b, c []float32, m, n, p int) { tDotAccIntoScalar(a, b, c, m, n, p) }

func dotTiled(a, b, result []float32, rowStart, rowEnd, n, p, colStart, colEnd int) {
	dotTiledScalar(a, b, result, rowStart, rowEnd, n, p, colStart, colEnd)
}
