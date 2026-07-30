package core

import (
	"fmt"
	"runtime"
	"strconv"
	"strings"
	"sync"
)

const (
	tileM = 64  // rows tile
	tileK = 256 // depth tile

	// Below this many multiply-adds, a Dot runs on the calling
	// goroutine: spawning workers costs more than the math.
	parallelThreshold = 64 * 64 * 64
)

type Mat struct {
	weights []float64
	Rows    int
	Columns int
}

// at returns the flat index for element (r, c).
func (mat Mat) at(r, c int) int {
	return r*mat.Columns + c
}

// Get returns the element at row r, column c.
func (mat Mat) Get(r, c int) float64 {
	return mat.weights[mat.at(r, c)]
}

// Set assigns the element at row r, column c.
func (mat Mat) Set(r, c int, v float64) {
	mat.weights[mat.at(r, c)] = v
}

func (mat Mat) assertSameShape(other Mat, op string) {
	if mat.Rows != other.Rows || mat.Columns != other.Columns {
		panic(fmt.Sprintf("incompatible matrices, can't %s (%v x %v) and (%v x %v)",
			op, mat.Rows, mat.Columns, other.Rows, other.Columns))
	}
}

func (mat Mat) Combine(other Mat, f func(a, b float64) float64) Mat {
	mat.assertSameShape(other, "combine")
	result := make([]float64, len(mat.weights))
	b := other.weights
	for i, v := range mat.weights {
		result[i] = f(v, b[i])
	}
	return Mat{weights: result, Rows: mat.Rows, Columns: mat.Columns}
}

// Add, Sub, Hadamard use direct loops instead of Combine + closure:
// the indirect call per element blocks inlining and vectorization.

func (mat Mat) Add(other Mat) Mat {
	mat.assertSameShape(other, "add")
	result := make([]float64, len(mat.weights))
	b := other.weights
	for i, v := range mat.weights {
		result[i] = v + b[i]
	}
	return Mat{weights: result, Rows: mat.Rows, Columns: mat.Columns}
}

func (mat Mat) Sub(other Mat) Mat {
	mat.assertSameShape(other, "sub")
	result := make([]float64, len(mat.weights))
	b := other.weights
	for i, v := range mat.weights {
		result[i] = v - b[i]
	}
	return Mat{weights: result, Rows: mat.Rows, Columns: mat.Columns}
}

func (mat Mat) Hadamard(other Mat) Mat {
	mat.assertSameShape(other, "hadamard")
	result := make([]float64, len(mat.weights))
	b := other.weights
	for i, v := range mat.weights {
		result[i] = v * b[i]
	}
	return Mat{weights: result, Rows: mat.Rows, Columns: mat.Columns}
}

// In-place variants: mutate the receiver's backing array and return it
// for chaining. Only use on freshly allocated intermediates (e.g. the
// result of a Dot/Add/Apply), never on values that are still referenced
// elsewhere, such as cached activations or parameters.

func (mat Mat) AddInPlace(other Mat) Mat {
	mat.assertSameShape(other, "add")
	b := other.weights
	for i := range mat.weights {
		mat.weights[i] += b[i]
	}
	return mat
}

func (mat Mat) SubInPlace(other Mat) Mat {
	mat.assertSameShape(other, "sub")
	b := other.weights
	for i := range mat.weights {
		mat.weights[i] -= b[i]
	}
	return mat
}

func (mat Mat) HadamardInPlace(other Mat) Mat {
	mat.assertSameShape(other, "hadamard")
	b := other.weights
	for i := range mat.weights {
		mat.weights[i] *= b[i]
	}
	return mat
}

func (mat Mat) ApplyInPlace(fn func(float64) float64) Mat {
	for i, v := range mat.weights {
		mat.weights[i] = fn(v)
	}
	return mat
}

// HadamardApply returns mat ⊙ fn(other) in a single pass, without
// materializing fn(other). Handy for gradient chains like
// df ⊙ dSigmoid(f).
func (mat Mat) HadamardApply(other Mat, fn func(float64) float64) Mat {
	mat.assertSameShape(other, "hadamard")
	result := make([]float64, len(mat.weights))
	b := other.weights
	for i, v := range mat.weights {
		result[i] = v * fn(b[i])
	}
	return Mat{weights: result, Rows: mat.Rows, Columns: mat.Columns}
}

// MulAdd returns mat⊙b + c⊙d in a single pass (e.g. the LSTM cell
// update f⊙cPrev + i⊙g).
func (mat Mat) MulAdd(b, c, d Mat) Mat {
	mat.assertSameShape(b, "muladd")
	c.assertSameShape(d, "muladd")
	mat.assertSameShape(c, "muladd")
	result := make([]float64, len(mat.weights))
	bw, cw, dw := b.weights, c.weights, d.weights
	for i, v := range mat.weights {
		result[i] = v*bw[i] + cw[i]*dw[i]
	}
	return Mat{weights: result, Rows: mat.Rows, Columns: mat.Columns}
}

func (mat Mat) Dot(other Mat) Mat {
	if mat.Columns != other.Rows {
		msg := fmt.Sprintf("incompatible matrices: (%v x %v) and (%v x %v)", mat.Rows, mat.Columns, other.Rows, other.Columns)
		panic(msg)
	}
	m, n, p := mat.Rows, mat.Columns, other.Columns
	result := make([]float64, m*p)
	dotInto(mat.weights, other.weights, result, m, n, p)
	return Mat{weights: result, Rows: m, Columns: p}
}

// dotInto computes result = a(m x n) · b(n x p), parallelizing across
// row chunks when the problem is large enough to amortize goroutine
// startup.
func dotInto(a, b, result []float64, m, n, p int) {
	if m*n*p <= parallelThreshold {
		dotTiled(a, b, result, 0, m, n, p)
		return
	}

	numWorkers := runtime.GOMAXPROCS(0)
	// split rows [0, m) into numWorkers chunks, rounded up to a
	// multiple of tileM so tiles aren't split across workers
	chunk := (m + numWorkers - 1) / numWorkers
	chunk = (chunk + tileM - 1) / tileM * tileM

	var wg sync.WaitGroup
	for rowStart := 0; rowStart < m; rowStart += chunk {
		rowEnd := min(rowStart+chunk, m)
		wg.Add(1)
		go func(rowStart, rowEnd int) {
			defer wg.Done()
			dotTiled(a, b, result, rowStart, rowEnd, n, p)
		}(rowStart, rowEnd)
	}
	wg.Wait()
}

// dotTiled computes result[rowStart:rowEnd, :] using tiling on i and k,
// with the k loop register-blocked by 4: each pass over the output row
// folds in four rows of b, so result traffic drops 4x versus a plain
// saxpy loop.
func dotTiled(a, b, result []float64, rowStart, rowEnd, n, p int) {
	for ii := rowStart; ii < rowEnd; ii += tileM {
		iMax := min(ii+tileM, rowEnd)
		for kk := 0; kk < n; kk += tileK {
			kMax := min(kk+tileK, n)
			for i := ii; i < iMax; i++ {
				aRow := i * n
				c := result[i*p : i*p+p]

				k := kk
				for ; k+3 < kMax; k += 4 {
					a0 := a[aRow+k]
					a1 := a[aRow+k+1]
					a2 := a[aRow+k+2]
					a3 := a[aRow+k+3]
					b0 := b[(k+0)*p : (k+0)*p+p]
					b1 := b[(k+1)*p : (k+1)*p+p]
					b2 := b[(k+2)*p : (k+2)*p+p]
					b3 := b[(k+3)*p : (k+3)*p+p]
					for j := range c {
						c[j] += a0*b0[j] + a1*b1[j] + a2*b2[j] + a3*b3[j]
					}
				}
				for ; k < kMax; k++ {
					aik := a[aRow+k]
					bb := b[k*p : k*p+p]
					for j := range c {
						c[j] += aik * bb[j]
					}
				}
			}
		}
	}
}

// DotT returns mat Â· otheráµ. Benchmarks showed a blocked transpose
// followed by the tiled Dot kernel beats a direct row-dot kernel, so
// this is a convenience wrapper that keeps gradient call sites clean:
// dz.DotT(W) instead of dz.Dot(W.Transpose()).
func (mat Mat) DotT(other Mat) Mat {
	if mat.Columns != other.Columns {
		panic(fmt.Sprintf("incompatible matrices for A·Bᵀ: (%v x %v) and (%v x %v)ᵀ",
			mat.Rows, mat.Columns, other.Rows, other.Columns))
	}
	return mat.Dot(other.Transpose())
}

// TDot returns matᵀ · other, e.g. weight gradients x.TDot(dz).
func (mat Mat) TDot(other Mat) Mat {
	if mat.Rows != other.Rows {
		panic(fmt.Sprintf("incompatible matrices for Aᵀ·B: (%v x %v)ᵀ and (%v x %v)",
			mat.Rows, mat.Columns, other.Rows, other.Columns))
	}
	return mat.Transpose().Dot(other)
}

func (mat Mat) Transpose() Mat {
	result := make([]float64, len(mat.weights))
	rows, cols := mat.Rows, mat.Columns
	src := mat.weights

	// blocked transpose: keeps both source rows and destination rows
	// resident in cache instead of striding through all of result
	const blk = 32
	for ii := 0; ii < rows; ii += blk {
		iMax := min(ii+blk, rows)
		for jj := 0; jj < cols; jj += blk {
			jMax := min(jj+blk, cols)
			for i := ii; i < iMax; i++ {
				rowOffset := i * cols
				for j := jj; j < jMax; j++ {
					result[j*rows+i] = src[rowOffset+j]
				}
			}
		}
	}

	return Mat{weights: result, Rows: mat.Columns, Columns: mat.Rows}
}

func (mat Mat) Apply(fn func(float64) float64) Mat {
	result := make([]float64, len(mat.weights))
	for i, v := range mat.weights {
		result[i] = fn(v)
	}
	return Mat{weights: result, Rows: mat.Rows, Columns: mat.Columns}
}

func (mat Mat) Scale(s float64) Mat {
	result := make([]float64, len(mat.weights))
	for i, v := range mat.weights {
		result[i] = v * s
	}
	return Mat{weights: result, Rows: mat.Rows, Columns: mat.Columns}
}

func (mat Mat) SumRows() Mat {
	cols := mat.Columns
	result := make([]float64, cols)
	for i := 0; i < mat.Rows; i++ {
		row := mat.weights[i*cols : (i+1)*cols]
		for j, v := range row {
			result[j] += v
		}
	}
	return Mat{weights: result, Rows: 1, Columns: cols}
}

func (mat Mat) Sum() float64 {
	var total float64
	for _, v := range mat.weights {
		total += v
	}
	return total
}

func (mat Mat) Mean() float64 {
	return mat.Sum() / float64(len(mat.weights))
}

func NewZeroMat(row, column int) Mat {
	return Mat{
		weights: make([]float64, row*column),
		Rows:    row,
		Columns: column,
	}
}

func NewMat(row, column int, data []float64) Mat {
	return Mat{
		weights: data,
		Rows:    row,
		Columns: column,
	}
}

func (mat Mat) Slice(start, end int) Mat {
	if start < 0 || end > mat.Rows || start > end {
		panic("invalid slice range")
	}

	rows := end - start
	result := NewZeroMat(rows, mat.Columns)

	copy(
		result.weights,
		mat.weights[start*mat.Columns:end*mat.Columns],
	)

	return result
}

func (mat Mat) String() string {
	var sb strings.Builder
	for i := 0; i < mat.Rows; i++ {
		sb.WriteString("[")
		for j := 0; j < mat.Columns; j++ {
			if j > 0 {
				sb.WriteString(", ")
			}
			sb.WriteString(strconv.FormatFloat(mat.Get(i, j), 'f', 2, 64))
		}
		sb.WriteString("]")
		if i < mat.Rows-1 {
			sb.WriteString("\n")
		}
	}
	return sb.String()
}

func (mat Mat) Broadcast(rows int) Mat {
	if mat.Rows != 1 {
		panic(fmt.Sprintf("Broadcast requires a (1, C) matrix, got (%v x %v)", mat.Rows, mat.Columns))
	}
	if rows == 1 {
		return mat
	}
	cols := mat.Columns
	result := make([]float64, rows*cols)
	for i := 0; i < rows; i++ {
		copy(result[i*cols:(i+1)*cols], mat.weights)
	}
	return Mat{weights: result, Rows: rows, Columns: cols}
}

func (mat Mat) AddBias(bias Mat) Mat {
	if bias.Rows != 1 || bias.Columns != mat.Columns {
		if mat.Rows == bias.Rows && mat.Columns == bias.Columns {
			return mat.Add(bias)
		}
		panic(fmt.Sprintf("AddBias requires a (1, %v) bias, got (%v x %v)", mat.Columns, bias.Rows, bias.Columns))
	}
	result := make([]float64, len(mat.weights))
	cols := mat.Columns
	bw := bias.weights
	for i := 0; i < mat.Rows; i++ {
		row := mat.weights[i*cols : (i+1)*cols]
		out := result[i*cols : (i+1)*cols]
		for j, v := range row {
			out[j] = v + bw[j]
		}
	}
	return Mat{weights: result, Rows: mat.Rows, Columns: cols}
}

func (mat Mat) AddBiasInPlace(bias Mat) Mat {
	if bias.Rows != 1 || bias.Columns != mat.Columns {
		panic(fmt.Sprintf("AddBias requires a (1, %v) bias, got (%v x %v)", mat.Columns, bias.Rows, bias.Columns))
	}
	cols := mat.Columns
	bw := bias.weights
	for i := 0; i < mat.Rows; i++ {
		row := mat.weights[i*cols : (i+1)*cols]
		for j := range row {
			row[j] += bw[j]
		}
	}
	return mat
}
