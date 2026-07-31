package core

import (
	"fmt"
	"strconv"
	"strings"
)

const (
	tileM = 32
	tileK = 32
)

type Mat struct {
	weights []float32
	Rows    int
	Columns int
}

func (mat Mat) at(r, c int) int {
	return r*mat.Columns + c
}

func (mat Mat) Get(r, c int) float32 {
	return mat.weights[mat.at(r, c)]
}

func (mat Mat) Set(r, c int, v float32) {
	mat.weights[mat.at(r, c)] = v
}

func (mat Mat) assertSameShape(other Mat, op string) {
	if mat.Rows != other.Rows || mat.Columns != other.Columns {
		panic(fmt.Sprintf("incompatible matrices, can't %s (%v x %v) and (%v x %v)",
			op, mat.Rows, mat.Columns, other.Rows, other.Columns))
	}
}

func (mat Mat) Combine(other Mat, f func(a, b float32) float32) Mat {
	mat.assertSameShape(other, "combine")
	result := make([]float32, len(mat.weights))
	b := other.weights
	for i, v := range mat.weights {
		result[i] = f(v, b[i])
	}
	return Mat{weights: result, Rows: mat.Rows, Columns: mat.Columns}
}

func (mat Mat) Add(other Mat) Mat {
	mat.assertSameShape(other, "add")
	result := make([]float32, len(mat.weights))
	b := other.weights
	for i, v := range mat.weights {
		result[i] = v + b[i]
	}
	return Mat{weights: result, Rows: mat.Rows, Columns: mat.Columns}
}

func (mat Mat) Sub(other Mat) Mat {
	mat.assertSameShape(other, "sub")
	result := make([]float32, len(mat.weights))
	b := other.weights
	for i, v := range mat.weights {
		result[i] = v - b[i]
	}
	return Mat{weights: result, Rows: mat.Rows, Columns: mat.Columns}
}

func (mat Mat) Hadamard(other Mat) Mat {
	mat.assertSameShape(other, "hadamard")
	result := make([]float32, len(mat.weights))
	b := other.weights
	for i, v := range mat.weights {
		result[i] = v * b[i]
	}
	return Mat{weights: result, Rows: mat.Rows, Columns: mat.Columns}
}

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

func (mat Mat) ApplyInPlace(fn func(float32) float32) Mat {
	for i, v := range mat.weights {
		mat.weights[i] = fn(v)
	}
	return mat
}

func (mat Mat) HadamardApply(other Mat, fn func(float32) float32) Mat {
	mat.assertSameShape(other, "hadamard")
	result := make([]float32, len(mat.weights))
	b := other.weights
	for i, v := range mat.weights {
		result[i] = v * fn(b[i])
	}
	return Mat{weights: result, Rows: mat.Rows, Columns: mat.Columns}
}

func (mat Mat) MulAdd(b, c, d Mat) Mat {
	mat.assertSameShape(b, "muladd")
	c.assertSameShape(d, "muladd")
	mat.assertSameShape(c, "muladd")
	result := make([]float32, len(mat.weights))
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
	result := make([]float32, m*p)
	dotInto(mat.weights, other.weights, result, m, n, p)
	return Mat{weights: result, Rows: m, Columns: p}
}

func dotInto(a, b, result []float32, m, n, p int) {
	dotTiled(a, b, result, 0, m, n, p, 0, p)
}

func (mat Mat) DotT(other Mat) Mat {
	if mat.Columns != other.Columns {
		panic(fmt.Sprintf("incompatible matrices for A·Bᵀ: (%v x %v) and (%v x %v)ᵀ",
			mat.Rows, mat.Columns, other.Rows, other.Columns))
	}
	m, n, q := mat.Rows, mat.Columns, other.Rows
	result := make([]float32, m*q)
	dotTInto(mat.weights, other.weights, result, m, n, q)
	return Mat{weights: result, Rows: m, Columns: q}
}

func (mat Mat) TDot(other Mat) Mat {
	if mat.Rows != other.Rows {
		panic(fmt.Sprintf("incompatible matrices for Aᵀ·B: (%v x %v)ᵀ and (%v x %v)",
			mat.Rows, mat.Columns, other.Rows, other.Columns))
	}
	result := NewZeroMat(mat.Columns, other.Columns)
	tDotAccInto(mat.weights, other.weights, result.weights, mat.Rows, mat.Columns, other.Columns)
	return result
}

// TDotAccInto computes dst += A^T . B in place.
func (mat Mat) TDotAccInto(other, dst Mat) Mat {
	tDotAccInto(mat.weights, other.weights, dst.weights, mat.Rows, mat.Columns, other.Columns)
	return dst
}

// SumRowsInto accumulates the column sums of mat into dst (1,C).
func (mat Mat) SumRowsInto(dst Mat) Mat {
	cols := mat.Columns
	for i := 0; i < mat.Rows; i++ {
		row := mat.weights[i*cols : (i+1)*cols]
		for j, v := range row {
			dst.weights[j] += v
		}
	}
	return dst
}

func (mat Mat) Transpose() Mat {
	result := make([]float32, len(mat.weights))
	rows, cols := mat.Rows, mat.Columns
	src := mat.weights

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

func (mat Mat) Apply(fn func(float32) float32) Mat {
	result := make([]float32, len(mat.weights))
	for i, v := range mat.weights {
		result[i] = fn(v)
	}
	return Mat{weights: result, Rows: mat.Rows, Columns: mat.Columns}
}

func (mat Mat) Scale(s float32) Mat {
	result := make([]float32, len(mat.weights))
	for i, v := range mat.weights {
		result[i] = v * s
	}
	return Mat{weights: result, Rows: mat.Rows, Columns: mat.Columns}
}

func (mat Mat) SumRows() Mat {
	cols := mat.Columns
	result := make([]float32, cols)
	for i := 0; i < mat.Rows; i++ {
		row := mat.weights[i*cols : (i+1)*cols]
		for j, v := range row {
			result[j] += v
		}
	}
	return Mat{weights: result, Rows: 1, Columns: cols}
}

func (mat Mat) Sum() float32 {
	var total float32
	for _, v := range mat.weights {
		total += v
	}
	return total
}

func (mat Mat) Mean() float32 {
	return mat.Sum() / float32(len(mat.weights))
}

func NewZeroMat(row, column int) Mat {
	return Mat{
		weights: make([]float32, row*column),
		Rows:    row,
		Columns: column,
	}
}

func NewMat(row, column int, data []float32) Mat {
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

func (mat Mat) SliceCols(start, end int) Mat {
	if start < 0 || end > mat.Columns || start > end {
		panic("invalid column slice range")
	}
	cols := end - start
	result := make([]float32, mat.Rows*cols)
	for i := 0; i < mat.Rows; i++ {
		copy(result[i*cols:(i+1)*cols], mat.weights[i*mat.Columns+start:i*mat.Columns+end])
	}
	return Mat{weights: result, Rows: mat.Rows, Columns: cols}
}

func HStack(ms ...Mat) Mat {
	if len(ms) == 0 {
		panic("HStack requires at least one matrix")
	}
	rows := ms[0].Rows
	total := 0
	for _, m := range ms {
		if m.Rows != rows {
			panic(fmt.Sprintf("HStack row mismatch: %v vs %v", rows, m.Rows))
		}
		total += m.Columns
	}
	result := make([]float32, rows*total)
	off := 0
	for _, m := range ms {
		for i := 0; i < rows; i++ {
			copy(result[i*total+off:i*total+off+m.Columns], m.weights[i*m.Columns:(i+1)*m.Columns])
		}
		off += m.Columns
	}
	return Mat{weights: result, Rows: rows, Columns: total}
}

func (mat Mat) ApplyColsInPlace(start, end int, fn func(float32) float32) Mat {
	if start < 0 || end > mat.Columns || start > end {
		panic("invalid column range")
	}
	for i := 0; i < mat.Rows; i++ {
		row := mat.weights[i*mat.Columns+start : i*mat.Columns+end]
		for j, v := range row {
			row[j] = fn(v)
		}
	}
	return mat
}

func (mat Mat) String() string {
	var sb strings.Builder
	for i := 0; i < mat.Rows; i++ {
		sb.WriteString("[")
		for j := 0; j < mat.Columns; j++ {
			if j > 0 {
				sb.WriteString(", ")
			}
			sb.WriteString(strconv.FormatFloat(float64(mat.Get(i, j)), 'f', 2, 32))
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
	result := make([]float32, rows*cols)
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
	result := make([]float32, len(mat.weights))
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

func (mat Mat) ScaleInPlace(s float32) Mat {
	for i := range mat.weights {
		mat.weights[i] *= s
	}
	return mat
}

// SumSquares returns the sum of squared elements [performance]
func (mat Mat) SumSquares() float32 {
	var s0, s1 float32
	w := mat.weights
	i := 0
	for ; i+1 < len(w); i += 2 {
		s0 += w[i] * w[i]
		s1 += w[i+1] * w[i+1]
	}
	for ; i < len(w); i++ {
		s0 += w[i] * w[i]
	}
	return s0 + s1
}

// AdamStep applies one fused Adam update to param in place[performance].
func AdamStep(param, grad, m, v Mat, rate, b1, b2, eps, corr1, corr2 float32) {
	p, g, mw, vw := param.weights, grad.weights, m.weights, v.weights
	ob1, ob2 := 1-b1, 1-b2
	for i := range p {
		gi := g[i]
		mi := b1*mw[i] + ob1*gi
		vi := b2*vw[i] + ob2*gi*gi
		mw[i], vw[i] = mi, vi
		p[i] -= rate * (mi / corr1) / (FastSqrt32(vi/corr2) + eps)
	}
}
