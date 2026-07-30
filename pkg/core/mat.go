package core

import (
	"fmt"
	"runtime"
	"strconv"
	"strings"
	"sync"
)

const (
	tileM = 32
	tileK = 32

	parallelThreshold = 128
)

type Mat struct {
	weights []float64
	Rows    int
	Columns int
}

func (mat Mat) at(r, c int) int {
	return r*mat.Columns + c
}

func (mat Mat) Get(r, c int) float64 {
	return mat.weights[mat.at(r, c)]
}

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

func (mat Mat) HadamardApply(other Mat, fn func(float64) float64) Mat {
	mat.assertSameShape(other, "hadamard")
	result := make([]float64, len(mat.weights))
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

func dotInto(a, b, result []float64, m, n, p int) {
	numWorkers := runtime.GOMAXPROCS(0)
	if numWorkers == 1 || m*n*p <= parallelThreshold {
		dotTiled(a, b, result, 0, m, n, p, 0, p)
		return
	}

	var wg sync.WaitGroup
	if m >= numWorkers*4 || m >= p {

		chunk := (m + numWorkers - 1) / numWorkers
		for rs := 0; rs < m; rs += chunk {
			re := min(rs+chunk, m)
			wg.Add(1)
			go func(rs, re int) {
				defer wg.Done()
				dotTiled(a, b, result, rs, re, n, p, 0, p)
			}(rs, re)
		}
	} else {

		chunk := (p + numWorkers - 1) / numWorkers
		chunk = (chunk + 7) / 8 * 8
		for cs := 0; cs < p; cs += chunk {
			ce := min(cs+chunk, p)
			wg.Add(1)
			go func(cs, ce int) {
				defer wg.Done()
				dotTiled(a, b, result, 0, m, n, p, cs, ce)
			}(cs, ce)
		}
	}
	wg.Wait()
}

func dotTiled(a, b, result []float64, rowStart, rowEnd, n, p, colStart, colEnd int) {
	for ii := rowStart; ii < rowEnd; ii += tileM {
		iMax := min(ii+tileM, rowEnd)
		for kk := 0; kk < n; kk += tileK {
			kMax := min(kk+tileK, n)
			for i := ii; i < iMax; i++ {
				aRow := i * n
				c := result[i*p+colStart : i*p+colEnd]

				k := kk
				for ; k+3 < kMax; k += 4 {
					a0 := a[aRow+k]
					a1 := a[aRow+k+1]
					a2 := a[aRow+k+2]
					a3 := a[aRow+k+3]
					b0 := b[(k+0)*p+colStart : (k+0)*p+colEnd]
					b1 := b[(k+1)*p+colStart : (k+1)*p+colEnd]
					b2 := b[(k+2)*p+colStart : (k+2)*p+colEnd]
					b3 := b[(k+3)*p+colStart : (k+3)*p+colEnd]
					for j := range c {
						c[j] += a0*b0[j] + a1*b1[j] + a2*b2[j] + a3*b3[j]
					}
				}
				for ; k < kMax; k++ {
					aik := a[aRow+k]
					bb := b[k*p+colStart : k*p+colEnd]
					for j := range c {
						c[j] += aik * bb[j]
					}
				}
			}
		}
	}
}

func (mat Mat) DotT(other Mat) Mat {
	if mat.Columns != other.Columns {
		panic(fmt.Sprintf("incompatible matrices for A·Bᵀ: (%v x %v) and (%v x %v)ᵀ",
			mat.Rows, mat.Columns, other.Rows, other.Columns))
	}
	return mat.Dot(other.Transpose())
}

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

func (mat Mat) SliceCols(start, end int) Mat {
	if start < 0 || end > mat.Columns || start > end {
		panic("invalid column slice range")
	}
	cols := end - start
	result := make([]float64, mat.Rows*cols)
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
	result := make([]float64, rows*total)
	off := 0
	for _, m := range ms {
		for i := 0; i < rows; i++ {
			copy(result[i*total+off:i*total+off+m.Columns], m.weights[i*m.Columns:(i+1)*m.Columns])
		}
		off += m.Columns
	}
	return Mat{weights: result, Rows: rows, Columns: total}
}

func (mat Mat) ApplyColsInPlace(start, end int, fn func(float64) float64) Mat {
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
