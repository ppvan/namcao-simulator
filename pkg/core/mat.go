package core

import (
	"fmt"
	"strconv"
	"strings"
	"sync"
)

const (
	tileM = 64 // rows tile
	tileK = 64 // depth tile
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

func (mat Mat) Combine(other Mat, f func(a, b float64) float64) Mat {
	if mat.Rows != other.Rows || mat.Columns != other.Columns {
		message := fmt.Sprintf("In-compatible matrix, can't combine (%v x %v) * (%v x %v)", mat.Rows, mat.Columns, other.Rows, other.Columns)
		panic(message)
	}
	result := make([]float64, len(mat.weights))
	for i := range mat.weights {
		result[i] = f(mat.weights[i], other.weights[i])
	}
	return Mat{
		weights: result,
		Rows:    mat.Rows,
		Columns: mat.Columns,
	}
}

func (mat Mat) Add(other Mat) Mat {
	return mat.Combine(other, func(a, b float64) float64 { return a + b })
}

func (mat Mat) Sub(other Mat) Mat {
	return mat.Combine(other, func(a, b float64) float64 { return a - b })
}

func (mat Mat) Hadamard(other Mat) Mat {
	return mat.Combine(other, func(a, b float64) float64 { return a * b })
}

func (mat Mat) Dot(other Mat) Mat {
	if mat.Columns != other.Rows {
		msg := fmt.Sprintf("incompatible matrices: (%v x %v) and (%v x %v)", mat.Rows, mat.Columns, other.Rows, other.Columns)
		panic(msg)
	}
	m, n, p := mat.Rows, mat.Columns, other.Columns
	result := make([]float64, m*p)
	a := mat.weights
	b := other.weights

	const numWorkers = 4
	// split rows [0, m) into numWorkers chunks
	chunk := (m + numWorkers - 1) / numWorkers

	var wg sync.WaitGroup
	for w := range numWorkers {
		rowStart := w * chunk
		rowEnd := min(rowStart+chunk, m)
		if rowStart >= rowEnd {
			continue
		}

		wg.Add(1)
		go func(rowStart, rowEnd int) {
			defer wg.Done()
			dotTiled(a, b, result, rowStart, rowEnd, n, p)
		}(rowStart, rowEnd)
	}
	wg.Wait()

	return Mat{
		weights: result,
		Rows:    m,
		Columns: p,
	}
}

// dotTiled computes result[rowStart:rowEnd, :] using tiling on i and k.
func dotTiled(a, b, result []float64, rowStart, rowEnd, n, p int) {
	for ii := rowStart; ii < rowEnd; ii += tileM {
		iMax := ii + tileM
		if iMax > rowEnd {
			iMax = rowEnd
		}
		for kk := 0; kk < n; kk += tileK {
			kMax := kk + tileK
			if kMax > n {
				kMax = n
			}
			for i := ii; i < iMax; i++ {
				aRow := i * n
				cRow := i * p
				c := result[cRow : cRow+p]
				for k := kk; k < kMax; k++ {
					aik := a[aRow+k]
					bRow := k * p
					bb := b[bRow : bRow+p]

					j := 0
					for ; j+3 < p; j += 4 {
						c[j+0] += aik * bb[j+0]
						c[j+1] += aik * bb[j+1]
						c[j+2] += aik * bb[j+2]
						c[j+3] += aik * bb[j+3]
					}
					for ; j < p; j++ {
						c[j] += aik * bb[j]
					}
				}
			}
		}
	}
}

func (mat Mat) Transpose() Mat {
	result := make([]float64, len(mat.weights))

	rows, cols := mat.Rows, mat.Columns
	src := mat.weights

	for i := range rows {
		rowOffset := i * cols
		for j := range cols {
			result[j*rows+i] = src[rowOffset+j]
		}
	}

	return Mat{
		weights: result,
		Rows:    mat.Columns,
		Columns: mat.Rows,
	}
}

func (mat Mat) Apply(fn func(float64) float64) Mat {
	result := make([]float64, len(mat.weights))
	for i, v := range mat.weights {
		result[i] = fn(v)
	}
	return Mat{
		weights: result,
		Rows:    mat.Rows,
		Columns: mat.Columns,
	}
}

func (mat Mat) Scale(s float64) Mat {
	return mat.Apply(func(v float64) float64 { return v * s })
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

// Broadcast replicates a (1, C) row matrix into an (rows, C) matrix
// using pure Dot: ones(rows,1) . self(1,C) -> (rows,C), where every
// row of the result is a copy of the original row. Panics if mat is
// not a single row.
func (mat Mat) Broadcast(rows int) Mat {
	if mat.Rows != 1 {
		panic(fmt.Sprintf("Broadcast requires a (1, C) matrix, got (%v x %v)", mat.Rows, mat.Columns))
	}
	if rows == 1 {
		return mat
	}
	ones := NewZeroMat(rows, 1).Apply(func(float64) float64 { return 1 })
	return ones.Dot(mat)
}

// AddBias adds a (1, C) bias row to every row of mat, broadcasting
// via Broadcast + the existing (non-broadcasting) Add/Combine.
func (mat Mat) AddBias(bias Mat) Mat {
	if mat.Rows == bias.Rows {
		return mat.Add(bias) // already same shape, e.g. batch=1
	}
	return mat.Add(bias.Broadcast(mat.Rows))
}
