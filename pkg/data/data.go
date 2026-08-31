package data

import (
	"encoding/csv"
	"fmt"
	"math/rand"
	"os"
	"strconv"

	"github.com/ppvan/gpt/pkg/core"
)

type Data struct {
	X, Y core.Mat
}

type Batch struct {
	X core.Mat
	Y core.Mat
}

func LoadCSV(path string, labelCol int, hasHeader bool) (Data, error) {
	f, err := os.Open(path)
	if err != nil {
		return Data{}, err
	}
	defer f.Close()

	rows, err := csv.NewReader(f).ReadAll()
	if err != nil {
		return Data{}, err
	}

	if hasHeader {
		rows = rows[1:]
	}

	if len(rows) == 0 {
		return Data{}, fmt.Errorf("empty csv")
	}

	numRows := len(rows)
	numCols := len(rows[0])

	xCols := numCols - 1

	x := core.NewZeroMat(numRows, xCols)
	y := core.NewZeroMat(numRows, 1)

	for r, row := range rows {
		xc := 0

		for c, val := range row {
			fv, err := strconv.ParseFloat(val, 32)
			if err != nil {
				return Data{}, err
			}

			if c == labelCol {
				y.Set(r, 0, float32(fv))
			} else {
				x.Set(r, xc, float32(fv))
				xc++
			}
		}
	}

	return Data{
		X: x,
		Y: y,
	}, nil
}

func (d Data) Split(trainRatio, devRatio, testRatio float32) (Data, Data, Data) {
	total := trainRatio + devRatio + testRatio

	if total <= 0 {
		panic("split ratios must be positive")
	}

	trainRatio /= total
	devRatio /= total
	testRatio /= total

	n := d.X.Rows

	trainEnd := int(float32(n) * trainRatio)
	devEnd := trainEnd + int(float32(n)*devRatio)

	train := Data{
		X: d.X.Slice(0, trainEnd),
		Y: d.Y.Slice(0, trainEnd),
	}

	dev := Data{
		X: d.X.Slice(trainEnd, devEnd),
		Y: d.Y.Slice(trainEnd, devEnd),
	}

	test := Data{
		X: d.X.Slice(devEnd, n),
		Y: d.Y.Slice(devEnd, n),
	}

	return train, dev, test
}

func (d Data) Shuffle() Data {
	n := d.X.Rows

	perm := rand.Perm(n)

	x := core.NewZeroMat(d.X.Rows, d.X.Columns)
	y := core.NewZeroMat(d.Y.Rows, d.Y.Columns)

	for newRow, oldRow := range perm {
		for c := 0; c < d.X.Columns; c++ {
			x.Set(newRow, c, d.X.Get(oldRow, c))
		}

		for c := 0; c < d.Y.Columns; c++ {
			y.Set(newRow, c, d.Y.Get(oldRow, c))
		}
	}

	return Data{
		X: x,
		Y: y,
	}
}

func (d Data) Transform(f func(x core.Mat, y core.Mat) (core.Mat, core.Mat)) Data {
	newX, newY := f(d.X, d.Y)

	return Data{
		X: newX,
		Y: newY,
	}
}

func (d Data) Batches(size int) []Batch {
	n := d.X.Rows
	if n == 0 {
		return nil
	}

	indices := make([]int, n)
	for i := 0; i < n; i++ {
		indices[i] = i
	}

	var batches []Batch

	for start := 0; start < n; start += size {
		end := start + size
		if end > n {
			end = n
		}

		batchSize := end - start

		xBatch := core.NewZeroMat(batchSize, d.X.Columns)
		yBatch := core.NewZeroMat(batchSize, d.Y.Columns)

		for i := 0; i < batchSize; i++ {
			row := indices[start+i]

			for c := 0; c < d.X.Columns; c++ {
				xBatch.Set(i, c, d.X.Get(row, c))
			}

			for c := 0; c < d.Y.Columns; c++ {
				yBatch.Set(i, c, d.Y.Get(row, c))
			}
		}

		batches = append(batches, Batch{
			X: xBatch,
			Y: yBatch,
		})
	}

	return batches
}
