package core

import (
	"errors"
	"fmt"
	"math"
)

type LossFunction interface {
	Forward(pred, target Mat) (Mat, error)
	Backward(pred, target Mat) (Mat, error)
}

type meanSquareError struct{}

func MeanSquareError() *meanSquareError {
	return &meanSquareError{}
}

func (m meanSquareError) Forward(pred, target Mat) (Mat, error) {
	if pred.Rows != target.Rows || pred.Columns != target.Columns {
		return Mat{}, fmt.Errorf("nn: MSE shape mismatch, pred %dx%d vs target %dx%d",
			pred.Rows, pred.Columns, target.Rows, target.Columns)
	}

	out := NewZeroMat(pred.Rows, 1)
	for i := 0; i < pred.Rows; i++ {
		sum := 0.0
		for j := 0; j < pred.Columns; j++ {
			d := pred.Get(i, j) - target.Get(i, j)
			sum += d * d
		}
		out.Set(i, 0, sum/float64(pred.Columns))
	}
	return out, nil
}

func (m meanSquareError) Backward(pred, target Mat) (Mat, error) {
	if pred.Rows != target.Rows || pred.Columns != target.Columns {
		return Mat{}, fmt.Errorf("nn: MSE shape mismatch, pred %dx%d vs target %dx%d",
			pred.Rows, pred.Columns, target.Rows, target.Columns)
	}

	scale := 2.0 / float64(pred.Columns)
	out := NewZeroMat(pred.Rows, pred.Columns)
	for i := 0; i < pred.Rows; i++ {
		for j := 0; j < pred.Columns; j++ {
			out.Set(i, j, scale*(pred.Get(i, j)-target.Get(i, j)))
		}
	}
	return out, nil
}

type crossEntropyLoss struct {
	probs Mat
}

func CrossEntropy() *crossEntropyLoss {
	return &crossEntropyLoss{}
}

func (c *crossEntropyLoss) Forward(logits, target Mat) (Mat, error) {
	if logits.Rows != target.Rows {
		return Mat{}, fmt.Errorf("nn: CrossEntropy row mismatch, logits %d vs target %d",
			logits.Rows, target.Rows)
	}
	if target.Columns != 1 {
		return Mat{}, errors.New("nn: CrossEntropy expects target as class-index column (Nx1)")
	}

	out := NewZeroMat(logits.Rows, 1)
	probs := NewZeroMat(logits.Rows, logits.Columns)

	for i := 0; i < logits.Rows; i++ {
		max := logits.Get(i, 0)
		for j := 1; j < logits.Columns; j++ {
			if v := logits.Get(i, j); v > max {
				max = v
			}
		}
		sumExp := 0.0
		for j := 0; j < logits.Columns; j++ {
			v := math.Exp(logits.Get(i, j) - max)
			probs.Set(i, j, v)
			sumExp += v
		}
		logSumExp := max + math.Log(sumExp)
		for j := 0; j < logits.Columns; j++ {
			probs.Set(i, j, probs.Get(i, j)/sumExp)
		}

		t := int(target.Get(i, 0))
		if t < 0 || t >= logits.Columns {
			return Mat{}, fmt.Errorf("nn: CrossEntropy target class %d out of range [0,%d)", t, logits.Columns)
		}
		loss := -(logits.Get(i, t) - logSumExp)
		out.Set(i, 0, loss)
	}

	c.probs = probs
	return out, nil
}

func (c *crossEntropyLoss) Backward(logits, target Mat) (Mat, error) {
	if c.probs.Rows == 0 {
		return Mat{}, errors.New("nn: CrossEntropy Backward called before Forward")
	}
	if logits.Rows != target.Rows {
		return Mat{}, fmt.Errorf("nn: CrossEntropy row mismatch, logits %d vs target %d",
			logits.Rows, target.Rows)
	}

	out := NewZeroMat(logits.Rows, logits.Columns)
	scale := 1.0 / float64(logits.Rows)
	for i := 0; i < logits.Rows; i++ {
		t := int(target.Get(i, 0))
		for j := 0; j < logits.Columns; j++ {
			grad := c.probs.Get(i, j)
			if j == t {
				grad -= 1.0
			}
			out.Set(i, j, scale*grad)
		}
	}
	return out, nil
}

type binaryCrossEntropyLoss struct{}

func BinaryCrossEntropy() *binaryCrossEntropyLoss {
	return &binaryCrossEntropyLoss{}
}

func (b binaryCrossEntropyLoss) Forward(pred, target Mat) (Mat, error) {
	if pred.Rows != target.Rows || pred.Columns != target.Columns {
		return Mat{}, fmt.Errorf("nn: BCE shape mismatch, pred %dx%d vs target %dx%d",
			pred.Rows, pred.Columns, target.Rows, target.Columns)
	}

	out := NewZeroMat(pred.Rows, 1)
	for i := 0; i < pred.Rows; i++ {
		p := pred.Get(i, 0)
		p = math.Max(1e-15, math.Min(1-1e-15, p))
		y := target.Get(i, 0)
		loss := -y*math.Log(p) - (1-y)*math.Log(1-p)
		out.Set(i, 0, loss)
	}
	return out, nil
}

func (b binaryCrossEntropyLoss) Backward(pred, target Mat) (Mat, error) {
	if pred.Rows != target.Rows || pred.Columns != target.Columns {
		return Mat{}, fmt.Errorf("nn: BCE shape mismatch, pred %dx%d vs target %dx%d",
			pred.Rows, pred.Columns, target.Rows, target.Columns)
	}

	out := NewZeroMat(pred.Rows, 1)
	scale := 1.0 / float64(pred.Rows)
	for i := 0; i < pred.Rows; i++ {
		p := pred.Get(i, 0)
		y := target.Get(i, 0)
		out.Set(i, 0, scale*(p-y)/(p*(1-p)))
	}
	return out, nil
}
