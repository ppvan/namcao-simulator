package mlp

import (
	"context"
	"errors"
	"fmt"

	"github.com/ppvan/gpt/pkg/core"
	"github.com/ppvan/gpt/pkg/data"
)

type Network struct {
	model     Sequential
	loss      core.LossFunction
	optimizer core.Optimizer
}

func NewNetwork(model Sequential, loss core.LossFunction, optimizer core.Optimizer, opts ...NetworkOption) *Network {
	n := &Network{model: model, loss: loss, optimizer: optimizer}
	for _, opt := range opts {
		opt(n)
	}
	return n
}

type EvalMetrics struct {
	Accuracy  float32
	Precision float32
	Recall    float32
	F1        float32
}

type NetworkOption func(*Network)

func (n *Network) Infer(x core.Mat) core.Mat {
	out, _ := n.model.Forward(x)
	return out
}

func (n *Network) Predict(x core.Mat) (Prediction, error) {
	logits := n.Infer(x)
	if logits.Rows == 0 || logits.Columns == 0 {
		return Prediction{}, errors.New("nn: empty logits from model")
	}

	preds := make([]Prediction, logits.Rows)
	for i := 0; i < logits.Rows; i++ {
		maxIdx := 0
		maxVal := logits.Get(i, 0)
		probs := make([]float32, logits.Columns)
		probs[0] = maxVal
		for j := 1; j < logits.Columns; j++ {
			v := logits.Get(i, j)
			probs[j] = v
			if v > maxVal {
				maxVal = v
				maxIdx = j
			}
		}
		preds[i] = Prediction{Class: maxIdx, Probs: probs}
	}

	if len(preds) == 1 {
		return preds[0], nil
	}
	return Prediction{}, fmt.Errorf("nn: Predict called with %d rows, use PredictBatch instead", logits.Rows)
}

func (n *Network) PredictBatch(x core.Mat) ([]Prediction, error) {
	logits := n.Infer(x)
	if logits.Rows == 0 || logits.Columns == 0 {
		return nil, errors.New("nn: empty logits from model")
	}

	preds := make([]Prediction, logits.Rows)
	for i := 0; i < logits.Rows; i++ {
		maxIdx := 0
		maxVal := logits.Get(i, 0)
		probs := make([]float32, logits.Columns)
		probs[0] = maxVal
		for j := 1; j < logits.Columns; j++ {
			v := logits.Get(i, j)
			probs[j] = v
			if v > maxVal {
				maxVal = v
				maxIdx = j
			}
		}
		preds[i] = Prediction{Class: maxIdx, Probs: probs}
	}
	return preds, nil
}

func (n *Network) Fit(ctx context.Context, data data.Data, epochs int, batchSize int) <-chan TrainMetrics {
	out := make(chan TrainMetrics)
	go func() {
		defer close(out)
		for epoch := 0; epoch < epochs; epoch++ {
			select {
			case <-ctx.Done():
				return
			default:
			}

			var epochLoss float32
			var batchCount float32
			for _, batch := range data.Batches(batchSize) {
				x := batch.X
				y := batch.Y

				pred, cache := n.model.Forward(x)
				lossMat, err := n.loss.Forward(pred, y)
				if err != nil {
					continue
				}
				dOut, err := n.loss.Backward(pred, y)
				if err != nil {
					continue
				}

				_, grads := n.model.Backward(cache, dOut)

				params := n.model.Params()
				newParams := core.Params{}
				for name, p := range params {
					newParams[name] = n.optimizer.Update(name, p, grads[name])
				}
				n.model.SetParams(newParams)

				epochLoss += lossMat.Mean()
				batchCount++
			}

			select {
			case out <- TrainMetrics{Epoch: epoch + 1, Loss: epochLoss / float32(batchCount)}:
			case <-ctx.Done():
				return
			}
		}
	}()
	return out
}

func (n *Network) Evaluate(dev data.Data) (EvalMetrics, error) {
	preds, err := n.PredictBatch(dev.X)
	if err != nil {
		return EvalMetrics{}, err
	}

	p := precision(preds, dev.Y)
	r := recall(preds, dev.Y)
	return EvalMetrics{
		Accuracy:  accuracy(preds, dev.Y),
		Precision: p,
		Recall:    r,
		F1:        f1(p, r),
	}, nil
}
func accuracy(preds []Prediction, y core.Mat) float32 {
	correct := 0
	for i, p := range preds {
		if p.Class == int(y.Get(i, 0)) {
			correct++
		}
	}
	return float32(correct) / float32(len(preds))
}

func precision(preds []Prediction, y core.Mat) float32 {
	n := len(preds)
	classes := map[int]float32{}
	for i := range y.Rows {
		classes[int(y.Get(i, 0))] = 0
	}
	for class := range classes {
		truePositives := 0
		positives := 0
		for i := 0; i < n; i++ {
			prediction := preds[i].Class
			label := int(y.Get(i, 0))
			if prediction == class {
				positives++
			}
			if prediction == class && label == class {
				truePositives++
			}
		}
		if positives == 0 {
			classes[class] = 0
			continue
		}
		classes[class] = float32(truePositives) / float32(positives)
	}

	var avg float32
	for _, v := range classes {
		avg += v
	}
	return avg / float32(len(classes))
}

func recall(preds []Prediction, y core.Mat) float32 {
	n := len(preds)
	classes := map[int]float32{}
	for i := range y.Rows {
		classes[int(y.Get(i, 0))] = 0
	}
	for class := range classes {
		truePositives := 0
		actuals := 0
		for i := 0; i < n; i++ {
			prediction := preds[i].Class
			label := int(y.Get(i, 0))
			if label == class {
				actuals++
			}
			if prediction == class && label == class {
				truePositives++
			}
		}
		if actuals == 0 {
			classes[class] = 0
			continue
		}
		classes[class] = float32(truePositives) / float32(actuals)
	}

	var avg float32
	for _, v := range classes {
		avg += v
	}
	return avg / float32(len(classes))
}

func f1(precision, recall float32) float32 {
	if precision+recall == 0 {
		return 0
	}
	return 2 * precision * recall / (precision + recall)
}

func (e EvalMetrics) String() string {
	return fmt.Sprintf(
		"Metrics {\n"+
			"\tacc:  %.2f%%\n"+
			"\tprec: %.2f%%\n"+
			"\trec:  %.2f%%\n"+
			"\tf1:   %.2f%%\n"+
			"}",
		e.Accuracy*100,
		e.Precision*100,
		e.Recall*100,
		e.F1*100,
	)
}
