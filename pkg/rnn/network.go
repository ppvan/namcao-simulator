package rnn

import (
	"errors"
	"fmt"
	"math"
	"math/rand"

	"github.com/ppvan/gpt/pkg/core"
	"github.com/ppvan/gpt/pkg/data"
)

func toTimeMajor(ids []int) [][]int {
	xsIDs := make([][]int, len(ids))
	for t, id := range ids {
		xsIDs[t] = []int{id}
	}
	return xsIDs
}

func targetMat(id int) core.Mat {
	m := core.NewZeroMat(1, 1)
	m.Set(0, 0, float64(id))
	return m
}

func LossAndGrad(logits []core.Mat, ys []int) (loss float64, dLogits []core.Mat, err error) {
	T := len(logits)
	if T == 0 {
		return 0, nil, errors.New("rnn: empty sequence")
	}
	if len(ys) != T {
		return 0, nil, fmt.Errorf("rnn: %d logits but %d targets", T, len(ys))
	}

	ce := core.CrossEntropy()
	dLogits = make([]core.Mat, T)
	scale := 1.0 / float64(T)

	var total float64
	for t, y := range logits {
		target := targetMat(ys[t])

		l, ferr := ce.Forward(y, target)
		if ferr != nil {
			return 0, nil, ferr
		}
		total += l.Mean()

		dy, berr := ce.Backward(y, target)
		if berr != nil {
			return 0, nil, berr
		}
		dLogits[t] = dy.Scale(scale)
	}

	return total * scale, dLogits, nil
}

func ClipGrads(grads core.Grads, maxNorm float64) core.Grads {
	if maxNorm <= 0 {
		return grads
	}

	var sumSq float64
	for _, g := range grads {
		sumSq += g.Hadamard(g).Sum()
	}
	norm := math.Sqrt(sumSq)

	if norm <= maxNorm || norm == 0 || math.IsNaN(norm) {
		return grads
	}

	scale := maxNorm / norm
	clipped := make(core.Grads, len(grads))
	for name, g := range grads {
		clipped[name] = g.Scale(scale)
	}
	return clipped
}

func (m *CharRNN) FitStep(xs, ys []int, s0 State, opt core.Optimizer, clipNorm float64) (loss float64, sFinal State, err error) {
	logits, s, cache := m.ForwardSeq(toTimeMajor(xs), s0)

	loss, dLogits, err := LossAndGrad(logits, ys)
	if err != nil {
		return 0, nil, err
	}

	grads, _ := m.BackwardSeq(cache, dLogits)
	grads = ClipGrads(grads, clipNorm)

	params := m.Params()
	for name, g := range grads {
		params[name] = opt.Update(name, params[name], g)
	}
	m.SetParams(params)

	return loss, s, nil
}

func (m *CharRNN) EvalLoss(ids []int, seqLen, maxWindows int) (float64, error) {
	if len(ids) < seqLen+1 {
		return 0, errors.New("rnn: not enough data to evaluate")
	}

	s := m.InitState(1)
	pos := 0
	var total float64
	windows := 0

	for windows < maxWindows {
		xs, ys, next, wrapped := data.NextBatch(ids, pos, seqLen)
		if wrapped && windows > 0 {
			break
		}
		pos = next

		logits, sNext, _ := m.ForwardSeq(toTimeMajor(xs), s)
		loss, _, err := LossAndGrad(logits, ys)
		if err != nil {
			return 0, err
		}

		total += loss
		s = sNext
		windows++
	}

	if windows == 0 {
		return 0, errors.New("rnn: no evaluation windows")
	}
	return total / float64(windows), nil
}

type TrainConfig struct {
	SeqLen      int
	Steps       int
	ClipNorm    float64
	LogEvery    int
	EvalEvery   int
	EvalWindows int
}

func DefaultTrainConfig() TrainConfig {
	return TrainConfig{
		SeqLen:      25,
		Steps:       10000,
		ClipNorm:    5.0,
		LogEvery:    100,
		EvalEvery:   1000,
		EvalWindows: 100,
	}
}

func (m *CharRNN) Train(train, val []int, opt core.Optimizer, cfg TrainConfig) error {
	if cfg.SeqLen <= 0 || cfg.Steps <= 0 {
		return errors.New("rnn: SeqLen and Steps must be positive")
	}
	if len(train) < cfg.SeqLen+1 {
		return errors.New("rnn: training data shorter than one window")
	}

	s := m.InitState(1)
	pos := 0
	running := 0.0
	seen := 0

	for step := 1; step <= cfg.Steps; step++ {
		xs, ys, next, wrapped := data.NextBatch(train, pos, cfg.SeqLen)
		if wrapped {
			s = m.InitState(1)
		}
		pos = next

		loss, sNext, err := m.FitStep(xs, ys, s, opt, cfg.ClipNorm)
		if err != nil {
			return fmt.Errorf("rnn: step %d: %w", step, err)
		}
		s = sNext

		running += loss
		seen++

		if cfg.LogEvery > 0 && step%cfg.LogEvery == 0 {
			fmt.Printf("step %d: train loss %.4f\n", step, running/float64(seen))
			running, seen = 0, 0
		}

		if cfg.EvalEvery > 0 && step%cfg.EvalEvery == 0 && len(val) > cfg.SeqLen {
			windows := cfg.EvalWindows
			if windows <= 0 {
				windows = 100
			}
			vLoss, err := m.EvalLoss(val, cfg.SeqLen, windows)
			if err != nil {
				return fmt.Errorf("rnn: step %d eval: %w", step, err)
			}
			fmt.Printf("step %d: val loss %.4f\n", step, vLoss)
		}
	}

	return nil
}

func sampleFromLogits(logits core.Mat, temperature float64, rng *rand.Rand) int {
	cols := logits.Columns

	if temperature <= 0 {
		best, bestVal := 0, math.Inf(-1)
		for j := 0; j < cols; j++ {
			if v := logits.Get(0, j); v > bestVal {
				best, bestVal = j, v
			}
		}
		return best
	}

	max := math.Inf(-1)
	for j := 0; j < cols; j++ {
		if v := logits.Get(0, j) / temperature; v > max {
			max = v
		}
	}

	probs := make([]float64, cols)
	sum := 0.0
	for j := 0; j < cols; j++ {
		p := math.Exp(logits.Get(0, j)/temperature - max)
		probs[j] = p
		sum += p
	}

	r := rng.Float64() * sum
	acc := 0.0
	for j, p := range probs {
		acc += p
		if r <= acc {
			return j
		}
	}
	return cols - 1
}

func (m *CharRNN) Sample(seed []int, n int, temperature float64, rng *rand.Rand) []int {
	if n <= 0 {
		return nil
	}
	if rng == nil {
		rng = rand.New(rand.NewSource(rand.Int63()))
	}
	if len(seed) == 0 {
		seed = []int{rng.Intn(m.vocabSize)}
	}

	s := m.InitState(1)
	var logits core.Mat
	for _, id := range seed {
		logits, s = m.StepOnce([]int{id}, s)
	}

	out := make([]int, 0, n)
	for i := 0; i < n; i++ {
		next := sampleFromLogits(logits, temperature, rng)
		out = append(out, next)
		logits, s = m.StepOnce([]int{next}, s)
	}
	return out
}
