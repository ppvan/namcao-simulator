package rnn

import (
	"errors"
	"fmt"
	"math"
	"math/rand"
	"time"

	"github.com/ppvan/gpt/pkg/core"
	"github.com/ppvan/gpt/pkg/data"
)

// writeTargets fills dst, a (batch, 1) class-index column, from one timestep's
// targets. CrossEntropy takes targets in this shape.
func writeTargets(dst core.Mat, ids []int) {
	for i, id := range ids {
		dst.Set(i, 0, float32(id))
	}
}

// checkSeq validates the shapes shared by the loss entry points and returns the
// sequence length and batch size.
func checkSeq(logits []core.Mat, ys [][]int) (T, batch int, err error) {
	T = len(logits)
	if T == 0 {
		return 0, 0, errors.New("rnn: empty sequence")
	}
	if len(ys) != T {
		return 0, 0, fmt.Errorf("rnn: %d logits but %d target steps", T, len(ys))
	}
	batch = logits[0].Rows
	for t := range logits {
		if logits[t].Rows != batch {
			return 0, 0, fmt.Errorf("rnn: step %d has batch %d, expected %d", t, logits[t].Rows, batch)
		}
		if len(ys[t]) != batch {
			return 0, 0, fmt.Errorf("rnn: step %d has %d targets for batch %d", t, len(ys[t]), batch)
		}
	}
	return T, batch, nil
}

func LossAndGrad(logits []core.Mat, ys [][]int) (loss float32, dLogits []core.Mat, err error) {
	T, batch, err := checkSeq(logits, ys)
	if err != nil {
		return 0, nil, err
	}

	ce := core.CrossEntropy()
	dLogits = make([]core.Mat, T)
	scale := 1.0 / float32(T)
	target := core.NewZeroMat(batch, 1) // refilled per step, read immediately

	var total float32
	for t, y := range logits {
		writeTargets(target, ys[t])

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

// SeqLoss is the forward-only half of LossAndGrad, for evaluation.
func SeqLoss(logits []core.Mat, ys [][]int) (float32, error) {
	T, batch, err := checkSeq(logits, ys)
	if err != nil {
		return 0, err
	}

	ce := core.CrossEntropy()
	target := core.NewZeroMat(batch, 1)

	var total float32
	for t, y := range logits {
		writeTargets(target, ys[t])
		l, ferr := ce.Forward(y, target)
		if ferr != nil {
			return 0, ferr
		}
		total += l.Mean()
	}

	return total / float32(T), nil
}

// ErrNonFiniteGrad reports a step whose gradients contained NaN or Inf. The
// update is skipped rather than applied: a single non-finite entry poisons
// every parameter it touches, and Adam's moment buffers then keep it alive
// forever, so the run never recovers.
var ErrNonFiniteGrad = errors.New("rnn: non-finite gradients, update skipped")

// ClipGrads rescales grads in place so their global L2 norm is at most
// maxNorm, and reports whether they were finite.
//
// The norm is accumulated in float64 deliberately. With a float32 accumulator
// a single gradient element above ~1.8e19 overflows the sum of squares to
// +Inf, FastSqrt32(+Inf) returns -Inf, and -Inf <= maxNorm reads as "small
// enough, no clipping needed" -- so clipping switched itself off at exactly
// the moment it was needed. float64 has the headroom to hold the norm of any
// float32 matrix, and math.Sqrt is exact; this runs once per step over the
// whole parameter set, so its cost is noise next to the matmuls.
func ClipGrads(grads core.Grads, maxNorm float32) (core.Grads, bool) {
	var sumSq float64
	for _, g := range grads {
		sumSq += g.SumSquares64()
	}

	if math.IsNaN(sumSq) || math.IsInf(sumSq, 0) {
		return grads, false
	}
	if maxNorm <= 0 {
		return grads, true
	}

	norm := math.Sqrt(sumSq)
	if norm <= float64(maxNorm) || norm == 0 {
		return grads, true
	}

	scale := float32(float64(maxNorm) / norm)
	for _, g := range grads {
		g.ScaleInPlace(scale)
	}
	return grads, true
}

// FitStep runs one forward/backward/update over a time-major minibatch:
// xs[t] and ys[t] each hold one id per stream, so xs is (seqLen, batch).
func (m *CharRNN) FitStep(xs, ys [][]int, s0 State, opt core.Optimizer, clipNorm float32) (loss float32, sFinal State, err error) {
	logits, s, cache := m.ForwardSeq(xs, s0)

	loss, dLogits, err := LossAndGrad(logits, ys)
	if err != nil {
		return 0, nil, err
	}

	grads, _ := m.BackwardSeq(cache, dLogits)

	grads, finite := ClipGrads(grads, clipNorm)
	if !finite || math.IsNaN(float64(loss)) || math.IsInf(float64(loss), 0) {
		// Leave the parameters untouched and hand the caller a fresh state:
		// whatever produced this is in the carried activations, not the
		// weights, and the weights are still good.
		return loss, m.InitState(len(xs[0])), ErrNonFiniteGrad
	}

	params := m.Params()
	for name, g := range grads {
		params[name] = opt.Update(name, params[name], g)
	}
	m.SetParams(params)

	return loss, s, nil
}

// EvalLoss averages the validation loss over up to maxWindows minibatches.
// If the split is too small to feed the requested batch size the batch is
// halved until it fits, so a large training batch never breaks evaluation.
func (m *CharRNN) EvalLoss(ids []int, batch, seqLen, maxWindows int) (float32, error) {
	if batch <= 0 {
		batch = 1
	}

	var (
		eb  *data.SeqBatcher
		err error
	)
	for ; batch >= 1; batch /= 2 {
		eb, err = data.NewSeqBatcher(ids, batch, seqLen)
		if err == nil {
			break
		}
	}
	if eb == nil {
		return 0, fmt.Errorf("rnn: not enough data to evaluate: %w", err)
	}

	s := m.InitState(eb.Batch())
	var total float32
	windows := 0

	for windows < maxWindows {
		xs, ys, wrapped := eb.Next()
		if wrapped && windows > 0 {
			break
		}

		logits, sNext, _ := m.ForwardSeq(xs, s)
		loss, err := SeqLoss(logits, ys)
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
	return total / float32(windows), nil
}

type TrainConfig struct {
	Batch  int
	SeqLen int

	// Epochs is the training budget: passes over the corpus. A step consumes
	// Batch*SeqLen characters, so a raw step count means different amounts of
	// training at different batch sizes; epochs do not. Changing Batch changes
	// how fast the run goes, not how much training happens.
	Epochs float64

	ClipNorm float32

	// LogEvery and EvalEvery are intervals in epochs, for the same reason.
	// Zero selects the default (ten log lines per epoch, one eval per epoch);
	// negative disables.
	LogEvery  float64
	EvalEvery float64

	EvalWindows int
}

// epochsToSteps turns an epoch-relative interval into a step count. Zero takes
// def; negative disables and returns 0.
func epochsToSteps(every, def float64, perEpoch int) int {
	if every < 0 {
		return 0
	}
	if every == 0 {
		every = def
	}
	return max(1, int(every*float64(perEpoch)))
}

func DefaultTrainConfig() TrainConfig {
	return TrainConfig{
		Batch:       16,
		SeqLen:      100,
		Epochs:      20,
		ClipNorm:    5.0,
		EvalWindows: 100,
	}
}

func (m *CharRNN) Train(train, val []int, opt core.Optimizer, cfg TrainConfig) error {
	if cfg.SeqLen <= 0 {
		return errors.New("rnn: SeqLen must be positive")
	}

	if cfg.Epochs <= 0 {
		return errors.New("rnn: Epochs must be positive")
	}

	if cfg.Batch <= 0 {
		return errors.New("rnn: Batch must be positive")
	}

	batch := cfg.Batch
	tb, err := data.NewSeqBatcher(train, batch, cfg.SeqLen)
	if err != nil {
		return fmt.Errorf("rnn: %w", err)
	}

	perEpoch := tb.WindowsPerEpoch()

	steps := max(1, int(cfg.Epochs*float64(perEpoch)))
	logEvery := epochsToSteps(cfg.LogEvery, 0.1, perEpoch)
	evalEvery := epochsToSteps(cfg.EvalEvery, 1, perEpoch)

	s := m.InitState(batch)
	var running float32 = 0.0
	seen := 0
	skipped, consecutive := 0, 0
	lastLog := time.Now()

	// A handful of skipped steps is survivable; a run of them means the
	// parameters themselves are already NaN and nothing will recover them.
	const maxConsecutiveSkips = 5

	for step := 1; step <= steps; step++ {
		xs, ys, wrapped := tb.Next()
		if wrapped {
			// every stream jumped back to the head of its chunk, so the
			// carried hidden state no longer describes the current context
			s = m.InitState(batch)
		}

		loss, sNext, err := m.FitStep(xs, ys, s, opt, cfg.ClipNorm)
		s = sNext

		if errors.Is(err, ErrNonFiniteGrad) {
			skipped++
			consecutive++
			fmt.Printf("step %d (epoch %.2f): non-finite gradients, update skipped (%d so far)\n",
				step, float64(step)/float64(perEpoch), skipped)
			if consecutive >= maxConsecutiveSkips {
				return fmt.Errorf("rnn: step %d: %d consecutive non-finite steps; parameters are unrecoverable, lower the learning rate or the clip threshold",
					step, consecutive)
			}
			continue
		}
		if err != nil {
			return fmt.Errorf("rnn: step %d: %w", step, err)
		}
		consecutive = 0

		running += loss
		seen++

		if logEvery > 0 && step%logEvery == 0 && seen > 0 {
			// chars/s is the only throughput number comparable across batch
			// sizes; steps/s is not.
			cps := float64(seen*batch*cfg.SeqLen) / time.Since(lastLog).Seconds()
			fmt.Printf("step %d (epoch %.2f): train loss %.4f | %.0f chars/s\n",
				step, float64(step)/float64(perEpoch), running/float32(seen), cps)
			running, seen = 0, 0
			lastLog = time.Now()
		}

		if evalEvery > 0 && step%evalEvery == 0 && len(val) > cfg.SeqLen {
			windows := cfg.EvalWindows
			if windows <= 0 {
				windows = 100
			}
			vLoss, err := m.EvalLoss(val, batch, cfg.SeqLen, windows)
			if err != nil {
				return fmt.Errorf("rnn: step %d eval: %w", step, err)
			}
			fmt.Printf("step %d (epoch %.2f): val loss %.4f\n",
				step, float64(step)/float64(perEpoch), vLoss)
		}
	}

	return nil
}

func sampleFromLogits(logits core.Mat, temperature float32, rng *rand.Rand) int {
	cols := logits.Columns

	if temperature <= 0 {
		best, bestVal := 0, core.NegInf32()
		for j := 0; j < cols; j++ {
			if v := logits.Get(0, j); v > bestVal {
				best, bestVal = j, v
			}
		}
		return best
	}

	max := core.NegInf32()
	for j := 0; j < cols; j++ {
		if v := logits.Get(0, j) / temperature; v > max {
			max = v
		}
	}

	probs := make([]float32, cols)
	var sum float32
	for j := 0; j < cols; j++ {
		p := core.FastExp32(logits.Get(0, j)/temperature - max)
		probs[j] = p
		sum += p
	}

	r := rng.Float32() * sum
	var acc float32
	for j, p := range probs {
		acc += p
		if r <= acc {
			return j
		}
	}
	return cols - 1
}

// SampleMasked is Sample with two generation controls:
//
//   - banned ids have their logits forced to -Inf before every draw, so they
//     can never be emitted — use it to exclude PAD/UNK/BOS/EOS/STORY.
//   - stop, if non-nil, is consulted after each emitted token; returning true
//     ends generation early. maxTokens caps the output either way.
//
// This is how "generate N sentences" works: ban the specials, and stop once
// enough sentence-final tokens ('.', '!', '?', '…') have been produced.
func (m *CharRNN) SampleMasked(seed []int, maxTokens int, temperature float32, banned []int, stop func(id int) bool, rng *rand.Rand) []int {
	if maxTokens <= 0 {
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

	out := make([]int, 0, maxTokens)
	for i := 0; i < maxTokens; i++ {
		for _, b := range banned {
			if b >= 0 && b < logits.Columns {
				logits.Set(0, b, core.NegInf32())
			}
		}
		next := sampleFromLogits(logits, temperature, rng)
		out = append(out, next)
		if stop != nil && stop(next) {
			break
		}
		logits, s = m.StepOnce([]int{next}, s)
	}
	return out
}

func (m *CharRNN) Sample(seed []int, n int, temperature float32, rng *rand.Rand) []int {
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
