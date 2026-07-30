package main

import (
	"fmt"
	"log"
	"math/rand"

	"github.com/ppvan/gpt/pkg/core"
	"github.com/ppvan/gpt/pkg/data"
	"github.com/ppvan/gpt/pkg/rnn"
)

func main() {
	text, err := data.LoadText("dataset/tinyshakespeare/shakespeare.txt")
	if err != nil {
		log.Fatalf("load text: %v", err)
	}

	vocab := data.BuildVocab(text)
	ids := data.Encode(text, vocab)
	train, val := data.SplitData(ids, 0.9)

	fmt.Printf("corpus %d chars, vocab %d, train %d, val %d\n",
		len(text), vocab.VocabSize(), len(train), len(val))

	model := rnn.NewCharLSTM(vocab.VocabSize(), 32, 256)
	opt := core.NewAdam(0.002, 0.9, 0.999, 1e-8)

	cfg := rnn.TrainConfig{
		SeqLen:      100,
		Steps:       20000,
		ClipNorm:    5.0,
		LogEvery:    200,
		EvalEvery:   2000,
		EvalWindows: 50,
	}

	params := 0
	for _, p := range model.Params() {
		params += p.Rows * p.Columns
	}
	fmt.Printf("lstm: embed 32, hidden 256, %d params, %d chars seen over %d steps\n",
		params, cfg.SeqLen*cfg.Steps, cfg.Steps)

	if err := model.Train(train, val, opt, cfg); err != nil {
		log.Fatalf("train: %v", err)
	}

	rng := rand.New(rand.NewSource(42))
	seed, err := vocab.Encode("The ")
	if err != nil {
		log.Fatalf("encode seed: %v", err)
	}

	for _, temp := range []float64{0.5, 0.8, 1.0} {
		generated := model.Sample(seed, 500, temp, rng)
		fmt.Printf("\n--- sample (temperature %.1f) ---\n", temp)
		fmt.Println(data.Decode(append(append([]int{}, seed...), generated...), vocab))
	}
}
