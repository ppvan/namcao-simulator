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

	model := rnn.NewCharRNN(vocab.VocabSize(), 32, 128)
	opt := core.NewAdam(0.001, 0.9, 0.999, 1e-8)

	cfg := rnn.TrainConfig{
		SeqLen:      25,
		Steps:       20000,
		ClipNorm:    5.0,
		LogEvery:    100,
		EvalEvery:   1000,
		EvalWindows: 100,
	}

	if err := model.Train(train, val, opt, cfg); err != nil {
		log.Fatalf("train: %v", err)
	}

	rng := rand.New(rand.NewSource(42))
	seed, err := vocab.Encode("The ")
	if err != nil {
		log.Fatalf("encode seed: %v", err)
	}

	generated := model.Sample(seed, 500, 0.8, rng)
	fmt.Println("\n--- sample ---")
	fmt.Println(data.Decode(append(seed, generated...), vocab))
}
