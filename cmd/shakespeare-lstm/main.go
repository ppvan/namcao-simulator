package main

import (
	"flag"
	"fmt"
	"log"
	"math/rand"
	"os"

	"github.com/ppvan/gpt/pkg/core"
	"github.com/ppvan/gpt/pkg/data"
	"github.com/ppvan/gpt/pkg/rnn"
)

func main() {
	// ── flags ────────────────────────────────────────────────────────────
	loadPath := flag.String("load", "", "resume training from this checkpoint file")
	savePath := flag.String("save", "shakespeare-lstm.ckpt", "write checkpoint to this file after training")
	noSaveOpt := flag.Bool("no-save-optimizer", false, "omit Adam state from checkpoint (smaller file, can't resume exactly)")
	flag.Parse()

	// ── data ─────────────────────────────────────────────────────────────
	text, err := data.LoadText("dataset/tinyshakespeare/shakespeare.txt")
	if err != nil {
		log.Fatalf("load text: %v", err)
	}

	vocab := data.BuildVocab(text)
	ids := data.Encode(text, vocab)
	train, val := data.SplitData(ids, 0.9)

	fmt.Printf("corpus %d chars, vocab %d, train %d, val %d\n",
		len(text), vocab.VocabSize(), len(train), len(val))

	// ── model + optimizer ────────────────────────────────────────────────
	var model *rnn.CharRNN
	opt := core.NewAdam(0.002, 0.9, 0.999, 1e-8)

	if *loadPath != "" {
		fmt.Printf("loading checkpoint from %q …\n", *loadPath)
		res, lerr := rnn.Load(*loadPath)
		if lerr != nil {
			log.Fatalf("load checkpoint: %v", lerr)
		}
		model = res.Model
		if res.Adam != nil {
			// restore moment buffers; set the same hyperparams the run started with
			res.Adam.Rate = opt.Rate
			res.Adam.Beta1 = opt.Beta1
			res.Adam.Beta2 = opt.Beta2
			res.Adam.Epsilon = opt.Epsilon
			opt = res.Adam
			fmt.Println("Adam optimizer state restored from checkpoint")
		}
		fmt.Printf("loaded model: vocab %d, embed %d, hidden %d\n",
			model.VocabSize(), model.Embed.Weights.Columns, model.HiddenSize())
	} else {
		model = rnn.NewCharLSTM(vocab.VocabSize(), 32, 256)
	}

	// ── training config ───────────────────────────────────────────────────
	cfg := rnn.TrainConfig{
		Batch:       16,
		SeqLen:      100,
		Epochs:      20,
		ClipNorm:    5.0,
		EvalWindows: 50,
	}

	params := 0
	for _, p := range model.Params() {
		params += p.Rows * p.Columns
	}
	fmt.Printf("lstm: embed %d, hidden %d, %d params, batch %d, %.0f epochs (%d chars)\n",
		model.Embed.Weights.Columns, model.HiddenSize(),
		params, cfg.Batch, cfg.Epochs, int(cfg.Epochs*float64(len(train))))

	if err := model.Train(train, val, opt, cfg); err != nil {
		log.Fatalf("train: %v", err)
	}

	// ── save checkpoint ───────────────────────────────────────────────────
	if *savePath != "" {
		saveOpts := rnn.SaveOptions{}
		if !*noSaveOpt {
			saveOpts.Optimizer = opt
		}
		fmt.Printf("saving checkpoint to %q …\n", *savePath)
		if err := rnn.Save(*savePath, model, saveOpts); err != nil {
			log.Fatalf("save checkpoint: %v", err)
		}
		info, _ := os.Stat(*savePath)
		fmt.Printf("checkpoint written (%.1f MB)\n", float64(info.Size())/1e6)
	}

	// ── sample ───────────────────────────────────────────────────────────
	rng := rand.New(rand.NewSource(42))
	seed, err := vocab.Encode("The ")
	if err != nil {
		log.Fatalf("encode seed: %v", err)
	}

	for _, temp := range []float32{0.5, 0.8, 1.0} {
		generated := model.Sample(seed, 500, temp, rng)
		fmt.Printf("\n--- sample (temperature %.1f) ---\n", temp)
		fmt.Println(data.Decode(append(append([]int{}, seed...), generated...), vocab))
	}
}
