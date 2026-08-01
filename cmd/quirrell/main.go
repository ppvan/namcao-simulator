// quirrell – a character-level LSTM trainer and text generator.
//
// Usage:
//
//	quirrell train  <model.bin>  [flags]   Train on a text file and save.
//	quirrell predict <model.bin> [flags]   Load a checkpoint and generate text.
//
// Examples:
//
//	quirrell train model.bin --data dataset/tinyshakespeare/shakespeare.txt
//	quirrell train model.bin --data corpus.txt --steps 50000 --hidden 512
//	quirrell predict model.bin --prompt "To be or not" --length 400 --temperature 0.8
package main

import (
	"context"
	"fmt"
	"log"
	"math/rand"
	"os"
	"time"

	"github.com/ppvan/gpt/pkg/core"
	"github.com/ppvan/gpt/pkg/data"
	"github.com/ppvan/gpt/pkg/rnn"
	"github.com/urfave/cli/v3"
)

const version = "0.1.0"

func main() {
	app := &cli.Command{
		Name:    "quirrell",
		Usage:   "character-level LSTM – train and generate text",
		Version: version,
		Commands: []*cli.Command{
			trainCmd(),
			predictCmd(),
		},
	}

	if err := app.Run(context.Background(), os.Args); err != nil {
		log.Fatal(err)
	}
}

// ── train ─────────────────────────────────────────────────────────────────────

func trainCmd() *cli.Command {
	return &cli.Command{
		Name:      "train",
		Usage:     "Train a character LSTM and save the checkpoint",
		ArgsUsage: "<model.bin>",
		Flags: []cli.Flag{
			&cli.StringFlag{
				Name:    "data",
				Aliases: []string{"d"},
				Value:   "dataset/tinyshakespeare/shakespeare.txt",
				Usage:   "path to the plain-text training corpus",
			},
			&cli.IntFlag{
				Name:  "embed",
				Value: 32,
				Usage: "character embedding dimension",
			},
			&cli.IntFlag{
				Name:  "hidden",
				Value: 256,
				Usage: "LSTM hidden state dimension",
			},
			&cli.IntFlag{
				Name:  "batch",
				Value: 32,
				Usage: "parallel streams per step (1 = old single-stream behaviour)",
			},
			&cli.IntFlag{
				Name:  "seqlen",
				Value: 100,
				Usage: "BPTT sequence length (chars per stream per step)",
			},
			&cli.IntFlag{
				Name:  "steps",
				Value: 20000,
				Usage: "number of gradient-update steps",
			},
			&cli.Float64Flag{
				Name:  "lr",
				Value: 0.002,
				Usage: "Adam learning rate",
			},
			&cli.Float64Flag{
				Name:  "clip",
				Value: 5.0,
				Usage: "gradient-norm clip threshold (0 = disabled)",
			},
			&cli.IntFlag{
				Name:  "log-every",
				Value: 200,
				Usage: "print train loss every N steps (0 = silent)",
			},
			&cli.IntFlag{
				Name:  "eval-every",
				Value: 2000,
				Usage: "evaluate validation loss every N steps (0 = disabled)",
			},
			&cli.IntFlag{
				Name:  "eval-windows",
				Value: 50,
				Usage: "number of windows to average for the validation loss",
			},
			&cli.Float64Flag{
				Name:  "train-ratio",
				Value: 0.9,
				Usage: "fraction of corpus used for training (rest is validation)",
			},
			&cli.BoolFlag{
				Name:  "resume",
				Usage: "if the checkpoint already exists, load it and keep training",
			},
			&cli.BoolFlag{
				Name:  "no-optimizer-state",
				Usage: "omit Adam moment buffers from the checkpoint (smaller file; can't resume exactly)",
			},
		},
		Action: runTrain,
	}
}

func runTrain(ctx context.Context, cmd *cli.Command) error {
	// ── positional arg: output path ───────────────────────────────────────
	if cmd.NArg() == 0 {
		return fmt.Errorf("missing argument: provide a path for the checkpoint, e.g. model.bin")
	}
	modelPath := cmd.Args().First()

	// ── load corpus ───────────────────────────────────────────────────────
	dataPath := cmd.String("data")
	fmt.Printf("quirrell: loading corpus from %q\n", dataPath)
	text, err := data.LoadText(dataPath)
	if err != nil {
		return fmt.Errorf("load corpus: %w", err)
	}

	vocab := data.BuildVocab(text)
	ids := data.Encode(text, vocab)
	trainRatio := float32(cmd.Float("train-ratio"))
	train, val := data.SplitData(ids, trainRatio)

	fmt.Printf("quirrell: corpus %d chars | vocab %d | train %d | val %d\n",
		len(text), vocab.VocabSize(), len(train), len(val))

	// ── model + optimizer ─────────────────────────────────────────────────
	lr := float32(cmd.Float("lr"))
	opt := core.NewAdam(lr, 0.9, 0.999, 1e-8)

	var model *rnn.CharRNN

	if cmd.Bool("resume") {
		if _, err := os.Stat(modelPath); err == nil {
			fmt.Printf("quirrell: resuming from %q\n", modelPath)
			res, lerr := rnn.Load(modelPath)
			if lerr != nil {
				return fmt.Errorf("resume load: %w", lerr)
			}
			model = res.Model
			if res.Adam != nil {
				res.Adam.Rate = lr
				opt = res.Adam
				fmt.Println("quirrell: Adam optimizer state restored")
			}
		} else {
			fmt.Println("quirrell: no existing checkpoint found – starting fresh")
		}
	}

	if model == nil {
		embedSize := cmd.Int("embed")
		hiddenSize := cmd.Int("hidden")
		model = rnn.NewCharLSTM(vocab.VocabSize(), int(embedSize), int(hiddenSize))
	}

	// ── print param count ─────────────────────────────────────────────────
	totalParams := 0
	for _, p := range model.Params() {
		totalParams += p.Rows * p.Columns
	}
	seqLen := int(cmd.Int("seqlen"))
	steps := int(cmd.Int("steps"))
	batch := int(cmd.Int("batch"))
	if batch < 1 {
		batch = 1
	}
	fmt.Printf("quirrell: embed %d | hidden %d | params %d | batch %d | seq %d | steps %d\n",
		model.Embed.Weights.Columns, model.HiddenSize(),
		totalParams, batch, seqLen, steps)
	fmt.Printf("quirrell: %d chars/step, %d chars total (%.1f epochs over %d train chars)\n",
		batch*seqLen, batch*seqLen*steps,
		float64(batch*seqLen*steps)/float64(len(train)), len(train))

	// ── train ─────────────────────────────────────────────────────────────
	cfg := rnn.TrainConfig{
		Batch:       batch,
		SeqLen:      seqLen,
		Steps:       steps,
		ClipNorm:    float32(cmd.Float("clip")),
		LogEvery:    int(cmd.Int("log-every")),
		EvalEvery:   int(cmd.Int("eval-every")),
		EvalWindows: int(cmd.Int("eval-windows")),
	}

	start := time.Now()
	if err := model.Train(train, val, opt, cfg); err != nil {
		return fmt.Errorf("training: %w", err)
	}
	elapsed := time.Since(start)
	fmt.Printf("\nquirrell: training finished in %s (%.1f steps/s, %.0f chars/s)\n",
		elapsed.Round(time.Second),
		float64(steps)/elapsed.Seconds(),
		float64(steps*batch*seqLen)/elapsed.Seconds())

	// ── save checkpoint ───────────────────────────────────────────────────
	saveOpts := rnn.SaveOptions{Vocab: vocab}
	if !cmd.Bool("no-optimizer-state") {
		saveOpts.Optimizer = opt
	}

	fmt.Printf("quirrell: saving checkpoint → %q\n", modelPath)
	if err := rnn.Save(modelPath, model, saveOpts); err != nil {
		return fmt.Errorf("save checkpoint: %w", err)
	}

	info, _ := os.Stat(modelPath)
	fmt.Printf("quirrell: checkpoint written (%.2f MB)\n", float64(info.Size())/1e6)

	// ── quick sample after training ───────────────────────────────────────
	fmt.Println("\n── sample (temperature 0.8) ─────────────────────────────────")
	sample := generateText(model, vocab, "The ", 300, 0.8, rand.New(rand.NewSource(42)))
	fmt.Println(sample)

	return nil
}

// ── predict ───────────────────────────────────────────────────────────────────

func predictCmd() *cli.Command {
	return &cli.Command{
		Name:      "predict",
		Aliases:   []string{"gen", "generate"},
		Usage:     "Load a checkpoint and generate text",
		ArgsUsage: "<model.bin>",
		Flags: []cli.Flag{
			&cli.StringFlag{
				Name:    "prompt",
				Aliases: []string{"p"},
				Value:   "The ",
				Usage:   "seed text fed to the model before sampling",
			},
			&cli.IntFlag{
				Name:    "length",
				Aliases: []string{"n"},
				Value:   500,
				Usage:   "number of characters to generate",
			},
			&cli.Float64Flag{
				Name:    "temperature",
				Aliases: []string{"t"},
				Value:   0.8,
				Usage:   "sampling temperature (lower = more conservative, higher = more creative)",
			},
			&cli.IntFlag{
				Name:  "seed",
				Value: 0,
				Usage: "random seed (0 = pick one from the clock)",
			},
			&cli.BoolFlag{
				Name:    "show-prompt",
				Aliases: []string{"s"},
				Value:   true,
				Usage:   "include the prompt text in the output",
			},
		},
		Action: runPredict,
	}
}

func runPredict(_ context.Context, cmd *cli.Command) error {
	// ── positional arg: checkpoint path ──────────────────────────────────
	if cmd.NArg() == 0 {
		return fmt.Errorf("missing argument: provide a checkpoint path, e.g. model.bin")
	}
	modelPath := cmd.Args().First()

	// ── load ──────────────────────────────────────────────────────────────
	fmt.Fprintf(os.Stderr, "quirrell: loading %q\n", modelPath)
	res, err := rnn.Load(modelPath)
	if err != nil {
		return fmt.Errorf("load checkpoint: %w", err)
	}
	if res.Vocab == nil {
		return fmt.Errorf("checkpoint %q has no embedded vocabulary; it was saved without --data (old format)", modelPath)
	}

	model := res.Model
	vocab := res.Vocab
	totalParams := 0
	for _, p := range model.Params() {
		totalParams += p.Rows * p.Columns
	}
	fmt.Fprintf(os.Stderr, "quirrell: vocab %d | embed %d | hidden %d | param %d\n\n",
		vocab.VocabSize(), model.Embed.Weights.Columns, model.HiddenSize(), totalParams)

	// ── seed ──────────────────────────────────────────────────────────────
	seedVal := cmd.Int("seed")
	var rngSeed int64
	if seedVal == 0 {
		rngSeed = time.Now().UnixNano()
	} else {
		rngSeed = int64(seedVal)
	}
	rng := rand.New(rand.NewSource(rngSeed))

	// ── generate ──────────────────────────────────────────────────────────
	prompt := cmd.String("prompt")
	length := int(cmd.Int("length"))
	temperature := float32(cmd.Float("temperature"))

	output := generateText(model, vocab, prompt, length, temperature, rng)

	if cmd.Bool("show-prompt") {
		fmt.Print(output)
	} else {
		// strip the prompt prefix from the output
		if len(output) > len(prompt) {
			fmt.Print(output[len(prompt):])
		} else {
			fmt.Print(output)
		}
	}
	fmt.Println()

	return nil
}

// ── shared helper ─────────────────────────────────────────────────────────────

// generateText encodes the prompt, runs the model, and returns prompt+generated text.
// Unknown characters in the prompt are silently dropped with a warning.
func generateText(model *rnn.CharRNN, vocab *data.Vocab, prompt string, length int, temperature float32, rng *rand.Rand) string {
	// encode prompt – skip any character not in vocab with a warning
	var seedIDs []int
	for i := 0; i < len(prompt); i++ {
		id, ok := vocab.CharToIdx[prompt[i]]
		if !ok {
			fmt.Fprintf(os.Stderr, "quirrell: warning: character %q not in vocabulary, skipping\n", prompt[i])
			continue
		}
		seedIDs = append(seedIDs, id)
	}

	if len(seedIDs) == 0 {
		// pick a random start token
		seedIDs = []int{rng.Intn(vocab.VocabSize())}
		fmt.Fprintln(os.Stderr, "quirrell: empty seed after filtering – using a random start token")
	}

	generated := model.Sample(seedIDs, length, temperature, rng)

	// decode: seed + generated
	all := append(append([]int(nil), seedIDs...), generated...)
	return data.Decode(all, vocab)
}
