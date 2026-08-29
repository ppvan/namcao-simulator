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
			&cli.Float64Flag{
				Name:  "epochs",
				Value: 20,
				Usage: "passes over the corpus (the training budget)",
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
			&cli.Float64Flag{
				Name:  "log-every-epochs",
				Value: 0,
				Usage: "epochs between train-loss lines (0 = ten per epoch, negative = silent)",
			},
			&cli.Float64Flag{
				Name:  "eval-every-epochs",
				Value: 1,
				Usage: "epochs between validation passes (negative = disabled)",
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

	// tokenizeOpts is the single source of truth for how text gets
	// tokenized in this run. It's saved into the checkpoint below so
	// `predict` can reproduce it exactly rather than hardcoding its own copy.
	tokenizeOpts := data.DefaultOptions

	// ── load corpus ───────────────────────────────────────────────────────
	dataPath := cmd.String("data")
	fmt.Printf("quirrell: loading corpus from %q\n", dataPath)
	text, err := data.LoadCorpus(dataPath)
	if err != nil {
		return fmt.Errorf("load corpus: %w", err)
	}

	vocab := data.BuildVocab(text, tokenizeOpts)
	ids := vocab.Encode(text, tokenizeOpts)
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
	epochs := cmd.Float("epochs")
	batch := int(cmd.Int("batch"))
	if batch < 1 {
		batch = 1
	}

	// The live BPTT cache stays resident between forward and backward. Once
	// batch*seqLen pushes it past L3 the backward pass becomes a DRAM stream
	// and chars/s drops, so surface it before committing to a long run.
	hidden := model.HiddenSize()
	cacheMB := float64(batch*seqLen*(7*hidden+model.Embed.Weights.Columns)*4) / 1e6

	totalChars := int(epochs * float64(len(train)))
	fmt.Printf("quirrell: embed %d | hidden %d | params %d | batch %d | seq %d\n",
		model.Embed.Weights.Columns, hidden, totalParams, batch, seqLen)
	fmt.Printf("quirrell: %d chars/step | BPTT cache %.1f MB | %.1f epochs (%d chars)\n",
		batch*seqLen, cacheMB, float64(totalChars)/float64(len(train)), totalChars)

	// ── train ─────────────────────────────────────────────────────────────
	cfg := rnn.TrainConfig{
		Batch:       batch,
		SeqLen:      seqLen,
		Epochs:      epochs,
		ClipNorm:    float32(cmd.Float("clip")),
		LogEvery:    cmd.Float("log-every-epochs"),
		EvalEvery:   cmd.Float("eval-every-epochs"),
		EvalWindows: int(cmd.Int("eval-windows")),
	}

	start := time.Now()
	if err := model.Train(train, val, opt, cfg); err != nil {
		return fmt.Errorf("training: %w", err)
	}
	elapsed := time.Since(start)
	fmt.Printf("\nquirrell: training finished in %s (%.0f chars/s)\n",
		elapsed.Round(time.Second),
		float64(totalChars)/elapsed.Seconds())

	// ── save checkpoint ───────────────────────────────────────────────────
	saveOpts := rnn.SaveOptions{Vocab: vocab, TokenizeOptions: tokenizeOpts}
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
	sample := generateText(model, vocab, "Tôi ", 300, 0, 0.8, tokenizeOpts, rand.New(rand.NewSource(42)))
	fmt.Println(sample)

	return nil
}

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
				Usage:   "maximum number of tokens to generate",
			},
			&cli.IntFlag{
				Name:  "sentences",
				Value: 0,
				Usage: "stop after this many sentences ('.', '!', '?', '…'); 0 = generate --length tokens",
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
	// tokenizeOpts came from training (saved into the checkpoint), so the
	// prompt below gets tokenized exactly the way the corpus was — no
	// separate flag or hardcoded default to keep in sync by hand.
	tokenizeOpts := res.TokenizeOptions
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
	sentences := int(cmd.Int("sentences"))
	temperature := float32(cmd.Float("temperature"))

	output := generateText(model, vocab, prompt, length, sentences, temperature, tokenizeOpts, rng)

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

func generateText(model *rnn.CharRNN, vocab *data.Vocab, prompt string, maxTokens, sentences int, temperature float32, opts data.Options, rng *rand.Rand) string {
	// tokenize the prompt the same way the corpus was tokenized for
	// training, so casing/punctuation rules match what the vocab expects.
	tokens := data.Tokenize(prompt, opts)

	seedIDs := make([]int, 0, len(tokens))
	for _, tok := range tokens {
		id, ok := vocab.TokenToIdx[tok]
		if !ok {
			fmt.Fprintf(os.Stderr, "quirrell: warning: token %q not in vocabulary, using <UNK>\n", tok)
			id = data.UNK_ID
		}
		seedIDs = append(seedIDs, id)
	}

	if len(seedIDs) == 0 {
		// pick a random start token
		seedIDs = []int{rng.Intn(vocab.VocabSize())}
		fmt.Fprintln(os.Stderr, "quirrell: empty seed after tokenizing – using a random start token")
	}

	// Specials are structural markers, not prose: force their probability to
	// zero so generation can never emit <UNK> (or restart the document with
	// <BOS>/<STORY>) no matter how much weight the model learned to give them.
	banned := []int{data.PAD_ID, data.UNK_ID, data.BOS_ID, data.EOS_ID, data.STORY_ID}

	// With sentences > 0, stop as soon as that many sentence-final tokens
	// have been emitted; maxTokens stays as a hard cap so a model that never
	// produces '.' can't run away.
	var stop func(id int) bool
	if sentences > 0 {
		count := 0
		stop = func(id int) bool {
			if id < 0 || id >= len(vocab.IdxToToken) {
				return false
			}
			if _, ok := data.SentenceEnders[vocab.IdxToToken[id]]; ok {
				count++
			}
			return count >= sentences
		}
	}

	generated := model.SampleMasked(seedIDs, maxTokens, temperature, banned, stop, rng)

	// decode: seed + generated
	all := append(append([]int(nil), seedIDs...), generated...)
	return vocab.Decode(all)
}
