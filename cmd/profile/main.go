package main

import (
	"flag"
	"fmt"
	"log"
	"math/rand"
	"os"
	"runtime"
	"runtime/pprof"
	"runtime/trace"

	"github.com/ppvan/gpt/pkg/core"
	"github.com/ppvan/gpt/pkg/data"
	"github.com/ppvan/gpt/pkg/rnn"
)

func main() {
	cpuprofile := flag.String("cpuprofile", "", "write cpu profile to file")
	memprofile := flag.String("memprofile", "", "write heap profile to file")
	traceFile := flag.String("trace", "", "write execution trace to file")
	steps := flag.Int("steps", 2000, "override training steps for a shorter profiling run")
	skipSample := flag.Bool("skip-sample", true, "skip generation/sampling after training (default: skip, since it's not the bottleneck)")
	flag.Parse()

	// --- CPU profile ---
	if *cpuprofile != "" {
		f, err := os.Create(*cpuprofile)
		if err != nil {
			log.Fatalf("create cpuprofile: %v", err)
		}
		defer f.Close()
		if err := pprof.StartCPUProfile(f); err != nil {
			log.Fatalf("start cpu profile: %v", err)
		}
		defer pprof.StopCPUProfile()
	}

	// --- Execution trace (goroutine scheduling, GC pauses, blocking) ---
	if *traceFile != "" {
		f, err := os.Create(*traceFile)
		if err != nil {
			log.Fatalf("create trace file: %v", err)
		}
		defer f.Close()
		if err := trace.Start(f); err != nil {
			log.Fatalf("start trace: %v", err)
		}
		defer trace.Stop()
	}

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
		Steps:       *steps, // overridden — keep this short for profiling runs
		ClipNorm:    5.0,
		LogEvery:    200,
		EvalEvery:   1 << 30, // effectively disable eval during profiling, it pollutes the profile
		EvalWindows: 50,
	}

	params := 0
	for _, p := range model.Params() {
		params += p.Rows * p.Columns
	}
	fmt.Printf("lstm: embed 32, hidden 256, %d params, profiling %d steps\n", params, cfg.Steps)

	if err := model.Train(train, val, opt, cfg); err != nil {
		log.Fatalf("train: %v", err)
	}

	// --- Heap profile (taken right after training, before any GC-triggering churn) ---
	if *memprofile != "" {
		f, err := os.Create(*memprofile)
		if err != nil {
			log.Fatalf("create memprofile: %v", err)
		}
		defer f.Close()
		runtime.GC() // get up-to-date statistics
		if err := pprof.WriteHeapProfile(f); err != nil {
			log.Fatalf("write heap profile: %v", err)
		}
	}

	if *skipSample {
		return
	}

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
