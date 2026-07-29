package main

import (
	"context"
	"fmt"

	"github.com/ppvan/gpt/pkg/core"
	"github.com/ppvan/gpt/pkg/data"
	"github.com/ppvan/gpt/pkg/mlp"
)

func main() {
	digits()
}

func digits() {
	d, err := data.LoadCSV("cmd/ndigits/digits.csv", 64, false)
	if err != nil {
		panic(err)
	}
	d = d.Shuffle()
	train, val, _ := d.Split(0.80, 0.20, 0)

	model := mlp.NewDense(
		mlp.NewLinear(64, 32),
		mlp.NewBatchNorm(32),
		core.NewLeakyReLU(0.02),

		mlp.NewLinear(32, 64),
		mlp.NewBatchNorm(64),
		core.NewLeakyReLU(0.02),

		mlp.NewLinear(64, 16),
		mlp.NewBatchNorm(16),
		core.NewLeakyReLU(0.02),

		mlp.NewLinear(16, 10),
	)
	opt := core.NewAdam(0.001, 0.9, 0.999, 1e-8)
	net := mlp.NewNetwork(model, core.CrossEntropy(), opt)

	epochs := 256
	batchSize := 64
	ctx := context.Background()
	for m := range net.Fit(ctx, train, epochs, batchSize) {
		fmt.Printf("epoch=%d loss=%.6f\r", m.Epoch, m.Loss)
	}
	fmt.Println()

	metrics, err := net.Evaluate(val)
	if err != nil {
		panic(err)
	}
	fmt.Println(metrics)
}
