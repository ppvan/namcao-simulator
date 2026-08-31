package main

import (
	"context"
	"fmt"

	"github.com/ppvan/gpt/pkg/core"
	"github.com/ppvan/gpt/pkg/data"
	"github.com/ppvan/gpt/pkg/mlp"
)

func main() {
	xor()

}

func xor() {
	x := core.NewMat(
		4, 2, []float32{0, 0, 0, 1, 1, 0, 1, 1},
	)
	labels := core.NewMat(4, 1, []float32{0, 1, 1, 0})
	data := data.Data{
		X: x, Y: labels,
	}
	model := mlp.NewDense(
		mlp.NewLinear(2, 4),
		core.NewLeakyReLU(0.1),
		mlp.NewLinear(4, 16),
		core.NewLeakyReLU(0.1),
		mlp.NewLinear(16, 2),
	)
	opt := core.NewGradient(0.01)
	net := mlp.NewNetwork(model, core.CrossEntropy(), opt)

	ctx := context.Background()
	for m := range net.Fit(ctx, data, 5000, 4) {
		fmt.Printf("epoch=%d loss=%.4f\r", m.Epoch, m.Loss)
	}
	fmt.Println()

	fmt.Println("===== FINAL PREDICTIONS =====")
	for i := 0; i < x.Rows; i++ {
		input := x.Slice(i, i+1)
		pred, err := net.Predict(input)
		if err != nil {
			fmt.Printf("%v | %v = error: %v\n", input.Get(0, 0), input.Get(0, 1), err)
			continue
		}
		fmt.Printf("%v | %v = %d\n", input.Get(0, 0), input.Get(0, 1), pred.Class)
	}
}
