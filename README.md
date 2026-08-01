## Nam Cao Simulator

A **1 million parameter character-level language model** trained from scratch to generate text in the style of **Nam Cao**. The entire training stack was implemented in Go without relying on machine learning frameworks.

### Highlights

* Custom RNN and LSTM implementations
* Adam optimizer with automatic differentiation
* Hand-written `float32` math kernels for improved CPU performance
* SIMD-accelerated matrix operations
* Model checkpoint save/load support for training continuation and inference
* End-to-end training pipeline built from scratch

### Preview

<img width="2078" height="283" alt="Nam Cao Simulator output" src="https://github.com/user-attachments/assets/0b35ef0b-1c4f-4810-8c99-a35df0990079" />
