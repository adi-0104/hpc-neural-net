# Project 3, Milestone 1 — Serial Neural Network

## Overview
Feedforward neural network for MNIST handwritten digit classification. Trained using mini-batch stochastic gradient descent with backpropagation.

## Architecture
```
Input (784)  →  Hidden 1 (128, ReLU)  →  Hidden 2 (256, ReLU)  →  Output (10, Softmax)
```
- Loss: Cross-entropy
- Optimizer: Mini-batch SGD
- Weight init: Kaiming uniform — Uniform(-1/sqrt(n_in), 1/sqrt(n_in))

## Data
Download the MNIST dataset from http://yann.lecun.com/exdb/mnist/ and place the four files in a `data/` folder at the project root.

If you have the `.gz` files, unzip them first:
```bash
gunzip -k data/*.gz
```

## Build and Run
```bash
cd milestone1/
make
./nn
```

## Expected Output
```
Train: 60000 images, Test: 10000 images
Epoch 1/20   cost: ...  accr: ...%
...
Epoch 20/20  cost: ...  accr: ...%
Training time: ...s
Grind rate: ... samples/sec
Inference time: ...s
Test accuracy: ~97.90%
Test Cost: ...
```
