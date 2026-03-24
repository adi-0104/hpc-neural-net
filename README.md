# Neural Network from Scratch — CPU & GPU

A feedforward neural network built entirely from scratch in C and CUDA, without any ML frameworks. Trained to recognize handwritten digits from the MNIST dataset — achieving **~97.7% accuracy**. Developed as part of the **High-Performance Computing** course (MPCS 51087) at the University of Chicago.

The goal wasn't just to get the network working — it was to make it *fast*, by progressively replacing naive code with parallelized CPU and GPU implementations and measuring the speedup at each step.

---

## What I Built

### The Neural Network
A 3-layer fully-connected network trained on the classic MNIST handwritten digit dataset (70,000 images of digits 0–9):

```
Input (784 pixels)  →  Hidden Layer 1 (128 neurons, ReLU)  →  Hidden Layer 2 (256 neurons, ReLU)  →  Output (10 classes, Softmax)
```

- **Training:** Mini-batch stochastic gradient descent (SGD) with backpropagation
- **Loss function:** Cross-entropy
- **Weight initialization:** Kaiming uniform (prevents vanishing/exploding gradients)
- **Final accuracy:** ~97.7% on the 10,000-image test set

Everything — the matrix math, the activation functions, the gradient calculations, the weight updates — was written by hand. No PyTorch, no NumPy, no ML libraries.

### The Performance Journey (Milestone by Milestone)

The project was structured as a progressive optimization challenge:

**Milestone 1 — Serial Baseline (C)**
A clean, correct serial implementation. Single-threaded, running entirely on CPU. This was the reference point for all future speedups.
- Accuracy: ~97.7%
- Speed: ~4,700 samples/sec

**Milestone 2 — Parallelism (OpenMP + CUDA)**
The core bottleneck in any neural network is matrix multiplication (GEMM). I replaced the naive triple-loop with:
- **Tiled GEMM with OpenMP** — cache-friendly tiling + multi-core parallelism across 32 CPU threads → ~14× faster than serial
- **Baseline CUDA kernel** — moved the computation to a GPU (NVIDIA V100), one thread per output element → **~52× faster** than serial

**Milestone 3 — Full Training + cuBLAS**
Extended to full 50-epoch training convergence and added cuBLAS (NVIDIA's optimized GPU math library) as a fourth implementation:

| Implementation | Time (50 epochs) | Throughput | Accuracy |
|---|---|---|---|
| CPU — Tiled GEMM (32 cores) | 39.5s | 63,244 smp/s | 97.62% |
| CPU — OpenBLAS (16 cores) | 24.3s | 102,857 smp/s | 97.62% |
| GPU — Native CUDA | **6.7s** | **372,419 smp/s** | 97.71% |
| GPU — cuBLAS | 7.2s | 347,523 smp/s | 97.71% |

**The GPU native implementation was 5.9× faster than the best CPU version** and 372K samples/sec throughput. An interesting finding: for this network's matrix sizes (max 256×784), hand-written CUDA kernels actually outperformed cuBLAS because cuBLAS's per-call dispatch overhead outweighs its kernel optimizations at small dimensions.

---

## Tech Stack

- **Language:** C (CPU), CUDA C (GPU)
- **Libraries:** OpenMP (CPU threading), OpenBLAS (CPU BLAS), cuBLAS (GPU BLAS)
- **Hardware:** Intel Xeon (32-core), NVIDIA V100 / A100
- **Cluster:** University of Chicago Midway3 HPC cluster (SLURM job scheduling)
- **Dataset:** MNIST (60K training, 10K test images)

---

## Project Structure

```
milestone1/         # Serial C implementation
milestone2/
├── nn/             # Parallelized CPU (OpenMP + tiled GEMM)
└── gpu/            # Baseline CUDA implementation
milestone3/
├── cpu/            # Final CPU: tiled GEMM + OpenBLAS
│   ├── main.c
│   ├── network.c   # Forward pass, backprop, SGD
│   ├── gemm.c      # Tiled matrix multiply
│   └── mnist.c     # Data loading
└── gpu/            # Final GPU: native CUDA + cuBLAS
    └── src/
```

---

## Building & Running

**Prerequisites:** GCC, CUDA toolkit, OpenBLAS, MNIST data files

Download MNIST from [yann.lecun.com/exdb/mnist](http://yann.lecun.com/exdb/mnist/) and place the four files in a `data/` folder.

**CPU (serial baseline):**
```bash
cd milestone1
make
./nn
```

**CPU (parallel, tiled GEMM + OpenMP):**
```bash
cd milestone3/cpu
make TILE=32
export OMP_NUM_THREADS=32
./nn
```

**CPU (OpenBLAS):**
```bash
cd milestone3/cpu
make TILE=32 BLAS=1
./nn
```

**GPU (native CUDA):**
```bash
cd milestone3/gpu
make
./nn_gpu
```

**GPU (cuBLAS):**
```bash
cd milestone3/gpu
make CUBLAS=1
./nn_gpu
```

**Generate convergence plots:**
```bash
uv run --with matplotlib milestone3/make_plots.py
```

---

## Results

All four implementations converge to the same validation loss (~0.082) and accuracy (~97.7%), confirming correctness. The loss curves below show convergence across 50 epochs:

![Loss curves](milestone3/loss_curves_overlay.png)

---

## Course Context

Built as part of **MPCS 51087 — High-Performance Computing** at the University of Chicago (Winter 2026). The project was designed to teach parallel programming by having students build something computationally intensive and optimize it systematically — not just make it parallel, but understand *why* and *how much* it speeds up at each step.
