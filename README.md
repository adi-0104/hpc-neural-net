# Neural Network from Scratch

MNIST digit classifier built from scratch in C and CUDA for the High-Performance Computing course (MPCS 51087) at UChicago. No ML frameworks - just raw matrix math, backprop, and progressively more parallel implementations across three milestones.

---

## Architecture

```
Input (784)  ->  Hidden 1 (128, ReLU)  ->  Hidden 2 (256, ReLU)  ->  Output (10, Softmax)
```

- Loss: Cross-entropy
- Optimizer: Mini-batch SGD
- Weight init: Kaiming uniform

---

## What I worked on

**Milestone 1 - Serial C**

Wrote the full forward pass, backprop, and SGD weight updates from scratch in C. Everything single-threaded. This was the baseline to benchmark everything else against.

**Milestone 2 - CPU parallelism + GPU baseline**

The main bottleneck in training is matrix multiplication (GEMM). Replaced the naive triple loop with:
- Tiled GEMM with OpenMP across CPU cores
- A baseline CUDA kernel (one thread per output element, global memory only, no shared memory tiling)

**Milestone 3 - Full training convergence + cuBLAS**

Extended to full 50-epoch training and added OpenBLAS on CPU and cuBLAS on GPU as two more variants. Ran all four implementations on the Midway3 HPC cluster.

---

## Results

All four implementations converge to the same accuracy (~97.7%), confirming the parallelism didn't break anything.

**Final benchmarks** (alpha=0.1, batch=500, 50 epochs, 50K train / 10K val):

| Version | Cores | Time (s) | Grind Rate (smp/s) | Accuracy (%) |
|---|---|---|---|---|
| CPU Native (tiled GEMM) | 32 | 39.53 | 63,244 | 97.62 |
| CPU BLAS (OpenBLAS) | 16 | 24.31 | 102,857 | 97.62 |
| GPU Native (CUDA) | - | 6.71 | 372,419 | 97.71 |
| GPU cuBLAS | - | 7.19 | 347,523 | 97.71 |

GPU native was 5.9x faster than CPU native and 3.6x faster than OpenBLAS.

One interesting finding: cuBLAS was actually slightly slower than the hand-written CUDA kernel for this network. For small GEMMs (max dimension 256x784), the per-call dispatch overhead from cublasSgemm adds up across ~25,000 calls and outweighs the kernel-level gains. When I tested with larger hidden layers (512, 1024), cuBLAS won by 1.7x - so it's a matrix size thing.

**Validation loss curves (50 epochs):**

![Loss curves](milestone3/loss_curves_overlay.png)

---

## Building and running

Download MNIST from http://yann.lecun.com/exdb/mnist/ and place the four files in `data/`.

**Serial (milestone 1):**
```bash
cd milestone1
make
./nn
```

**CPU parallel:**
```bash
cd milestone3/cpu
make TILE=32
export OMP_NUM_THREADS=32
./nn
```

**CPU with OpenBLAS:**
```bash
make TILE=32 BLAS=1
./nn
```

**GPU native CUDA:**
```bash
cd milestone3/gpu
make
./nn_gpu
```

**GPU with cuBLAS:**
```bash
make CUBLAS=1
./nn_gpu
```

**Plot loss curves:**
```bash
uv run --with matplotlib milestone3/make_plots.py
```

---

## What I learnt

- Writing backprop from scratch in C clarified a lot about how gradient flow actually works, especially getting the matrix dimension ordering right for each layer's weight updates.
- TILE=32 was consistently better than TILE=64 for the GEMM. Larger tiles exceed the working set of our small matrices and hurt cache reuse rather than help it.
- OpenBLAS starts degrading past 16-32 threads due to contention. Adding more OpenMP threads doesn't always mean more speed.
- The cuBLAS vs native result was surprising. It's a good reminder that optimized libraries have overhead that only pays off at larger problem sizes.

## Possible improvements

- Shared memory tiling in the CUDA kernels would likely bring more GPU speedup.
- Could try Adam or momentum-based SGD instead of vanilla SGD.
- Learning rate scheduling might help convergence speed.
- The serial C version could be made cleaner - some memory management is a bit ad hoc.
