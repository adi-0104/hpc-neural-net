# Milestone 3 — cuBLAS GPU Training

MNIST digit classifier (784 → 128 → 256 → 10) trained with SGD.
α=0.1, batch=500, 50 epochs, 50K train / 10K validation split.

## Structure

```
cpu/        tiled GEMM + OpenMP implementation
gpu/        CUDA implementation (native kernels + cuBLAS)
bash/       SLURM batch scripts
```

## Building

**CPU:**
```bash
cd cpu
make TILE=32          # native tiled GEMM
make TILE=32 BLAS=1   # OpenBLAS
```

**GPU:**
```bash
cd gpu
make                  # native CUDA kernels
make CUBLAS=1         # cuBLAS GEMMs
```

## Running

**CPU (set threads via OMP_NUM_THREADS):**
```bash
export OMP_NUM_THREADS=32
./nn
```

**GPU:**
```bash
./nn_gpu_v100_500
```

## Plots

```bash
uv run --with matplotlib make_plots.py
```
