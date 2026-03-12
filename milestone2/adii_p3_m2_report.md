# Project 3 — Milestone 2 Report

## Neural Network — MNIST Digit Classification

## cnet: adii

## Performance Benchmarks

α = 0.01, 5 epochs, Midway3 cluster.
CPU: caslake partition (Intel Xeon). GPU: V100 (sm_70).

| Version | Tile Size | Core Count | Time to Solution (s) | | Grind Rate (samples/s) | | Accuracy (%) | |
|---|---|---|---|---|---|---|---|---|
| | | | Batch 256 | Batch 512 | Batch 256 | Batch 512 | Batch 256 | Batch 512 |
| CPU (no BLAS) | 32 | 32 | 5.31 | 4.41 | 56,508 | 67,993 | 85.74 | 72.73 |
| CPU (BLAS)    | N/A | 16 | 2.68 | 3.02 | 111,877 | 99,203 | 85.74 | 72.73 |
| GPU (baseline) | N/A | N/A | 1.24 | 1.46 | 242,721 | 205,361 | 85.84 | 72.80 |

---

## Notes

**Parameters:**
784 (input) → 128 (ReLU) → 256 (ReLU) → 10 (Softmax)
SGD optimizer, α=0.01, 5 epochs.

**Core Count (CPU):** Table reports the best single thread count for batch 256. For CPU (no BLAS), 48 threads was faster for batch 512 (4.41s vs 4.77s at 32 threads). For CPU (BLAS), 32 threads was marginally faster for batch 512 (3.02s vs 3.08s at 16 threads).

**Tile size:** TILE=32 was optimal across all tested configurations. TILE=64 was consistently slower — larger tiles exceed the working set of our small matrices (max dimension 256), leading to cache inefficiency rather than reuse.

**OpenBLAS thread scaling:** BLAS performance degrades beyond 16–32 threads. Adding more OpenMP threads introduces contention rather than additional parallelism. 48 threads was 1.8× slower than 16 threads for batch 256.

**GPU baseline:** Naive implementation — one thread per output matrix element, inner k-loop over global memory. No shared memory tiling or cuBLAS. Thread block 16×16 for 2D kernels (GEMM, bias broadcast), 256 threads for 1D kernels (ReLU, softmax, rowsum, SGD).

**GPU vs CPU speedup (batch 256):** 4.3× over hand-coded GEMM, 2.2× over OpenBLAS — from raw GPU parallelism alone with no memory optimizations.

**Accuracy vs batch size:** Batch 512 produces 117 gradient updates/epoch vs 234 for batch 256. Fewer updates means less convergence in 5 epochs, resulting in lower accuracy (72.73% vs 85.74%).
