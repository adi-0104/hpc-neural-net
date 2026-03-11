#pragma once

// Block size for 2D thread blocks (BLOCK_SIZE x BLOCK_SIZE threads per block)
// Each thread computes one element of the output matrix C
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 16
#endif

__global__ void gemm_nn_kernel(int M, int N, int K,
                               float alpha,
                               const float *A, int lda,
                               const float *B, int ldb,
                               float beta,
                               float *C, int ldc);

__global__ void gemm_tn_kernel(int M, int N, int K,
                               float alpha,
                               const float *A, int lda,
                               const float *B, int ldb,
                               float beta,
                               float *C, int ldc);

__global__ void gemm_nt_kernel(int M, int N, int K,
                               float alpha,
                               const float *A, int lda,
                               const float *B, int ldb,
                               float beta,
                               float *C, int ldc);

// forward pass calc
void gpu_gemm_nn(int M, int N, int K,
                 float alpha,
                 const float *d_A, int lda,
                 const float *d_B, int ldb,
                 float beta,
                 float *d_C, int ldc);

// Backprop delta
void gpu_gemm_tn(int M, int N, int K,
                 float alpha,
                 const float *d_A, int lda,
                 const float *d_B, int ldb,
                 float beta,
                 float *d_C, int ldc);

// Backprop dW multiplocations
void gpu_gemm_nt(int M, int N, int K,
                 float alpha,
                 const float *d_A, int lda,
                 const float *d_B, int ldb,
                 float beta,
                 float *d_C, int ldc);
