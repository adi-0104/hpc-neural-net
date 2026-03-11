#include "gpu_gemm.h"
#include <stdio.h>

// GEMM_NN: C(M×N) = alpha * A(M×K) * B(K×N) + beta * C(M×N)
__global__ void gemm_nn_kernel(int M, int N, int K,
                               float alpha,
                               const float *A, int lda,
                               const float *B, int ldb,
                               float beta,
                               float *C, int ldc)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N)
    {
        float sum = 0.0f;
        for (int k = 0; k < K; k++)
        {
            sum += A[row*lda + k] * B[k * ldb + col];
        }

        C[row*ldc + col] = alpha*sum + beta*C[row*ldc + col];
    }
}

// GEMM_TN: C(M×N) = alpha * A^T(M×K) * B(K×N) + beta * C(M×N)
__global__ void gemm_tn_kernel(int M, int N, int K,
                               float alpha,
                               const float *A, int lda,
                               const float *B, int ldb,
                               float beta,
                               float *C, int ldc)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N)
    {
        float sum = 0.0f;
        for (int k = 0; k < K; k++)
        {
            sum += A[k*lda + row] * B[k * ldb + col];
        }

        C[row*ldc + col] = alpha*sum + beta*C[row*ldc + col];
    }
}

// GEMM_NT: C(M×N) = alpha * A(M×K) * B^T(K×N) + beta * C(M×N)
__global__ void gemm_nt_kernel(int M, int N, int K,
                               float alpha,
                               const float *A, int lda,
                               const float *B, int ldb,
                               float beta,
                               float *C, int ldc)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N)
    {
        float sum = 0.0f;
        for (int k = 0; k < K; k++)
        {
            sum += A[row*lda + k] * B[col * ldb + k];
        }

        C[row*ldc + col] = alpha*sum + beta*C[row*ldc + col];
    }
}

static inline int ceil_div(int n, int bs) { return (n + bs - 1) / bs; }

void gpu_gemm_nn(int M, int N, int K,
                 float alpha,
                 const float *d_A, int lda,
                 const float *d_B, int ldb,
                 float beta,
                 float *d_C, int ldc)
{
    // grid: columns along x, rows along y
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(ceil_div(N, BLOCK_SIZE), ceil_div(M, BLOCK_SIZE));

    gemm_nn_kernel<<<grid, block>>>(M, N, K, alpha, d_A, lda, d_B, ldb, beta, d_C, ldc);
}

void gpu_gemm_tn(int M, int N, int K,
                 float alpha,
                 const float *d_A, int lda,
                 const float *d_B, int ldb,
                 float beta,
                 float *d_C, int ldc)
{
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(ceil_div(N, BLOCK_SIZE), ceil_div(M, BLOCK_SIZE));

    gemm_tn_kernel<<<grid, block>>>(M, N, K, alpha, d_A, lda, d_B, ldb, beta, d_C, ldc);
}

void gpu_gemm_nt(int M, int N, int K,
                 float alpha,
                 const float *d_A, int lda,
                 const float *d_B, int ldb,
                 float beta,
                 float *d_C, int ldc)
{
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(ceil_div(N, BLOCK_SIZE), ceil_div(M, BLOCK_SIZE));

    gemm_nt_kernel<<<grid, block>>>(M, N, K, alpha, d_A, lda, d_B, ldb, beta, d_C, ldc);
}
