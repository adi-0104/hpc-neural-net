#include <stdlib.h>
#include "gemm.h"
// GEMM_NN: C(M×N) = alpha * A(M×K) * B(K×N) + beta * C(M×N)
void my_gemm_nn(int M, int N, int K, float alpha,
                float *A, int lda,
                float *B, int ldb,
                float beta, float *C, int ldc)
{
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            C[i * ldc + j] *= beta;
        }
    }
// tile loop. ikj loop to sequentially scan b matrix
#pragma omp parallel for collapse(2) schedule(static)
    for (int ii = 0; ii < M; ii += TILE)
    {
        for (int jj = 0; jj < N; jj += TILE)
        {
            for (int kk = 0; kk < K; kk += TILE)
            {
                int imax = ii + TILE < M ? ii + TILE : M;
                int kmax = kk + TILE < K ? kk + TILE : K;
                int jmax = jj + TILE < N ? jj + TILE : N;

                // main loop
                for (int i = ii; i < imax; i++)
                {
                    for (int k = kk; k < kmax; k++)
                    {
                        float a_ik = alpha * A[i * lda + k];
                        for (int j = jj; j < jmax; j++)
                        {
                            C[i * ldc + j] += a_ik * B[k * ldb + j];
                        }
                    }
                }
            }
        }
    }
}

// GEMM_TN: C(M×N) = alpha * A^T(M×K) * B(K×N) + beta * C(M×N)
void my_gemm_tn(int M, int N, int K, float alpha,
                float *A, int lda,
                float *B, int ldb,
                float beta, float *C, int ldc)
{
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            C[i * ldc + j] *= beta;
        }
    }
// tile loop. kij loop to sequentially scan a_t matrix
#pragma omp parallel for collapse(2) schedule(static)
    for (int ii = 0; ii < M; ii += TILE)
    {
        for (int jj = 0; jj < N; jj += TILE)
        {
            for (int kk = 0; kk < K; kk += TILE)
            {
                int imax = ii + TILE < M ? ii + TILE : M;
                int kmax = kk + TILE < K ? kk + TILE : K;
                int jmax = jj + TILE < N ? jj + TILE : N;

                // main loop
                for (int k = kk; k < kmax; k++)
                {
                    for (int i = ii; i < imax; i++)
                    {
                        float a_ik = alpha * A[k * lda + i];
                        for (int j = jj; j < jmax; j++)
                        {
                            C[i * ldc + j] += a_ik * B[k * ldb + j];
                        }
                    }
                }
            }
        }
    }
}

// GEMM_NT: C(M×N) = alpha * A(M×K) * B^T(K×N) + beta * C(M×N)
void my_gemm_nt(int M, int N, int K, float alpha,
                float *A, int lda,
                float *B, int ldb,
                float beta, float *C, int ldc)
{
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            C[i * ldc + j] *= beta;
        }
    }
// tile loop. ijk loop to sequentially scan a and b_t matrix
#pragma omp parallel for collapse(2) schedule(static)
    for (int ii = 0; ii < M; ii += TILE)
    {
        for (int jj = 0; jj < N; jj += TILE)
        {
            for (int kk = 0; kk < K; kk += TILE)
            {
                int imax = ii + TILE < M ? ii + TILE : M;
                int kmax = kk + TILE < K ? kk + TILE : K;
                int jmax = jj + TILE < N ? jj + TILE : N;

                // main loop
                for (int i = ii; i < imax; i++)
                {
                    for (int j = jj; j < jmax; j++)
                    {
                        float sum = 0.0f;
                        for (int k = kk; k < kmax; k++)
                        {
                            sum += A[i * lda + k] * B[j * ldb + k];
                        }
                        C[i * ldc + j] += alpha * sum;
                    }
                }
            }
        }
    }
}
