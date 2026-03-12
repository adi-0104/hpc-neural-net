#pragma once

#ifndef TILE
#define TILE 32
#endif

// tiling without cblas
// C(MxN) = alpha * A(MxK) * B(KxN) + beta * C(MxN)
void my_gemm_nn(int M, int N, int K, float alpha,
                float *A, int lda,
                float *B, int ldb,
                float beta, float *C, int ldc);

// C(MxN) = alpha * A^T(KxM) * B(KxN)   + beta*C
void my_gemm_tn(int M, int N, int K, float alpha,
                float *A, int lda,
                float *B, int ldb,
                float beta, float *C, int ldc);

// C(MxN) = alpha * A(MxK)   * B^T(N x K) + beta*C
void my_gemm_nt(int M, int N, int K, float alpha,
                float *A, int lda,
                float *B, int ldb,
                float beta, float *C, int ldc);

// idef for manual vs cblas
#ifdef USE_BLAS
#include <cblas.h>
static inline void gemm_nn(int M, int N, int K, float alpha,
                           float *A, int lda, float *B, int ldb,
                           float beta, float *C, int ldc)
{
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}
static inline void gemm_tn(int M, int N, int K, float alpha,
                           float *A, int lda, float *B, int ldb,
                           float beta, float *C, int ldc)
{
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}
static inline void gemm_nt(int M, int N, int K, float alpha,
                           float *A, int lda, float *B, int ldb,
                           float beta, float *C, int ldc)
{
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}
#else
// Without BLAS: call hand-coded versions
static inline void gemm_nn(int M, int N, int K, float alpha,
                           float *A, int lda, float *B, int ldb,
                           float beta, float *C, int ldc)
{
    my_gemm_nn(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}
static inline void gemm_tn(int M, int N, int K, float alpha,
                           float *A, int lda, float *B, int ldb,
                           float beta, float *C, int ldc)
{
    my_gemm_tn(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}
static inline void gemm_nt(int M, int N, int K, float alpha,
                           float *A, int lda, float *B, int ldb,
                           float beta, float *C, int ldc)
{
    my_gemm_nt(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}
#endif
