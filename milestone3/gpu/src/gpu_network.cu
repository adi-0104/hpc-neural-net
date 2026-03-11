#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include "gpu_network.h"
#include "gpu_gemm.h"

// macro for error handling
#define CUDA_CHECK(call)                                          \
    do                                                            \
    {                                                             \
        cudaError_t err = (call);                                 \
        if (err != cudaSuccess)                                   \
        {                                                         \
            fprintf(stderr, "CUDA error %s:%d: %s\n",             \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1);                                              \
        }                                                         \
    } while (0)

// Helper for ceiling division
static inline int ceil_div(int n, int bs) { return (n + bs - 1) / bs; }

static uint32_t prng_state = 123456789;
static inline uint32_t xorshift32()
{
    uint32_t x = prng_state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    prng_state = x;
    return x;
}
static inline float rand_uniform()
{
    return (float)xorshift32() / (float)UINT32_MAX;
}

__global__ void relu_kernel(const float *Z, float *A, int total)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total)
    {
        A[i] = Z[i] > 0 ? Z[i] : 0.0f;
    }
}

__global__ void relu_deriv_kernel(float *Delta, const float *Z, int total)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total)
    {
        Delta[i] *= (Z[i] > 0 ? 1.0f : 0.0f);
    }
}


__global__ void softmax_kernel(const float *Z, float *A, int n_classes, int bs)
{
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s < bs)
    {
        float z_max = -INFINITY;
        for (int j = 0; j < n_classes; j++)
        {
            z_max = Z[j * bs + s] > z_max ? Z[j * bs + s] : z_max;
        }

        // exp of z
        float sum_exp_z = 0;
        for (int i = 0; i < n_classes; i++)
        {
            A[i * bs + s] = expf(Z[i * bs + s] - z_max);
            sum_exp_z += A[i * bs + s];
        }

        for (int i = 0; i < n_classes; i++)
        {
            // a3 = exp_z / sum(exp_z)
            A[i * bs + s] /= sum_exp_z;
        }
    }
}

__global__ void bias_broadcast_kernel(float *Z, const float *biases, int n_out, int bs)
{
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (j < n_out && s < bs)
    {
        Z[j * bs + s] = biases[j];
    }
}

__global__ void output_delta_kernel(float *Delta, const float *A, const float *Y_onehot, int total)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total)
    {
        Delta[i] = A[i] - Y_onehot[i];
    }
}

__global__ void rowsum_kernel(const float *Delta, float *db, int n_out, int bs)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < n_out)
    {
        float delta_sum = 0.0f;
        for (int s = 0; s < bs; s++)
        {
            delta_sum += Delta[j * bs + s];
        }
        db[j] += delta_sum;
    }
}

__global__ void sgd_update_kernel(float *param, const float *grad, float scale, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        param[i] -= scale * grad[i];
    }
}


GPULayer gpu_init_layer(int n_in, int n_out, int max_batch,
                        const float *h_weights, const float *h_biases)
{
    GPULayer l;
    l.n_in = n_in;
    l.n_out = n_out;
    l.max_batch = max_batch;

    // Allocate device arrays
    CUDA_CHECK(cudaMalloc(&l.d_weights, n_out * n_in * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&l.d_biases, n_out * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&l.d_Z, n_out * max_batch * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&l.d_A, n_out * max_batch * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&l.d_Delta, n_out * max_batch * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&l.d_dW, n_out * n_in * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&l.d_db, n_out * sizeof(float)));

    if (h_weights && h_biases)
    {
        // Copy pre-computed weights from host
        CUDA_CHECK(cudaMemcpy(l.d_weights, h_weights,
                              n_out * n_in * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(l.d_biases, h_biases,
                              n_out * sizeof(float), cudaMemcpyHostToDevice));
    }
    else
    {
        // init on host 
        float *tmp_w = (float *)malloc(n_out * n_in * sizeof(float));
        float *tmp_b = (float *)malloc(n_out * sizeof(float));
        float limit = 1.0f / sqrtf((float)n_in);
        float range = 2.0f * limit;
        for (int i = 0; i < n_out * n_in; i++)
            tmp_w[i] = (rand_uniform() * range) - limit;
        for (int i = 0; i < n_out; i++)
            tmp_b[i] = (rand_uniform() * range) - limit;
        CUDA_CHECK(cudaMemcpy(l.d_weights, tmp_w,
                              n_out * n_in * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(l.d_biases, tmp_b,
                              n_out * sizeof(float), cudaMemcpyHostToDevice));
        free(tmp_w);
        free(tmp_b);
    }

    // init layer buffers
    CUDA_CHECK(cudaMemset(l.d_Z, 0, n_out * max_batch * sizeof(float)));
    CUDA_CHECK(cudaMemset(l.d_A, 0, n_out * max_batch * sizeof(float)));
    CUDA_CHECK(cudaMemset(l.d_Delta, 0, n_out * max_batch * sizeof(float)));
    CUDA_CHECK(cudaMemset(l.d_dW, 0, n_out * n_in * sizeof(float)));
    CUDA_CHECK(cudaMemset(l.d_db, 0, n_out * sizeof(float)));

    return l;
}

void gpu_free_layer(GPULayer *l)
{
    cudaFree(l->d_weights);
    cudaFree(l->d_biases);
    cudaFree(l->d_Z);
    cudaFree(l->d_A);
    cudaFree(l->d_Delta);
    cudaFree(l->d_dW);
    cudaFree(l->d_db);
}

GPUNetwork gpu_init_network(int *layer_sizes, int n_layers, int max_batch,
                            const float *h_weights_flat[],
                            const float *h_biases_flat[])
{
    GPUNetwork net;
    net.n_layers = n_layers;
    net.max_batch = max_batch;
    net.layers = (GPULayer *)malloc(n_layers * sizeof(GPULayer));
    cublasCreate(&net.cublas_handle);

    for (int i = 0; i < n_layers; i++)
    {
        const float *hw = h_weights_flat ? h_weights_flat[i] : NULL;
        const float *hb = h_biases_flat ? h_biases_flat[i] : NULL;
        net.layers[i] = gpu_init_layer(layer_sizes[i], layer_sizes[i + 1],
                                       max_batch, hw, hb);
    }
    return net;
}

void gpu_free_network(GPUNetwork *net)
{
    for (int i = 0; i < net->n_layers; i++)
        gpu_free_layer(&net->layers[i]);
    cublasDestroy(net->cublas_handle);
    free(net->layers);
}



void gpu_zero_gradients(GPUNetwork *net)
{
    for (int i = 0; i < net->n_layers; i++)
    {
        GPULayer *l = &net->layers[i];
        CUDA_CHECK(cudaMemset(l->d_dW, 0, l->n_out * l->n_in * sizeof(float)));
        CUDA_CHECK(cudaMemset(l->d_db, 0, l->n_out * sizeof(float)));
    }
}


void gpu_batched_forward_pass(GPUNetwork *net, const float *d_X_batch, int bs)
{
    for (int layer_idx = 0; layer_idx < net->n_layers; layer_idx++)
    {
        GPULayer *l = &net->layers[layer_idx];
        const float *d_A_prev = (layer_idx == 0) ? d_X_batch
                                                 : net->layers[layer_idx - 1].d_A;

        dim3 bb_block(BLOCK_SIZE, BLOCK_SIZE);
        dim3 bb_grid(ceil_div(bs, BLOCK_SIZE), ceil_div(l->n_out, BLOCK_SIZE));
        bias_broadcast_kernel<<<bb_grid, bb_block>>>(l->d_Z, l->d_biases, l->n_out, bs);

#ifdef USE_CUBLAS
        {
            const float one = 1.0f;
            cublasSgemm(net->cublas_handle,
                        CUBLAS_OP_N, CUBLAS_OP_N,
                        bs, l->n_out, l->n_in,
                        &one,
                        d_A_prev,    bs,
                        l->d_weights, l->n_in,
                        &one,
                        l->d_Z,      bs);
        }
#else
        gpu_gemm_nn(l->n_out, bs, l->n_in,
                    1.0f, l->d_weights, l->n_in,
                    d_A_prev, bs,
                    1.0f, l->d_Z, bs);
#endif

        int total = l->n_out * bs;
        if (layer_idx == net->n_layers - 1)
        {
            // softmax
            int block_s = 256;
            int grid_s = ceil_div(bs, block_s);
            softmax_kernel<<<grid_s, block_s>>>(l->d_Z, l->d_A, l->n_out, bs);
        }
        else
        {   
            //relu
            int block1d = 256;
            int grid1d = ceil_div(total, block1d);
            relu_kernel<<<grid1d, block1d>>>(l->d_Z, l->d_A, total);
        }
    }
}

void gpu_batched_backward_pass(GPUNetwork *net,
                               const float *d_X_batch,
                               const float *d_Y_onehot,
                               int bs)
{
    for (int i = net->n_layers - 1; i >= 0; i--)
    {
        GPULayer *l = &net->layers[i];
        int total = l->n_out * bs;

        if (i == net->n_layers - 1)
        {
            // Output delta: Delta = A - Y_onehot
            int block1d = 256;
            int grid1d = ceil_div(total, block1d);
            output_delta_kernel<<<grid1d, block1d>>>(l->d_Delta, l->d_A, d_Y_onehot, total);
        }
        else
        {
            GPULayer *l_next = &net->layers[i + 1];
#ifdef USE_CUBLAS
            {
                const float one = 1.0f, zero = 0.0f;
                cublasSgemm(net->cublas_handle,
                            CUBLAS_OP_N, CUBLAS_OP_T,
                            bs, l->n_out, l_next->n_out,
                            &one,
                            l_next->d_Delta,   bs,
                            l_next->d_weights, l->n_out,
                            &zero,
                            l->d_Delta,        bs);
            }
#else
            gpu_gemm_tn(l->n_out, bs, l_next->n_out,
                        1.0f, l_next->d_weights, l->n_out,
                        l_next->d_Delta, bs,
                        0.0f, l->d_Delta, bs);
#endif
            int block1d = 256;
            int grid1d = ceil_div(total, block1d);
            relu_deriv_kernel<<<grid1d, block1d>>>(l->d_Delta, l->d_Z, total);
        }

        // Gradient: dW += Delta * A_prev^T
        const float *d_A_prev = (i == 0) ? d_X_batch : net->layers[i - 1].d_A;
#ifdef USE_CUBLAS
        {
            const float one = 1.0f;
            cublasSgemm(net->cublas_handle,
                        CUBLAS_OP_T, CUBLAS_OP_N,
                        l->n_in, l->n_out, bs,
                        &one,
                        d_A_prev,   bs,
                        l->d_Delta, bs,
                        &one,
                        l->d_dW,    l->n_in);
        }
#else
        gpu_gemm_nt(l->n_out, l->n_in, bs,
                    1.0f, l->d_Delta, bs,
                    d_A_prev, bs,
                    1.0f, l->d_dW, l->n_in);
#endif

        // Gradient: db += rowsum(Delta)
        int block_j = 256;
        int grid_j = ceil_div(l->n_out, block_j);
        rowsum_kernel<<<grid_j, block_j>>>(l->d_Delta, l->d_db, l->n_out, bs);
    }
}

void gpu_update_weights(GPUNetwork *net, float lr, int bs)
{
    float scale = lr / (float)bs;
    int block1d = 256;

    for (int i = 0; i < net->n_layers; i++)
    {
        GPULayer *l = &net->layers[i];

        int nw = l->n_out * l->n_in;
        sgd_update_kernel<<<ceil_div(nw, block1d), block1d>>>(l->d_weights, l->d_dW, scale, nw);

        int nb = l->n_out;
        sgd_update_kernel<<<ceil_div(nb, block1d), block1d>>>(l->d_biases, l->d_db, scale, nb);
    }
}
