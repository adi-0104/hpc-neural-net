#pragma once
#include <stdint.h>

typedef struct
{
    int n_in, n_out;
    int max_batch;

    // Parameters (device)
    float *d_weights;
    float *d_biases;

    // Activations
    float *d_Z;
    float *d_A;

    // Backprop buffers (device)
    float *d_Delta;
    float *d_dW;
    float *d_db;
} GPULayer;

typedef struct
{
    int n_layers;
    GPULayer *layers;
    int max_batch;
} GPUNetwork;

GPULayer gpu_init_layer(int n_in, int n_out, int max_batch,
                        const float *h_weights, const float *h_biases);
void gpu_free_layer(GPULayer *l);

GPUNetwork gpu_init_network(int *layer_sizes, int n_layers, int max_batch,
                            const float *h_weights_flat[],
                            const float *h_biases_flat[]);
void gpu_free_network(GPUNetwork *net);

void gpu_zero_gradients(GPUNetwork *net);

void gpu_batched_forward_pass(GPUNetwork *net, const float *d_X_batch, int bs);

void gpu_batched_backward_pass(GPUNetwork *net,
                               const float *d_X_batch,
                               const float *d_Y_onehot,
                               int bs);

void gpu_update_weights(GPUNetwork *net, float lr, int bs);

__global__ void relu_kernel(const float *Z, float *A, int total);

__global__ void relu_deriv_kernel(float *Delta, const float *Z, int total);

__global__ void softmax_kernel(const float *Z, float *A, int n_classes, int bs);

__global__ void bias_broadcast_kernel(float *Z, const float *biases, int n_out, int bs);

__global__ void output_delta_kernel(float *Delta, const float *A, const float *Y_onehot, int total);

__global__ void rowsum_kernel(const float *Delta, float *db, int n_out, int bs);

__global__ void sgd_update_kernel(float *param, const float *grad, float scale, int n);
