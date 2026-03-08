#pragma once
#include <stdint.h>

typedef struct
{
    int n_in, n_out;
    int max_batch;
    float *weights;
    float *biases;
    float *Z;
    float *A;
    float *Delta;
    float *dW;
    float *db;
} Layer;

// Network
typedef struct
{
    int n_layers;
    Layer *layers;
    int max_batch;
} Network;

// Layer lifecycle
Layer init_layer(int n_in, int n_out, int max_batch);
void free_layer(Layer *l);

// Network lifecycle
Network init_network(int *layer_sizes, int n_layers, int max_batch);
void free_network(Network *n);

// Batch data helpers
void build_batch_matrix(float *X_batch, float *images, int *indices,
                        int batch_start, int bs, int n_features);

void build_onehot_matrix(float *Y_onehot, unsigned char *labels, int *indices,
                         int batch_start, int bs, int n_classes);

// Activations
void relu_batch(float *Z, float *A, int total);
void softmax_batch(float *Z, float *A, int n_classes, int bs);

void batched_forward_pass(Network *net, float *X_batch, int bs);
void batched_backward_pass(Network *net, float *X_batch, float *Y_onehot, int bs);

// SGD helpers
void zero_gradients(Network *net);
void update_weights(Network *net, float lr, int bs);
void shuffle_indices(int *arr, int n);
