#pragma once

typedef struct
{
    int n_in;
    int n_out;
    float *weights;
    float *biases;
    // activators
    float *z;
    float *a;
    // error/derivitive
    float *delta;
    // accumulators
    float *dw;
    float *db;
} Layer;

// Neural Net
typedef struct
{
    int n_layers;  // number of layers
    Layer *layers; // array of layers
    float *input;  // a0 pointer
} Network;

// layer structs
Layer init_layer(int n_in, int n_out);
void free_layer(Layer *l);

// multiplication loops
void mat_vec_mul(float *W, float *x, float *b, float *z, int rows, int cols);
void mat_transpose_vec_mul(float *W, float *x, float *result, int rows, int cols);
void outer_product_accumulate(float *delta, float *a_prev, float *dw, int n_out, int n_in);

// activations
void relu(float *z, float *a, int n);
void softmax(float *z, float *a, int n);

// Network base struct
Network init_network(int *layer_sizes, int n_layers);
void free_network(Network *n);

// network core
void forward_pass(Network *net, float *input);
void backward_pass(Network *net, int label);

// helpers
void zero_gradients(Network *net);
void update_weights(Network *net, float lr, int batch_size);
