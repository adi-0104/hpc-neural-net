#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include "network.h"

// ----- RAndom Generator --------
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
    // 0 to 1 generator
    return (float)xorshift32() / (float)UINT32_MAX;
}


void shuffle_indices(int *arr, int n){
    for(int i = n-1; i > 0; i--){
        int j = xorshift32() % (i + 1);
        int tmp = arr[i];
        arr[i] = arr[j];
        arr[j] = tmp;
    }
}

// ---- MATRIX MULTIPLIERS -------

// z = W * x + b
// W (rows * Cols)
void mat_vec_mul(float *W, float *x, float *b, float *z, int rows, int cols)
{
    for (int j = 0; j < rows; j++)
    {
        z[j] = b[j];
        for (int k = 0; k < cols; k++)
        {
            z[j] += W[j * cols + k] * x[k];
        }
    }
}

// result = W^T * x  (transpose multiply for backprop)
void mat_transpose_vec_mul(float *W, float *x, float *result, int rows, int cols)
{
    // NOTE:  make sure result is zeroes
    for (int j = 0; j < rows; j++)
    {
        for (int k = 0; k < cols; k++)
        {
            result[k] += W[j * cols + k] * x[j];
        }
    }
}

// dw += outer(delta, a_prev) (accumulate calculated products)
// delta is (n_out), a_prev is (n_in), dw is (n_out x n_in)
void outer_product_accumulate(float *delta, float *a_prev, float *dw, int n_out, int n_in)
{
    for (int j = 0; j < n_out; j++)
    {
        for (int k = 0; k < n_in; k++)
        {
            dw[j * n_in + k] += delta[j] * a_prev[k];
        }
    }
}

// ------ NEURAL NET CORE ------
Layer init_layer(int n_in, int n_out)
{
    // n_in, and n_out determine the dimensions of weight and bias
    Layer l;
    l.n_in = n_in;
    l.n_out = n_out;
    l.weights = malloc(n_out * n_in * sizeof(float));
    l.biases = malloc(n_out * sizeof(float));
    l.z = calloc(n_out, sizeof(float));
    l.a = calloc(n_out, sizeof(float));
    l.delta = calloc(n_out, sizeof(float));
    l.dw = calloc(n_out * n_in, sizeof(float));
    l.db = calloc(n_out, sizeof(float));

    // init weight and bias
    float limit = 1 / sqrtf(n_in);
    float range = 2.0f * limit;

    for (int i = 0; i < n_out * n_in; i++)
    {
        l.weights[i] = (rand_uniform() * range) - limit;
    }

    for (int i = 0; i < n_out; i++)
    {
        l.biases[i] = (rand_uniform() * range) - limit;
    }

    return l;
}
// Activations
void relu(float *z, float *a, int n)
{
    // max(0,z)
    for (int i = 0; i < n; i++)
    {
        a[i] = z[i] > 0 ? z[i] : 0.0f;
    }
}
void softmax(float *z, float *a, int n)
{
    // z3 = z3 - max(z3)
    float z_max = -INFINITY;
    for (int i = 0; i < n; i++)
    {
        z_max = z[i] > z_max ? z[i] : z_max;
    }

    // exp_z = np.exp(z3)
    float sum_exp_z = 0;
    for (int i = 0; i < n; i++)
    {
        a[i] = expf(z[i] - z_max);
        sum_exp_z += a[i];
    }

    // # softmax
    for (int i = 0; i < n; i++)
    {
        // a3 = exp_z / sum(exp_z)
        a[i] /= sum_exp_z;
    }
}

Network init_network(int *layer_sizes, int n_layers)
{
    // layers - [784, 128, 256, 10], nlayers 3
    // initialose network
    Network network;
    network.n_layers = n_layers;

    Layer *layers = malloc(n_layers * sizeof(Layer));

    for (int i = 1; i <= n_layers; i++)
    {
        Layer l = init_layer(layer_sizes[i - 1], layer_sizes[i]);
        layers[i - 1] = l;
    }

    network.layers = layers;
    network.input = NULL;
    return network;
}

void forward_pass(Network *net, float *input)
{
    net->input = input;
    float *x = input;
    // loop over each layer in the network
    for (int i = 0; i < net->n_layers; i++)
    {
        Layer *current_layer = &net->layers[i];
        // calculate z
        mat_vec_mul(current_layer->weights, x, current_layer->biases, current_layer->z, current_layer->n_out, current_layer->n_in);
        // apply activation function - relu or softmax
        if (i == net->n_layers - 1)
        {
            // apply softmax
            softmax(current_layer->z, current_layer->a, current_layer->n_out);
        }
        else
        {
            relu(current_layer->z, current_layer->a, current_layer->n_out);
        }

        // update input
        x = current_layer->a;
    }
}

void backward_pass(Network *net, int label)
{

    for (int i = net->n_layers - 1; i >= 0; i--)
    {
        Layer *l = &net->layers[i];
        // calculate delta based on layer position
        if (i == net->n_layers - 1)
        {
            for (int j = 0; j < l->n_out; j++)
            {
                l->delta[j] = l->a[j] - (j == label ? 1.0f : 0.0f);
            }
        }
        else
        {
            // relu derivative
            // d2 = np.dot(W3.T, d3) * (z2>0)
            Layer *l_plus_1 = &net->layers[i + 1];
            memset(l->delta, 0, l->n_out * sizeof(float));
            mat_transpose_vec_mul(l_plus_1->weights, l_plus_1->delta, l->delta, l_plus_1->n_out, l_plus_1->n_in);
            for (int j = 0; j < l->n_out; j++)
            {

                l->delta[j] *= (l->z[j] > 0 ? 1.0f : 0.0f);
            }
        }

        // calculate dw
        if (i != 0)
        {
            Layer *l_prev = &net->layers[i - 1];
            // dw3 = np.outer(d3, a2)
            outer_product_accumulate(l->delta, l_prev->a, l->dw, l->n_out, l->n_in);
        }
        else
        {
            outer_product_accumulate(l->delta, net->input, l->dw, l->n_out, l->n_in);
        }
        // calculate db
        for (int j = 0; j < l->n_out; j++)
        {
            l->db[j] += l->delta[j];
        }
    }
}

void zero_gradients(Network *net)
{
    for (int l = 0; l < net->n_layers; l++)
    {
        // init dw and db gradients
        memset(net->layers[l].dw, 0, net->layers[l].n_in * net->layers[l].n_out * sizeof(float));
        memset(net->layers[l].db, 0, net->layers[l].n_out * sizeof(float));
    }
}
void update_weights(Network *net, float lr, int batch_size)
{
    float scale = lr / batch_size;
    for (int l = 0; l < net->n_layers; l++)
    {
        Layer *curr_l = &net->layers[l];

        // update weights
        for (int j = 0; j < curr_l->n_out * curr_l->n_in; j++)
        {
            curr_l->weights[j] -= scale * curr_l->dw[j];
        }
        // update bias
        for (int j = 0; j < curr_l->n_out; j++)
        {
            curr_l->biases[j] -= scale * curr_l->db[j];
        }
    }
}

void free_network(Network *n)
{
    for (int i = 0; i < n->n_layers; i++)
    {
        free_layer(&n->layers[i]);
    }
    free(n->layers);
}

void free_layer(Layer *l)
{
    free(l->weights);
    free(l->biases);
    free(l->z);
    free(l->a);
    free(l->delta);
    free(l->dw);
    free(l->db);
}
