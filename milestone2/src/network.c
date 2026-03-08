#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include "network.h"
#include "gemm.h"

// ----- RAndom Generator --------
static uint32_t prng_state = 123456789;
static inline uint32_t xorshift32() {
    uint32_t x = prng_state;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    prng_state = x;
    return x;
}
static inline float rand_uniform() {
    return (float)xorshift32() / (float)UINT32_MAX;
}

void shuffle_indices(int *arr, int n) {
    for (int i = n-1; i > 0; i--) {
        int j = xorshift32() % (i + 1);
        int tmp = arr[i]; arr[i] = arr[j]; arr[j] = tmp;
    }
}

// ---- Layer lifecycle ----
Layer init_layer(int n_in, int n_out, int max_batch)
{
    Layer l;
    l.n_in = n_in; l.n_out = n_out; l.max_batch = max_batch;
    l.weights = malloc(n_out * n_in  * sizeof(float));
    l.biases  = malloc(n_out         * sizeof(float));
    l.Z       = calloc(n_out * max_batch, sizeof(float));
    l.A       = calloc(n_out * max_batch, sizeof(float));
    l.Delta   = calloc(n_out * max_batch, sizeof(float));
    l.dW      = calloc(n_out * n_in,  sizeof(float));
    l.db      = calloc(n_out,          sizeof(float));

    // Kaiming uniform: same as M1
    float limit = 1.0f / sqrtf((float)n_in);
    float range = 2.0f * limit;
    for (int i = 0; i < n_out * n_in; i++)
        l.weights[i] = (rand_uniform() * range) - limit;
    for (int i = 0; i < n_out; i++)
        l.biases[i] = (rand_uniform() * range) - limit;

    return l;
}

void free_layer(Layer *l) {
    free(l->weights); free(l->biases);
    free(l->Z); free(l->A); free(l->Delta);
    free(l->dW); free(l->db);
}

Network init_network(int *layer_sizes, int n_layers, int max_batch)
{
    Network net;
    net.n_layers  = n_layers;
    net.max_batch = max_batch;
    net.layers    = malloc(n_layers * sizeof(Layer));
    for (int i = 0; i < n_layers; i++)
        net.layers[i] = init_layer(layer_sizes[i], layer_sizes[i+1], max_batch);
    return net;
}

void free_network(Network *n) {
    for (int i = 0; i < n->n_layers; i++) free_layer(&n->layers[i]);
    free(n->layers);
}

//Batch data helpers

void build_batch_matrix(float *X_batch, float *images, int *indices,
                        int batch_start, int batch_size, int img_pixels)
{
    for(int s = 0; s < batch_size; s++){
        int idx = indices[batch_start + s];
        for(int p = 0; p < img_pixels; p++){
            X_batch[p * batch_size + s] = images[idx * img_pixels + p];
        }
    }

}

void build_onehot_matrix(float *Y_onehot, unsigned char *labels, int *indices,
                         int batch_start, int batch_size, int n_classes)
{
    memset(Y_onehot, 0, n_classes * batch_size * sizeof(float));
    
    for(int s = 0; s < batch_size; s++){
        int label = labels[indices[batch_start + s]];
        Y_onehot[label * batch_size + s] = 1.0f;
    }

}

// Activations ----

void relu_batch(float *Z, float *A, int total)
{
    for (int i = 0; i < total; i++)
    {
        A[i] = Z[i] > 0 ? Z[i] : 0.0f;
    }
}

void softmax_batch(float *Z, float *A, int n_classes, int batch_size)
{
    // for each sample find the z and a vals
    #pragma omp parallel for schedule(static)
    for(int s =0; s< batch_size; s++){
        // find max
        float z_max = -INFINITY;
        for (int j = 0; j < n_classes; j++)
        {
            z_max = Z[j*batch_size + s] > z_max ? Z[j*batch_size + s] : z_max;
        }

        // exp of z
        float sum_exp_z = 0;
        for (int i = 0; i < n_classes; i++)
        {
            A[i*batch_size + s] = expf(Z[i*batch_size + s] - z_max);
            sum_exp_z += A[i*batch_size + s];
        }

        for (int i = 0; i < n_classes; i++)
        {
            // a3 = exp_z / sum(exp_z)
            A[i*batch_size + s] /= sum_exp_z;
        }

    }
}

// forward pass

void batched_forward_pass(Network *net, float *X_batch, int batch_size)
{
    
    // loop over each layer in the network
    for (int l = 0; l < net->n_layers; l++)
    {
        Layer *current_layer = &net->layers[l];
        
        float *a_prev = (l == 0) ? X_batch : net->layers[l-1].A;
        // calculate z
        for(int j = 0; j < current_layer->n_out; j++){
            for (int s = 0; s < batch_size; s++){
                current_layer->Z[j*batch_size + s] = current_layer->biases[j];
            }
        }
        gemm_nn(current_layer->n_out, batch_size, current_layer->n_in,1.0f,current_layer->weights,current_layer->n_in, a_prev, batch_size, 1.0f, current_layer->Z, batch_size );
        
        if (l == net->n_layers - 1)
        {
            // apply softmax
            softmax_batch(current_layer->Z, current_layer->A, current_layer->n_out, batch_size);
        }
        else
        {
            relu_batch(current_layer->Z, current_layer->A, current_layer->n_out * batch_size);
        }
    }
}

// backward pass

void batched_backward_pass(Network *net, float *X_batch, float *Y_onehot, int batch_size)
{
    for (int i = net->n_layers - 1; i >= 0; i--){
        Layer *l = &net->layers[i];

        if (i == net->n_layers - 1) {
            for (int j = 0; j < l->n_out; j++)
            {
                for(int s = 0; s < batch_size; s++){

                    l->Delta[j*batch_size + s] = l->A[j*batch_size + s] - Y_onehot[j*batch_size + s];
                }
            }
        } else {
            Layer *l_plus_1 = &net->layers[i + 1];
            gemm_tn(l->n_out,batch_size,l_plus_1->n_out,1.0f,l_plus_1->weights,l->n_out, l_plus_1->Delta,batch_size, 0.0f, l->Delta, batch_size);
            for (int j = 0; j < l->n_out; j++)
            {
                for(int s = 0; s < batch_size; s++){

                    l->Delta[j*batch_size + s] *= (l->Z[j*batch_size + s] > 0 ? 1.0f : 0.0f);
                }
            }
        }

        // dw and db
        float *A_prev = (i == 0) ? X_batch : net->layers[i - 1].A;
        gemm_nt(l->n_out,l->n_in,batch_size,1.0f, l->Delta,batch_size,A_prev, batch_size, 1.0f, l->dW, l->n_in);

    //     for (j in 0..n_out):
    // db[j] += sum over s of Delta[j*bs + s]

        for (int j = 0; j < l->n_out; j++)
        {
            float delta_sum = 0.0f;
            for(int s = 0; s < batch_size; s++){
                delta_sum += l->Delta[j*batch_size + s];

            }
            l->db[j] += delta_sum;
        }

    }
}

//SGD inits 

void zero_gradients(Network *net)
{
    for (int l = 0; l < net->n_layers; l++) {
        memset(net->layers[l].dW, 0, net->layers[l].n_out * net->layers[l].n_in * sizeof(float));
        memset(net->layers[l].db, 0, net->layers[l].n_out * sizeof(float));
    }
}

void update_weights(Network *net, float lr, int bs)
{
    // Same structure as M1 — scale = lr / bs
    float scale = lr / (float)bs;
    for (int l = 0; l < net->n_layers; l++) {
        Layer *lay = &net->layers[l];
        for (int i = 0; i < lay->n_out * lay->n_in; i++)
            lay->weights[i] -= scale * lay->dW[i];
        for (int i = 0; i < lay->n_out; i++)
            lay->biases[i]  -= scale * lay->db[i];
    }
}
