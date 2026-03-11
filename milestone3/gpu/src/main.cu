#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include "mnist.h"
#include "gpu_network.h"

static uint32_t prng_state = 123456789;
static inline uint32_t xorshift32()
{
    uint32_t x = prng_state;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    prng_state = x;
    return x;
}

static void shuffle_indices(int *arr, int n)
{
    for (int i = n - 1; i > 0; i--)
    {
        int j = xorshift32() % (i + 1);
        int tmp = arr[i]; arr[i] = arr[j]; arr[j] = tmp;
    }
}

static void build_batch_matrix(float *X_batch, float *images, int *indices,
                               int batch_start, int bs, int n_features)
{
    for (int s = 0; s < bs; s++)
    {
        int idx = indices[batch_start + s];
        for (int p = 0; p < n_features; p++)
            X_batch[p * bs + s] = images[idx * n_features + p];
    }
}

static void build_onehot_matrix(float *Y_onehot, unsigned char *labels, int *indices,
                                int batch_start, int bs, int n_classes)
{
    memset(Y_onehot, 0, n_classes * bs * sizeof(float));
    for (int s = 0; s < bs; s++)
    {
        int label = labels[indices[batch_start + s]];
        Y_onehot[label * bs + s] = 1.0f;
    }
}

int main()
{
    FILE *loss_file = fopen("loss.csv", "w");
    fprintf(loss_file, "epoch,val_loss\n");

    MNISTData train = load_mnist("../../data/train-images-idx3-ubyte",
                                 "../../data/train-labels-idx1-ubyte");
    MNISTData test  = load_mnist("../../data/t10k-images-idx3-ubyte",
                                 "../../data/t10k-labels-idx1-ubyte");
    printf("Train: %d images  Test: %d images\n", train.n_images, test.n_images);

    int layer_sizes[] = {784, 128, 256, 10};
    int n_layers  = 3;
    int batch_size = 500;
    int epochs     = 50;
    float lr       = 0.1f;

    printf("Learning rate: %.4f  Batch size: %d  Epochs: %d\n", lr, batch_size, epochs);

    GPUNetwork net = gpu_init_network(layer_sizes, n_layers, batch_size, NULL, NULL);

    int *indices = (int *)malloc(train.n_images * sizeof(int));
    for (int i = 0; i < train.n_images; i++) indices[i] = i;

    float *h_X_batch = (float *)malloc(784 * batch_size * sizeof(float));
    float *h_Y_batch = (float *)malloc(10  * batch_size * sizeof(float));
    float *h_out     = (float *)malloc(10  * batch_size * sizeof(float));

    float *d_X_batch, *d_Y_batch;
    cudaMalloc(&d_X_batch, 784 * batch_size * sizeof(float));
    cudaMalloc(&d_Y_batch, 10  * batch_size * sizeof(float));

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    for (int epoch = 0; epoch < epochs; epoch++)
    {
        shuffle_indices(indices, 50000);
        float total_cost = 0.0f;
        int   correct    = 0;

        for (int b = 0; b < 50000; b += batch_size)
        {
            int bs = (b + batch_size <= 50000) ? batch_size : 50000 - b;

            build_batch_matrix(h_X_batch, train.images, indices, b, bs, 784);
            build_onehot_matrix(h_Y_batch, train.labels, indices, b, bs, 10);

            cudaMemcpy(d_X_batch, h_X_batch, 784 * bs * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_Y_batch, h_Y_batch, 10  * bs * sizeof(float), cudaMemcpyHostToDevice);

            gpu_zero_gradients(&net);
            gpu_batched_forward_pass(&net, d_X_batch, bs);

            GPULayer *out_layer = &net.layers[n_layers - 1];
            cudaMemcpy(h_out, out_layer->d_A, 10 * bs * sizeof(float), cudaMemcpyDeviceToHost);

            for (int s = 0; s < bs; s++)
            {
                int label = train.labels[indices[b + s]];
                total_cost -= logf(h_out[label * bs + s] + 1e-7f);
                int pred = 0;
                for (int k = 1; k < 10; k++)
                    if (h_out[k * bs + s] > h_out[pred * bs + s]) pred = k;
                if (pred == label) correct++;
            }

            gpu_batched_backward_pass(&net, d_X_batch, d_Y_batch, bs);
            gpu_update_weights(&net, lr, bs);
        }

        int   val_correct = 0;
        float val_cost    = 0.0f;
        int   val_n       = train.n_images - 50000;  // 10000
        for (int b = 50000; b < train.n_images; b += batch_size)
        {
            int bs = (b + batch_size <= train.n_images) ? batch_size : train.n_images - b;
            int *val_idx = (int *)malloc(bs * sizeof(int));
            for (int s = 0; s < bs; s++) val_idx[s] = b + s;

            build_batch_matrix(h_X_batch, train.images, val_idx, 0, bs, 784);
            cudaMemcpy(d_X_batch, h_X_batch, 784 * bs * sizeof(float), cudaMemcpyHostToDevice);

            gpu_batched_forward_pass(&net, d_X_batch, bs);

            GPULayer *out_layer = &net.layers[n_layers - 1];
            cudaMemcpy(h_out, out_layer->d_A, 10 * bs * sizeof(float), cudaMemcpyDeviceToHost);

            for (int s = 0; s < bs; s++)
            {
                int label = train.labels[b + s];
                val_cost -= logf(h_out[label * bs + s] + 1e-7f);
                int pred = 0;
                for (int k = 1; k < 10; k++)
                    if (h_out[k * bs + s] > h_out[pred * bs + s]) pred = k;
                if (pred == label) val_correct++;
            }
            free(val_idx);
        }

        float val_loss = val_cost / val_n;
        fprintf(loss_file, "%d,%.6f\n", epoch + 1, val_loss);

        printf("Epoch %2d/%d  train_cost: %.4f  train_acc: %.2f%%  val_loss: %.4f  val_acc: %.2f%%\n",
               epoch + 1, epochs,
               total_cost / 50000,
               100.0f * correct / 50000,
               val_loss,
               100.0f * val_correct / val_n);
    }

    cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double train_time = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
    double grind_rate = (double)(50000) * epochs / train_time;
    printf("\nTraining time: %.2f s  Grind rate: %.0f smp/s\n", train_time, grind_rate);

    // test set inference
    int   test_correct = 0;
    float test_cost    = 0.0f;

    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int b = 0; b < test.n_images; b += batch_size)
    {
        int bs = (b + batch_size <= test.n_images) ? batch_size : test.n_images - b;
        int *test_idx = (int *)malloc(bs * sizeof(int));
        for (int s = 0; s < bs; s++) test_idx[s] = b + s;

        build_batch_matrix(h_X_batch, test.images, test_idx, 0, bs, 784);
        cudaMemcpy(d_X_batch, h_X_batch, 784 * bs * sizeof(float), cudaMemcpyHostToDevice);

        gpu_batched_forward_pass(&net, d_X_batch, bs);

        GPULayer *out_layer = &net.layers[n_layers - 1];
        cudaMemcpy(h_out, out_layer->d_A, 10 * bs * sizeof(float), cudaMemcpyDeviceToHost);

        for (int s = 0; s < bs; s++)
        {
            int label = test.labels[b + s];
            test_cost -= logf(h_out[label * bs + s] + 1e-7f);
            int pred = 0;
            for (int k = 1; k < 10; k++)
                if (h_out[k * bs + s] > h_out[pred * bs + s]) pred = k;
            if (pred == label) test_correct++;
        }
        free(test_idx);
    }
    cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double infer_time = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;

    printf("Test accuracy:  %.2f%%  Test cost: %.4f\n",
           100.0f * test_correct / test.n_images,
           test_cost / test.n_images);
    printf("Inference time: %.4f s\n", infer_time);

    fclose(loss_file);

    cudaFree(d_X_batch);
    cudaFree(d_Y_batch);
    free(h_X_batch);
    free(h_Y_batch);
    free(h_out);
    free(indices);
    gpu_free_network(&net);
    free_mnist(&train);
    free_mnist(&test);

    return 0;
}
