#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "mnist.h"
#include "network.h"

int main()
{
    MNISTData train = load_mnist("../data/train-images-idx3-ubyte", "../data/train-labels-idx1-ubyte");
    MNISTData test = load_mnist("../data/t10k-images-idx3-ubyte", "../data/t10k-labels-idx1-ubyte");
    printf("Train: %d images, Test: %d images\n", train.n_images, test.n_images);

    int layer_sizes[] = {784, 128, 256, 10};
    Network neural_net = init_network(layer_sizes, 3);

    // params
    float learning_rate = 0.05;
    int batch_size = 64;
    int epochs = 20;
    int pixels = 784;

    int *indices = malloc(train.n_images * sizeof(int));
    for (int i = 0; i < train.n_images; i++)
        indices[i] = i;

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    for (int epoch = 0; epoch < epochs; epoch++)
    {
        shuffle_indices(indices, train.n_images);
        float total_cost = 0.0f;
        int correct = 0;

        for (int batch_start = 0; batch_start < train.n_images; batch_start += batch_size)
        {
            int batch_end = batch_start + batch_size < train.n_images ? batch_start + batch_size : train.n_images;
            zero_gradients(&neural_net);

            for (int sample = batch_start; sample < batch_end; sample++)
            {
                int idx = indices[sample];
                forward_pass(&neural_net, &train.images[idx * pixels]);
                backward_pass(&neural_net, train.labels[idx]);
                float *out = neural_net.layers[neural_net.n_layers - 1].a;
                total_cost -= logf(out[train.labels[idx]] + 1e-7f);
                int pred = 0;
                for (int k = 1; k < 10; k++)
                    if (out[k] > out[pred])
                        pred = k;
                if (pred == train.labels[idx])
                    correct++;
            }
            update_weights(&neural_net, learning_rate, batch_end - batch_start);
        }
        printf("Epoch %d/%d   cost: %.4f  accr: %.2f%%\n",
               epoch + 1, epochs, total_cost / train.n_images, 100.0f * correct / train.n_images);
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
    printf("Training time: %.2fs\n", elapsed);
    double grind_rate = (epochs * train.n_images) / elapsed;
    printf("Grind rate: %.0f samples/sec\n", grind_rate);


    // test accuracy 
    int test_correct = 0;
    float test_cost = 0.0f;

    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int i = 0; i < test.n_images; i++)
    {
        forward_pass(&neural_net, &test.images[i * pixels]);
        float *out = neural_net.layers[neural_net.n_layers - 1].a;
        int pred = 0;
        test_cost -= logf(out[test.labels[i]] + 1e-7f);
        for (int k = 1; k < 10; k++)
            if (out[k] > out[pred])
                pred = k;
        if (pred == (int)test.labels[i])
            test_correct++;
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    printf("Inference time: %.4fs\n", (t1.tv_sec-t0.tv_sec) + (t1.tv_nsec-t0.tv_nsec)/1e9);
    printf("Test accuracy: %.2f%%\n", 100.0f * test_correct / test.n_images);
    printf("Test Cost: %.4f\n",  test_cost / test.n_images);
    free(indices);

    free_network(&neural_net);

    free_mnist(&train);
    free_mnist(&test);
}