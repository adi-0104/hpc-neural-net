#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "mnist.h"
#include "network.h"
#include <omp.h>

int main()
{
    FILE *loss_file = fopen("loss.csv", "w");
    fprintf(loss_file, "epoch,val_loss\n");

    // load data
    MNISTData train = load_mnist("../data/train-images-idx3-ubyte",
                                 "../data/train-labels-idx1-ubyte");
    MNISTData test = load_mnist("../data/t10k-images-idx3-ubyte",
                                "../data/t10k-labels-idx1-ubyte");
    printf("Train: %d images  Test: %d images\n", train.n_images, test.n_images);

    // params
    int layer_sizes[] = {784, 128, 256, 10};
    int n_layers = 3;
    int batch_size = 500;
    int epochs = 50;
    float lr = 0.1f;

    Network net = init_network(layer_sizes, n_layers, batch_size);

    // index array to shuffle
    int *indices = malloc(train.n_images * sizeof(int));
    for (int i = 0; i < train.n_images; i++)
        indices[i] = i;

    // allocate batch input and output buffers
    float *X_batch = malloc(784 * batch_size * sizeof(float));
    float *Y_batch = malloc(10 * batch_size * sizeof(float));

    printf("Learning rate: %.4f  Batch size: %d  Epochs: %d\n", lr, batch_size, epochs);
    printf("OpenMP threads: %d\n", omp_get_max_threads()); 

    // train
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    for (int epoch = 0; epoch < epochs; epoch++)
    {
        shuffle_indices(indices, 50000);
        float total_cost = 0.0f;
        int correct = 0;

        for (int b = 0; b < 50000; b += batch_size)
        {

            int bs = (b + batch_size <= 50000) ? batch_size : 50000 - b;

            build_batch_matrix(X_batch, train.images, indices, b, bs, 784);
            build_onehot_matrix(Y_batch, train.labels, indices, b, bs, 10);

            zero_gradients(&net);
            batched_forward_pass(&net, X_batch, bs);

            float *out = net.layers[n_layers - 1].A;
            for (int s = 0; s < bs; s++)
            {
                int label = train.labels[indices[b + s]];
                total_cost -= logf(out[label * bs + s] + 1e-7f);
                int pred = 0;
                for (int k = 1; k < 10; k++)
                    if (out[k * bs + s] > out[pred * bs + s])
                        pred = k;
                if (pred == label)
                    correct++;
            }

            batched_backward_pass(&net, X_batch, Y_batch, bs);
            update_weights(&net, lr, bs);
        }

        int val_correct = 0;
        float val_cost = 0.0f;
        int val_n = train.n_images - 50000;  // 10000
        for (int b = 50000; b < train.n_images; b += batch_size)
        {
            int bs = (b + batch_size <= train.n_images) ? batch_size : train.n_images - b;
            int *val_idx = malloc(bs * sizeof(int));
            for (int s = 0; s < bs; s++)
                val_idx[s] = b + s;

            build_batch_matrix(X_batch, train.images, val_idx, 0, bs, 784);
            batched_forward_pass(&net, X_batch, bs);

            float *out = net.layers[n_layers - 1].A;
            for (int s = 0; s < bs; s++)
            {
                int label = train.labels[b + s];
                val_cost -= logf(out[label * bs + s] + 1e-7f);
                int pred = 0;
                for (int k = 1; k < 10; k++)
                    if (out[k * bs + s] > out[pred * bs + s])
                        pred = k;
                if (pred == label)
                    val_correct++;
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

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double train_time = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
    double grind_rate = (double)(50000) * epochs / train_time;
    printf("\nTraining time: %.2f s  Grind rate: %.0f smp/s\n", train_time, grind_rate);

    // testing loop
    int test_correct = 0;
    float test_cost = 0.0f;

    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int b = 0; b < test.n_images; b += batch_size)
    {
        int bs = (b + batch_size <= test.n_images) ? batch_size : test.n_images - b;
        int *test_idx = malloc(bs * sizeof(int));
        for (int s = 0; s < bs; s++)
            test_idx[s] = b + s;

        build_batch_matrix(X_batch, test.images, test_idx, 0, bs, 784);
        batched_forward_pass(&net, X_batch, bs);

        float *out = net.layers[n_layers - 1].A;
        for (int s = 0; s < bs; s++)
        {
            int label = test.labels[b + s];
            test_cost -= logf(out[label * bs + s] + 1e-7f);
            int pred = 0;
            for (int k = 1; k < 10; k++)
                if (out[k * bs + s] > out[pred * bs + s])
                    pred = k;
            if (pred == label)
                test_correct++;
        }
        free(test_idx);
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double infer_time = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;

    printf("Test accuracy:  %.2f%%  Test cost: %.4f\n",
           100.0f * test_correct / test.n_images,
           test_cost / test.n_images);
    printf("Inference time: %.4f s\n", infer_time);

    fclose(loss_file);

    // clean
    free(X_batch);
    free(Y_batch);
    free(indices);
    free_network(&net);
    free_mnist(&train);
    free_mnist(&test);
    return 0;
}
