#include <stdio.h>
#include "mnist.h"
#include "network.h" 

int main() {
    MNISTData train = load_mnist("data/train-images-idx3-ubyte", "data/train-labels-idx1-ubyte");
    MNISTData test  = load_mnist("data/t10k-images-idx3-ubyte", "data/t10k-labels-idx1-ubyte");
    printf("Train: %d images, Test: %d images\n", train.n_images, test.n_images);
    
    int layer_sizes[] = {784,256,128,10};
    Network neural_net = init_network(layer_sizes, 3);
    
    // forward pass
    forward_pass(&neural_net, &train.images[0]);

    // check softmax output
    float *output = neural_net.layers[2].a;
    float sum = 0.0f;
    printf("Softmax output:\n");
    for(int i = 0; i < 10; i++){
        printf("  class %d: %f\n", i, output[i]);
        sum += output[i];
    }
    printf("Sum: %f (should be ~1.0)\n", sum);

    // check predicted vs actual
    int pred = 0;
    for(int i = 1; i < 10; i++){
        if(output[i] > output[pred]) pred = i;
        
    }
    printf("Predicted: %d, Actual: %d\n", pred, train.labels[0]);

    free_network(&neural_net);

    free_mnist(&train);
    free_mnist(&test);
}