import matplotlib.pyplot as plt
import gzip
import numpy as np
import os
from math import sqrt, exp

data_sources = {
    "training_images": "train-images-idx3-ubyte.gz",  # 60,000 training images.
    "test_images": "t10k-images-idx3-ubyte.gz",  # 10,000 test images.
    "training_labels": "train-labels-idx1-ubyte.gz",  # 60,000 training labels.
    "test_labels": "t10k-labels-idx1-ubyte.gz",  # 10,000 test labels.
}

mnist_dataset = {}
data_dir = "data/"

# Images
for key in ("training_images", "test_images"):
    with gzip.open(os.path.join(data_dir, data_sources[key]), "rb") as mnist_file:
        mnist_dataset[key] = np.frombuffer(
            mnist_file.read(), np.uint8, offset=16
        ).reshape(-1, 28 * 28)
# Labels
for key in ("training_labels", "test_labels"):
    with gzip.open(os.path.join(data_dir, data_sources[key]), "rb") as mnist_file:
        mnist_dataset[key] = np.frombuffer(mnist_file.read(), np.uint8, offset=8)

# x images , y labels
x_train, y_train, x_test, y_test = (
    mnist_dataset["training_images"],
    mnist_dataset["training_labels"],
    mnist_dataset["test_images"],
    mnist_dataset["test_labels"],
)

print(
    "The shape of training images: {} and training labels: {}".format(
        x_train.shape, y_train.shape
    )
)
print(
    "The shape of test images: {} and test labels: {}".format(
        x_test.shape, y_test.shape
    )
)

print(x_train.dtype)

# prep data , normaloise
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# 1 hotencoding labels
y_train_encoded = np.eye(10)[y_train]
y_test_encoded = np.eye(10)[y_test]


# print("The data type of training images: {}".format(x_train.dtype))
# print("sample: {}".format(y_test_encoded[0]))


###### Weight Init ## ####
seed_value = 12345
rng = np.random.default_rng(seed=seed_value)


def init_layer(input_layer: int, output_layer: int) -> float:

    limit = 1 / sqrt(input_layer)

    weight_init = rng.uniform(-limit, limit, size=(output_layer, input_layer))
    bias_init = rng.uniform(-limit, limit, size=(output_layer))

    return weight_init, bias_init


w1, b1 = init_layer(784, 128)
w2, b2 = init_layer(128, 256)
w3, b3 = init_layer(256, 10)

W = (w1, w2, w3)
B = (b1, b2, b3)

print(w1.shape, b2.shape)


# relu
def relu(val):
    return np.maximum(0, val)


# forward pass
def forward_pass(input_layer, W, B):
    # activation is
    # w.a_prev + b for layer l
    w1, w2, w3 = W
    b1, b2, b3 = B

    # layer 1
    z1 = np.dot(w1, input_layer) + b1
    a1 = relu(z1)

    # layer 2
    z2 = np.dot(w2, a1) + b2
    a2 = relu(z2)

    # layer 3
    # apply softmax
    z3 = np.dot(w3, a2) + b3
    z3 = z3 - max(z3)
    exp_z = np.exp(z3)
    # softmax
    a3 = exp_z / sum(exp_z)

    return (z1, z2, z3, a1, a2, a3)


def backpropogate(input_l, encoded_label, z1, z2, z3, a1, a2, a3):
    # outer layer
    d3 = a3 - encoded_label

    dw3 = np.outer(d3, a2)
    db3 = d3

    # hidden layer n-1
    relu_derv_z2 = z2 > 0
    d2 = np.dot(np.transpose(w3), d3) * relu_derv_z2

    dw2 = np.outer(d2, a1)
    db2 = d2

    # hidden layer n-2
    relu_derv_z1 = z1 > 0
    d1 = np.dot(np.transpose(w2), d2) * relu_derv_z1

    dw1 = np.outer(d1, input_l)
    db1 = d1

    return (dw1, db1, dw2, db2, dw3, db3)


store = []
epsilon = 1e-7

learning_rate = 0.01
batch_size = 64
epochs = 20

n_samples = x_train.shape[0]
indices = np.arange(n_samples)

for epoch in range(epochs):
    # shuffle dataset
    np.random.shuffle(indices)
    x_train_shuffled = x_train[indices]
    y_train_encoded_shuffled = y_train_encoded[indices]
    y_train_shuffled = y_train[indices]

    total_cost = 0

    # Loop over mini-batches
    for batch_start in range(0, n_samples, batch_size):
        batch_x = x_train_shuffled[batch_start : batch_start + batch_size]
        batch_y_encoded = y_train_encoded_shuffled[
            batch_start : batch_start + batch_size
        ]
        batch_y = y_train_shuffled[batch_start : batch_start + batch_size]

        # Zero out gradient accumulators (shaped to match each weight/bias)
        dw1_acc = np.zeros_like(w1)
        dw2_acc = np.zeros_like(w2)
        dw3_acc = np.zeros_like(w3)
        db1_acc = np.zeros_like(b1)
        db2_acc = np.zeros_like(b2)
        db3_acc = np.zeros_like(b3)

        for index, image_vec in enumerate(batch_x):
            # forward pass
            z1, z2, z3, a1, a2, a3 = forward_pass(image_vec, W, B)
            # calculate cost
            C = -np.log(a3[batch_y[index]] + epsilon)
            total_cost += C

            # backpropogate
            dw1, db1, dw2, db2, dw3, db3 = backpropogate(
                image_vec, batch_y_encoded[index], z1, z2, z3, a1, a2, a3
            )

            # accumulate gradients
            dw1_acc += dw1
            dw2_acc += dw2
            dw3_acc += dw3

            db1_acc += db1
            db2_acc += db2
            db3_acc += db3

        # update for batch_x
        n_batch_x = len(batch_x)
        w1 -= (learning_rate / n_batch_x) * dw1_acc
        b1 -= (learning_rate / n_batch_x) * db1_acc
        w2 -= (learning_rate / n_batch_x) * dw2_acc
        b2 -= (learning_rate / n_batch_x) * db2_acc
        w3 -= (learning_rate / n_batch_x) * dw3_acc
        b3 -= (learning_rate / n_batch_x) * db3_acc

        # Refresh tuples so forward_pass and backpropogate see updated weights
        W = (w1, w2, w3)
        B = (b1, b2, b3)

    # End of epoch — evaluate
    avg_cost = total_cost / n_samples

    correct = sum(
        np.argmax(forward_pass(input_img, W, B)[5]) == y_test[i]
        for i, input_img in enumerate(x_test)
    )
    test_accuracy = correct / len(x_test)

    print(
        f"Epoch {epoch + 1}/{epochs}  cost: {avg_cost:.4f}  test accuracy: {test_accuracy:.4f}"
    )
