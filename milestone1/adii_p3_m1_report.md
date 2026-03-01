# Project 3, Milestone 1
**Serial Neural Network — MNIST Digit Classification**

**cnet: adii**


## Architecture
- **Network:** 784 (input) → 128 (ReLU) → 256 (ReLU) → 10 (Softmax)
- **Loss:** Cross-entropy
- **Weight/Bias Init:** Kaiming uniform — Uniform(-1/sqrt(n_in), 1/sqrt(n_in))
- **Optimizer:** Mini-batch SGD

## Results

| Metric               | lr = 0.01   | lr = 0.05   |
|----------------------|-------------|-------------|
| Success rate         | 95.73%      | 97.69%      |
| Grind rate           | 4703 smp/s  | 4709 smp/s  |
| Total training time  | 265.26 s    | 254.82 s    |
| Total inference time | 1.76 s           | 1.74 s      |
| Learning rate        | 0.01        | 0.05        |
| Batch size           | 64          | 100          |
