import csv
import sys
import matplotlib.pyplot as plt

# Usage:
#   python plot_loss.py cpu_native.csv cpu_blas.csv gpu_native.csv gpu_cublas.csv
#
# Each CSV has a header: epoch,val_loss
# Written by the training program once per epoch

VERSIONS = ["CPU Native", "CPU BLAS", "GPU Native", "GPU cuBLAS"]

def parse_csv(filepath):
    epochs, losses = [], []
    with open(filepath) as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            losses.append(float(row["val_loss"]))
    return epochs, losses

def main():
    if len(sys.argv) != 5:
        print("Usage: python plot_loss.py cpu_native.csv cpu_blas.csv gpu_native.csv gpu_cublas.csv")
        sys.exit(1)

    files = sys.argv[1:]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes = axes.flatten()

    for i, (fpath, name) in enumerate(zip(files, VERSIONS)):
        epochs, losses = parse_csv(fpath)
        if not losses:
            print(f"Warning: no data in {fpath}")
            continue
        axes[i].plot(epochs, losses, linewidth=1.5)
        axes[i].set_title(name)
        axes[i].set_xlabel("Epoch")
        axes[i].set_ylabel("Validation Loss")
        axes[i].grid(True, alpha=0.3)

    fig.suptitle("Validation Loss Curves — MNIST (α=0.1, batch=500, 50 epochs)", fontsize=12)
    plt.tight_layout()
    plt.savefig("loss_curves.png", dpi=150)
    print("Saved loss_curves.png")

if __name__ == "__main__":
    main()
