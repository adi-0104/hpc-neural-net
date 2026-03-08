#!/bin/bash
#SBATCH --job-name=nn_cpu_bench
#SBATCH --output=bench_cpu_%j.out
#SBATCH --error=bench_cpu_%j.err
#SBATCH --partition=caslake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --mem=8G

module load gcc
module load openblas

cd $SLURM_SUBMIT_DIR

echo "========================================"
echo "  CPU Benchmark — $(date)"
echo "  Node: $SLURMD_NODENAME"
echo "========================================"

# ---- Thread counts to sweep ----
THREADS="1 4 8 16"

# ---- TILE sizes to sweep (no-BLAS only) ----
TILES="32 64 128"

# ==========================================
# Part 1: GEMM — vary TILE + threads
# ==========================================
echo ""
echo "=== Hand-coded GEMM ==="

for TILE in $TILES; do
    make clean -s
    make TILE=$TILE -s
    for T in $THREADS; do
        export OMP_NUM_THREADS=$T
        echo ""
        echo "--- TILE=$TILE  Threads=$T ---"
        ./nn
    done
done

# ==========================================
# Part 2: OpenBLAS — vary threads only
# ==========================================
echo ""
echo "=== OpenBLAS ==="

make clean -s
make BLAS=1 -s

for T in $THREADS; do
    export OMP_NUM_THREADS=$T
    echo ""
    echo "--- BLAS  Threads=$T ---"
    ./nn
done

echo ""
echo "========================================"
echo "  Done"
echo "========================================"
