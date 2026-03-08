#!/bin/bash
#
#SBATCH --mail-user=adii@rcc.uchicago.edu
#SBATCH --mail-type=ALL
#SBATCH --job-name=hpc_p3_milestone2_openmp_blas
#SBATCH --output=/home/adii/HPC/project-3-winter-2026-adi-0104/milestone2/bash/slurm/out/%j.%N.stdout
#SBATCH --error=/home/adii/HPC/project-3-winter-2026-adi-0104/milestone2/bash/slurm/out/%j.%N.stderr
#SBATCH --chdir=/home/adii/HPC/project-3-winter-2026-adi-0104/milestone2
#SBATCH --account=mpcs51087
#SBATCH --partition=caslake
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --time=10:00

module load gcc
module load openblas

THREADS="10 16 32"
TILES="32"

# --- Build ---
echo "batch size 512"
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
