#!/bin/bash
#
#SBATCH --mail-user=adii@rcc.uchicago.edu
#SBATCH --mail-type=ALL
#SBATCH --job-name=p3_m3_cpu
#SBATCH --output=/home/adii/HPC/project-3-winter-2026-adi-0104/milestone3/bash/slurm/out/%j.%N.stdout
#SBATCH --error=/home/adii/HPC/project-3-winter-2026-adi-0104/milestone3/bash/slurm/out/%j.%N.stderr
#SBATCH --chdir=/home/adii/HPC/project-3-winter-2026-adi-0104/milestone3/cpu
#SBATCH --account=mpcs51087
#SBATCH --partition=caslake
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=20:00

module load gcc
module load openblas

THREADS="16 32"
TILE=32

echo "batch size 500, epochs 50, lr 0.1"
echo ""

# --- CPU TILED ---
echo "=== CPU Native (TILE=$TILE) ==="
make clean -s && make TILE=$TILE -s
for T in $THREADS; do
    export OMP_NUM_THREADS=$T
    echo ""
    echo "--- TILE=$TILE  Threads=$T ---"
    ./nn
done
mv loss.csv cpu_native.csv

echo ""

# --- CPU BLAS (OpenBLAS) ---
echo "=== CPU BLAS (OpenBLAS) ==="
make clean -s && make BLAS=1 -s
for T in $THREADS; do
    export OMP_NUM_THREADS=$T
    echo ""
    echo "--- BLAS  Threads=$T ---"
    ./nn
done
mv loss.csv cpu_blas.csv

echo ""
echo "Done."
