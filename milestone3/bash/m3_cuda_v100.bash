#!/bin/bash
#
#SBATCH --mail-user=adii@rcc.uchicago.edu
#SBATCH --mail-type=ALL
#SBATCH --job-name=p3_m3_gpu
#SBATCH --output=/home/adii/HPC/project-3-winter-2026-adi-0104/milestone3/bash/slurm/out/%j.%N.stdout
#SBATCH --error=/home/adii/HPC/project-3-winter-2026-adi-0104/milestone3/bash/slurm/out/%j.%N.stderr
#SBATCH --chdir=/home/adii/HPC/project-3-winter-2026-adi-0104/milestone3/gpu
#SBATCH --account=mpcs51087
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=v100
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=7:00

module load gcc
module load cuda

echo "batch size 500, epochs 50, lr 0.1"
echo ""

# --- GPU Naive ---
echo "=== GPU Native ==="
make clean -s && make -s
./nn_gpu_v100_500
mv loss.csv gpu_native.csv

echo ""

# --- GPU cuBLAS ---
echo "=== GPU cuBLAS ==="
make clean -s && make CUBLAS=1 -s
./nn_gpu_v100_500
mv loss.csv gpu_cublas.csv

echo ""
echo "========================================"
echo "Done. CSVs written: gpu_native.csv  gpu_cublas.csv"
