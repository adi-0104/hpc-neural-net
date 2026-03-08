#!/bin/bash
#SBATCH --job-name=nn_gpu_bench
#SBATCH --output=bench_gpu_%j.out
#SBATCH --error=bench_gpu_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --mem=8G

module load cuda

cd $SLURM_SUBMIT_DIR

echo "========================================"
echo "  GPU Benchmark — $(date)"
echo "  Node: $SLURMD_NODENAME"
echo "========================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

make clean -s
make -s

echo ""
echo "--- GPU Baseline (batch=256) ---"
./nn_gpu

echo ""
echo "========================================"
echo "  Done"
echo "========================================"
