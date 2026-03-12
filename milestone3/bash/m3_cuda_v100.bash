#!/bin/bash
#
#SBATCH --mail-user=adii@rcc.uchicago.edu
#SBATCH --mail-type=ALL
#SBATCH --job-name=p3_m3_cuda_test
#SBATCH --output=/home/adii/HPC/project-3-winter-2026-adi-0104/milestone3/bash/slurm/out/%j.%N.stdout
#SBATCH --error=/home/adii/HPC/project-3-winter-2026-adi-0104/milestone3/bash/slurm/out/%j.%N.stderr
#SBATCH --chdir=/home/adii/HPC/project-3-winter-2026-adi-0104/milestone3/gpu
#SBATCH --account=mpcs51087
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=v100
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=5:00

module load gcc
module load cuda


echo "Running v100 test 256"
echo "Finished run."
echo "  "
echo "_____________________________"
echo "  "
echo "Running v100 test 512"
./nn_gpu_v100_500
echo "Finished run."
