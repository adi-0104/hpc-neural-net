#!/bin/bash
#
#SBATCH --mail-user=adii@rcc.uchicago.edu
#SBATCH --mail-type=ALL
#SBATCH --job-name=p3_m2_cuda_test
#SBATCH --output=/home/adii/HPC/project-3-winter-2026-adi-0104/milestone2/bash/slurm/out/%j.%N.stdout
#SBATCH --error=/home/adii/HPC/project-3-winter-2026-adi-0104/milestone2/bash/slurm/out/%j.%N.stderr
#SBATCH --chdir=/home/adii/HPC/project-3-winter-2026-adi-0104/milestone2/gpu
#SBATCH --account=mpcs51087
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=10:00

module load gcc
module load cuda


echo "Running "
./nn_gpu
echo "Finished run."

