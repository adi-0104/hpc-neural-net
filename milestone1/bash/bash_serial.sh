#!/bin/bash
#
#SBATCH --mail-user=adii@rcc.uchicago.edu
#SBATCH --mail-type=ALL
#SBATCH --job-name=hpc_p3_milestone1_serial
#SBATCH --output=/home/adii/HPC/project-3-winter-2026-adi-0104/milestone1/bash/slurm/out/%j.%N.stdout
#SBATCH --error=/home/adii/HPC/project-3-winter-2026-adi-0104/milestone1/bash/slurm/out/%j.%N.stderr
#SBATCH --chdir=/home/adii/HPC/project-3-winter-2026-adi-0104/milestone1
#SBATCH --account=mpcs51087
#SBATCH --partition=caslake
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=15:00


# --- Build ---
echo "Compiling serial nn version..."
make clean && make
if [ $? -ne 0 ]; then
    echo "Build FAILED! Exiting."
    exit 1
fi

echo "Running..."
./nn
