#!/bin/bash
# The interpreter used to execute the script
#\#SBATCH" directives that convey submission options:
#SBATCH --job-name=fsyang_test
#SBATCH --mail-user=fsyang@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=01:00:00
#SBATCH --output=fsyang_test%j.log
#SBATCH --error=fsyang_test%j.err 
#SBATCH --partition=gpu_mig40,gpu,spgpu
#SBATCH --gres=gpu:1
echo "hello world"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate HW3

python starter.py --clip --data /scratch/eecs542f25_class_root/eecs542f25_class/shared_data/imagenet-100
echo "done"
