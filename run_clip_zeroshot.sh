#!/bin/bash
# Zero-shot CLIP (RN50) evaluation on ImageNet-100.
# Usage: IMAGENET100_DIR=/path/to/imagenet-100 sbatch run_clip_zeroshot.sh
#SBATCH --job-name=clip_zeroshot_eval
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=01:00:00
#SBATCH --output=clip_zeroshot_eval_%j.log
#SBATCH --error=clip_zeroshot_eval_%j.err 
#SBATCH --partition=gpu_mig40,gpu,spgpu
#SBATCH --gres=gpu:1
: "${IMAGENET100_DIR:?Set IMAGENET100_DIR to the ImageNet-100 root (the folder that contains val/)}"
echo "hello world"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate debiaspl

python clip_vs_debiaspl_analysis.py --clip --data "$IMAGENET100_DIR"
echo "done"
