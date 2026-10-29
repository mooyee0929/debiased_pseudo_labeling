#!/bin/bash
# DebiasPL checkpoint evaluation on ImageNet-100 (run run_clip_zeroshot.sh first).
# Usage: IMAGENET100_DIR=/path/to/imagenet-100 sbatch run_debiaspl_eval.sh
#SBATCH --job-name=debiaspl_eval
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=01:00:00
#SBATCH --output=debiaspl_eval_%j.log
#SBATCH --error=debiaspl_eval_%j.err 
#SBATCH --partition=gpu_mig40,gpu,spgpu
#SBATCH --gres=gpu:1
: "${IMAGENET100_DIR:?Set IMAGENET100_DIR to the ImageNet-100 root (the folder that contains val/)}"
echo "hello world"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate debiaspl

python clip_vs_debiaspl_analysis.py --pretrained debiaspl_zsl_epoch_50.pth.tar --data "$IMAGENET100_DIR"
echo "done"
