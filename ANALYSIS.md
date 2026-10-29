# CLIP vs. DebiasPL Analysis: Quick Reference

## Environment Setup

conda create -n debiaspl python=3.8

conda activate debiaspl

pip install -r requirement.txt

pip install git+https://github.com/openai/CLIP.git

download the DebiasPL zero-shot checkpoint (debiaspl_zsl_epoch_50.pth.tar) from https://drive.google.com/file/d/1UB_9c-2Hr0YVkg_qq24f_D9tsJHT7EjY/view?usp=sharing

## How to run

export IMAGENET100_DIR=/path/to/imagenet-100

sbatch run_clip_zeroshot.sh

sbatch run_debiaspl_eval.sh


## Expected Output Files

clip_confusion_matrix.png

clip_precision_recall.png

clip_prediction_info.pkl

clip_sorted_indices.pkl

debiaspl_confusion_matrix_clip_order.png

debiaspl_precision_recall_clip_order.png

debiaspl_precision_recall.png

