# EECS 542 Homework 3: Debiased Pseudo-labeling

## Environment Setup

conda create -n HW3 python=3.8

conda activate HW3

pip install -r requirement.txt

pip install git+https://github.com/openai/CLIP.git

install checkpoint from https://drive.google.com/file/d/1UB_9c-2Hr0YVkg_qq24f_D9tsJHT7EjY/view?usp=sharing

## How to run

sbatch runq1.sh

sbatch runq2.sh


## Expected Exported File

q1_confusion_matrix.png

q1_precision_recall.png

q1_prediction_info.pkl

q1_sorted_indices.pkl

q2_confusion_matrix.png

q2_precision_recall.png

q2f_precision_recall.png

