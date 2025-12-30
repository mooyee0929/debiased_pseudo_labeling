# The entire homework can be completed on this file. 
# It is derived from main_DebiasPL(_ZeroShot).py

# The command to run this code on 1 gpu is: 
# python starter.py
# add the tag --clip to load untrained CLIP (question 1), without the tag, the SSL model is loaded (q.2)

# RANDOM ASIDE: You can also check out the huggingface imagenet data. 
# To access it, you need to create a free huggingface account
# then, in terminal type: huggingface-cli login
# follow the link to create a token (giving it read permissions to public gated repos) 
# copy the token to login
# navigate to: https://huggingface.co/datasets/ILSVRC/imagenet-1k
# click agree and access repository


import argparse
import builtins
import math
import os
import shutil
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler

from torchvision import transforms

import data.datasets as datasets
import backbone as backbone_models
from models import get_fixmatch_model
from utils import utils, lr_schedule, get_norm, dist_utils
import data.transforms as data_transforms
# from engine import validate
# we are making a new validate function
from torch.utils.tensorboard import SummaryWriter

import clip

from tqdm import tqdm




backbone_model_names = sorted(name for name in backbone_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(backbone_models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', default = '', metavar='DIR',
                    help='path to dataset')
# OPTIONAL: you can modify to add your ImageNet100 path here
parser.add_argument('--arch', metavar='ARCH', default='FixMatch',
                    help='model architecture')
parser.add_argument('--backbone', default='resnet50_encoder',
                    choices=backbone_model_names,
                    help='model architecture: ' +
                        ' | '.join(backbone_model_names) +
                        ' (default: resnet50_encoder)')
parser.add_argument('--cls', default=1000, type=int, metavar='N',
                    help='number of classes')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('-c', '--clip', dest='clip', action='store_true',
                    help='use regular clip')
parser.add_argument('--pretrained', default='', 
                    type=str, metavar='PATH',
                    help='path to pretrained model (default: none)')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--eman', action='store_true', default=False,
                    help='use EMAN')
parser.add_argument('--ema-m', default=0.999, type=float,
                    help='EMA decay rate')
parser.add_argument('--norm', default='None', type=str,
                    help='the normalization for backbone (default: None)')

best_acc1 = 0




class NoTensorImageFolder(datasets.ImageFolder):
    def __init__(self, root):
        super().__init__(root)

    def __getitem__(self, index):
        original_image, original_label = super().__getitem__(index)
        return original_image, original_label


def main():
    args = parser.parse_args()
    print(args)

    # 添加一個全局變數來儲存 Q1 的順序
    global q1_sorted_indices
    q1_sorted_indices = None

    if not args.clip:
        # create model
        print("=> creating model '{}' with backbone '{}'".format(args.arch, args.backbone))
        model_func = get_fixmatch_model(args.arch)
        norm = get_norm(args.norm)
        model = model_func(
            backbone_models.__dict__[args.backbone],
            eman=args.eman,
            momentum=args.ema_m,
            norm=norm
        )
        # print(model)
        # print("Total params: {:.2f}M".format(sum(p.numel() for p in model.parameters())/1e6))

    elif args.clip:
        # model is clip
        model, preprocess = clip.load("RN50")
        print('Using CLIP')


    if args.pretrained and not args.clip:
        # load trained weights 
        if os.path.isfile(args.pretrained):
            print("=> loading pretrained model from '{}'".format(args.pretrained))
            checkpoint = torch.load(args.pretrained, map_location="cpu")
            state_dict = checkpoint['state_dict']
            for k in list(state_dict.keys()):
                if "module.main" in k:
                    new_key = k.replace("module.", "")
                    state_dict[new_key] = state_dict[k]
                del state_dict[k]
            model.load_state_dict(state_dict)
            print("=> loaded pre-trained model '{}' (epoch {})".format(args.pretrained, checkpoint['epoch']))
        else:
            print("=> no pretrained model found at '{}'".format(args.pretrained))

    model.cuda() 
    try:
        preprocess.cuda() 
    except:
        pass

    criterion = nn.CrossEntropyLoss().cuda()

    cudnn.benchmark = True

    transform_val = data_transforms.get_transforms('DefaultVal')

    if not args.clip:
        # TODO: Edit validate_542_logit for q.2 
        # This is how you will generate the graphs you need
        validate_542_logit(transform_val, model, criterion, args)
        return
    
    elif args.clip:
        # TODO: Edit validate_542_clip for q.1 
        # This is how you will generate the graphs you need
        validate_542_clip(model, preprocess, args)
        return
    

def get_centroids(prob):
    # this is from the original code. It was used for CLDLoss
    # might be helpful for your implementation in part c
    # but you should not use it directly for getting centroids!
    N, D = prob.shape
    K = D
    cl = prob.argmin(dim=1).long().view(-1)  # -> class index
    Ncl = cl.view(cl.size(0), 1).expand(-1, D)
    unique_labels, labels_count = Ncl.unique(dim=0, return_counts=True)
    labels_count_all = torch.ones([K]).long().cuda() # -> counts of each class
    labels_count_all[unique_labels[:,0]] = labels_count
    c = torch.zeros([K, D], dtype=prob.dtype).cuda().scatter_add_(0, Ncl, prob) # -> class centroids
    c = c / labels_count_all.float().unsqueeze(1)
    return cl, c

def CLDLoss(prob_s, prob_w, mask=None, weights=None):
    # this is from the original code, not used here
    cl_w, c_w = get_centroids(prob_w)
    affnity_s2w = torch.mm(prob_s, c_w.t())
    if mask is None:
        loss = F.cross_entropy(affnity_s2w.div(0.07), cl_w, weight=weights)
    else:
        loss = (F.cross_entropy(affnity_s2w.div(0.07), cl_w, reduction='none', weight=weights) * (1 - mask)).mean()
    return loss


def group_batch(batch):
    return {k: [v] for k, v in batch.items()}

def collate_fn(batch):
    images, labels = zip(*batch)
    return list(images), list(labels)


#################################################


def plot_precision_recall_histogram(results, save_path='precision_recall_plot.png'):
    total_counts = results['total_counts']
    precisions = results['precisions']
    recalls = results['recalls']
    sorted_indices = results['sorted_indices']
    
    sorted_counts = [total_counts[i] for i in sorted_indices]
    sorted_precisions = [precisions[i] for i in sorted_indices]
    sorted_recalls = [recalls[i] for i in sorted_indices]
    
    x = np.arange(len(sorted_indices))
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # ========== Top plot: Precision ==========
    ax1_right = ax1.twinx()
    
    # smooth version
    # ax1_right.fill_between(x, sorted_counts, alpha=0.3, color='gray')
    # ax1_right.plot(x, sorted_counts, color='gray', linewidth=2, alpha=0.7)

    #histogram version
    ax1_right.bar(x, sorted_counts, alpha=0.3, color='gray', width=0.7,edgecolor='darkgray', linewidth=0.5)

    
    ax1.scatter(x, sorted_precisions, alpha=0.6, s=10, color='#4A90E2')
    
    # 使用多項式回歸代替移動平均
    # 可以調整 degree 來控制平滑程度 (3-5 通常效果不錯)
    degree = 3
    z_precision = np.polyfit(x, sorted_precisions, degree)
    p_precision = np.poly1d(z_precision)
    precision_smooth = p_precision(x)

    ax1.plot(x, precision_smooth, color='#4A90E2', linewidth=2.5)
    
    ax1.set_ylabel('Precision', fontsize=14, color='#4A90E2')
    ax1.tick_params(axis='y', labelcolor='#4A90E2')
    ax1.set_ylim(0, 1.0)
    ax1.set_xlim(0, len(sorted_indices))
    ax1.grid(False)
    
    ax1_right.set_ylabel('# of Predictions', fontsize=14)
    ax1_right.set_ylim(0, max(sorted_counts) * 1.1)
    ax1_right.grid(False)
    ax1.set_xticklabels([])
    
    # ========== Bottom plot: Recall ==========
    ax2_right = ax2.twinx()
    
    # smoothed version
    # ax2_right.fill_between(x, sorted_counts, alpha=0.3, color='gray')
    # ax2_right.plot(x, sorted_counts, color='gray', linewidth=2, alpha=0.7)

    # histogram version
    ax2_right.bar(x, sorted_counts, alpha=0.3, color='gray', width=0.7,edgecolor='darkgray', linewidth=0.5)
    
    ax2.scatter(x, sorted_recalls, alpha=0.6, s=10, color='#E87D3E')
    
    z_recall = np.polyfit(x, sorted_recalls, degree)
    p_recall = np.poly1d(z_recall)
    recall_smooth = p_recall(x)

    ax2.plot(x, recall_smooth, color='#E87D3E', linewidth=2.5)
    
    ax2.set_ylabel('Recall', fontsize=14, color='#E87D3E')
    ax2.tick_params(axis='y', labelcolor='#E87D3E')
    ax2.set_ylim(0, 1.0)
    ax2.set_xlim(0, len(sorted_indices))
    ax2.set_xlabel('Ranked Class Index', fontsize=14)
    ax2.grid(False)
    
    ax2_right.set_ylabel('# of Predictions', fontsize=14)
    ax2_right.set_ylim(0, max(sorted_counts) * 1.1)
    ax2_right.grid(False)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    plt.show()
    
    return fig

#################################################


#################################################
def plot_confusion_matrix(confusion_matrix, class_names, sorted_indices, save_path='confusion_matrix.png'):
    """
    Plot confusion matrix ordered by prediction frequency (row-wise).
    Rows are ordered by total number of predictions FOR that class (sum across columns).
    
    Args:
        confusion_matrix: 100x100 numpy array [true_class, predicted_class]
        class_names: Dictionary mapping class index to class name
        sorted_indices: List of class indices sorted by prediction frequency (high to low)
        save_path: Path to save the figure
    """
    
    # Reorder confusion matrix: rows by predicted frequency, columns by predicted frequency
    reordered_cm = confusion_matrix[sorted_indices, :][:, sorted_indices]
    
    # Get reordered class names
    reordered_names = [class_names[i] for i in sorted_indices]
    
    # Create very large figure
    fig, ax = plt.subplots(figsize=(30, 28))

    col_sums = confusion_matrix.sum(axis=0)  # Sum across rows for each column
    sorted_indices_cols = sorted(range(100), key=lambda x: col_sums[x], reverse=True)
    
    # Plot heatmap
    im = ax.imshow(reordered_cm, cmap='RdBu_r', aspect='auto', interpolation='nearest')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('# of Predictions', rotation=270, labelpad=30, fontsize=16)
    cbar.ax.tick_params(labelsize=14)
    
    # Set all ticks
    tick_positions = list(range(100))
    ax.set_xticks(tick_positions)
    ax.set_yticks(tick_positions)
    
    # Set labels
    ax.set_xticklabels(reordered_names, rotation=90, ha='center', fontsize=8)
    ax.set_yticklabels(reordered_names, fontsize=8)
    
    # Add text annotations for ALL cells
    max_val = reordered_cm.max()
    for i in range(100):
        for j in range(100):
            value = int(reordered_cm[i, j])
            if value > 0:  # Only show non-zero values
                # Choose text color based on background intensity
                text_color = "white" if reordered_cm[i, j] > max_val * 0.5 else "black"
                
                # Make diagonal values bold
                if i == j:
                    ax.text(j, i, value, ha="center", va="center", 
                           color=text_color, fontsize=5, fontweight='bold')
                else:
                    ax.text(j, i, value, ha="center", va="center", 
                           color=text_color, fontsize=4)
    
    # Labels and title
    ax.set_xlabel('Predicted Class', fontsize=16, labelpad=10)
    ax.set_ylabel('True Class (Ordered by Prediction Frequency)', fontsize=16, labelpad=10)
    ax.set_title('Confusion Matrix (Ordered by Total Predictions per Class)', 
                 fontsize=18, pad=20, fontweight='bold')
    
    # Add subtle grid
    ax.set_xticks([x - 0.5 for x in range(1, 100)], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, 100)], minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.1, alpha=0.2)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to {save_path}")
    print(f"Top 5 most predicted classes: {[reordered_names[i] for i in range(5)]}")
    print(f"Column sums (top 5): {[int(col_sums[sorted_indices_cols[i]]) for i in range(5)]}")
    plt.close()
    
    return fig

####################################################


####################################################

def analyze_centroid_similarity(class_logits, prediction_counts, imagenet100_to_name, 
                                use_q1_classes=False):
    """
    Analyze centroid similarity given pre-collected logits.
    
    Args:
        class_logits: Dict mapping class_idx -> list of logit tensors
        prediction_counts: List of prediction counts per class
        imagenet100_to_name: Dict mapping class_idx -> class_name
        use_q1_classes: If True, load most/least predicted from Q1
    """
    
    print("\n" + "="*80)
    print("Computing logit centroids and similarity analysis...")
    print("="*80)
    
    # Compute centroids
    centroids = torch.zeros(100, 100)
    for cls in range(100):
        if len(class_logits[cls]) > 0:
            centroids[cls] = torch.stack(class_logits[cls]).mean(dim=0)
    
    # Normalize for cosine similarity
    centroids_normalized = F.normalize(centroids, p=2, dim=1)
    similarity_matrix = centroids_normalized @ centroids_normalized.T
    
    # Determine which classes to analyze
    if use_q1_classes and os.path.exists('q1_prediction_info.pkl'):
        with open('q1_prediction_info.pkl', 'rb') as f:
            q1_info = pickle.load(f)
        most_predicted_idx = q1_info['most_predicted_idx']
        least_predicted_idx = q1_info['least_predicted_idx']
        print("\n=== Using Q1's most/least predicted classes ===")
        print(f"Q1 Most predicted: {q1_info['most_predicted_name']}")
        print(f"Q1 Least predicted: {q1_info['least_predicted_name']}")
    else:
        most_predicted_idx = np.argmax(prediction_counts)
        least_predicted_idx = np.argmin(prediction_counts)
        if use_q1_classes:
            print("\n=== WARNING: Q1 prediction info not found, using current model's predictions ===")
    
    print(f"\nAnalyzing: Most predicted = idx {most_predicted_idx} ('{imagenet100_to_name[most_predicted_idx]}'), count={prediction_counts[most_predicted_idx]}")
    print(f"Analyzing: Least predicted = idx {least_predicted_idx} ('{imagenet100_to_name[least_predicted_idx]}'), count={prediction_counts[least_predicted_idx]}")
    
    results = {}
    for cls_idx, cls_name in [(most_predicted_idx, 'most_predicted'), 
                               (least_predicted_idx, 'least_predicted')]:
        similarities = similarity_matrix[cls_idx].numpy()
        similarities[cls_idx] = -2  # Exclude self
        
        top10_indices = np.argsort(similarities)[::-1][:10]
        top10_similarities = similarities[top10_indices]
        top10_names = [imagenet100_to_name[idx] for idx in top10_indices]
        
        results[cls_name] = {
            'class_idx': cls_idx,
            'class_name': imagenet100_to_name[cls_idx],
            'top10_indices': top10_indices.tolist(),
            'top10_similarities': top10_similarities.tolist(),
            'top10_names': top10_names
        }
        
        print(f"\n{cls_name.replace('_', ' ').title()}: {imagenet100_to_name[cls_idx]}")
        print(f"{'Rank':<6} {'Class Name':<30} {'Index':<8} {'Similarity':<12}")
        print("-" * 60)
        for i, (idx, sim, name) in enumerate(zip(top10_indices, top10_similarities, top10_names)):
            print(f"{i+1:<6} {name:<30} {idx:<8} {sim:.6f}")
    
    return results

####################################################

def validate_542_clip(model, preprocess, args): 
    # TODO: Implement the visualizations needed here 
    valdir = os.path.join(args.data, 'val')
    val_dataset = NoTensorImageFolder(valdir)

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, collate_fn=collate_fn
    ) 
    imagenet100_to_name = {0: 'robin', 1: 'Gila_monster', 2: 'hognose_snake', 3: 'garter_snake', 4: 'green_mamba', 5: 'garden_spider', 6: 'lorikeet', 7: 'goose', 8: 'rock_crab', 9: 'fiddler_crab', 10: 'American_lobster', 11: 'little_blue_heron', 12: 'American_coot', 13: 'Chihuahua', 14: 'Shih-Tzu', 15: 'papillon', 16: 'toy_terrier', 17: 'Walker_hound', 18: 'English_foxhound', 19: 'borzoi', 20: 'Saluki', 21: 'American_Staffordshire_terrier', 22: 'Chesapeake_Bay_retriever', 23: 'vizsla', 24: 'kuvasz', 25: 'komondor', 26: 'Rottweiler', 27: 'Doberman', 28: 'boxer', 29: 'Great_Dane', 30: 'standard_poodle', 31: 'Mexican_hairless', 32: 'coyote', 33: 'African_hunting_dog', 34: 'red_fox', 35: 'tabby', 36: 'meerkat', 37: 'dung_beetle', 38: 'walking_stick', 39: 'leafhopper', 40: 'hare', 41: 'wild_boar', 42: 'gibbon', 43: 'langur', 44: 'ambulance', 45: 'bannister', 46: 'bassinet', 47: 'boathouse', 48: 'bonnet', 49: 'bottlecap', 50: 'car_wheel', 51: 'chime', 52: 'cinema', 53: 'cocktail_shaker', 54: 'computer_keyboard', 55: 'Dutch_oven', 56: 'football_helmet', 57: 'gasmask', 58: 'hard_disc', 59: 'harmonica', 60: 'honeycomb', 61: 'iron', 62: 'jean', 63: 'lampshade', 64: 'laptop', 65: 'milk_can', 66: 'mixing_bowl', 67: 'modem', 68: 'moped', 69: 'mortarboard', 70: 'mousetrap', 71: 'obelisk', 72: 'park_bench', 73: 'pedestal', 74: 'pickup', 75: 'pirate', 76: 'purse', 77: 'reel', 78: 'rocking_chair', 79: 'rotisserie', 80: 'safety_pin', 81: 'sarong', 82: 'ski_mask', 83: 'slide_rule', 84: 'stretcher', 85: 'theater_curtain', 86: 'throne', 87: 'tile_roof', 88: 'tripod', 89: 'tub', 90: 'vacuum', 91: 'window_screen', 92: 'wing', 93: 'head_cabbage', 94: 'cauliflower', 95: 'pineapple', 96: 'carbonara', 97: 'chocolate_sauce', 98: 'gyromitra', 99: 'stinkhorn'}

    names = list(imagenet100_to_name.values())
    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in names])

    text_inputs = text_inputs.cuda()

    with torch.no_grad():
        text_features = model.encode_text(text_inputs)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    top1 = utils.AverageMeter('Acc@1', ':6.2f')
    top5 = utils.AverageMeter('Acc@5', ':6.2f')

    # Initialize counters for precision and recall
    true_positives = [0 for _ in range(100)]  # TP for each class
    false_positives = [0 for _ in range(100)]  # FP for each class
    false_negatives = [0 for _ in range(100)]  # FN for each class
    total_counts = [0 for _ in range(100)]  # Total predictions per class

    # Initialize confusion matrix
    confusion_matrix = np.zeros((100, 100), dtype=np.int32)

    class_logits = {i: [] for i in range(100)}  # Store logits per class for centroid analysis


    # switch to evaluate mode
    model.eval()

    start_time = time.time()
    with torch.no_grad():

        total_counts = [0 for _ in range(100)]

        for _, i in tqdm(enumerate(val_loader)): #dataset

            images_raw = i[0]
            target = torch.tensor(i[1])
            

            images = []
            for ii in range(len(images_raw)):
                images.append(preprocess(images_raw[ii]))

            images = torch.stack(images, dim=0)

            images = images.cuda()
            target = target.cuda() 

            image_features = model.encode_image(images)

            image_features /= image_features.norm(dim=-1, keepdim=True)

            

            logits = 100.0 * image_features @ text_features.T  # store logits for centroid analysis
            probs = logits.softmax(dim=-1)  #similarity

            pred = torch.argmax(probs, dim=-1)

            # Update counters
            for idx, (p, t) in enumerate(zip(pred, target)):
                
                p_item = p.item()
                t_item = t.item()

                class_logits[t_item].append(logits[idx].cpu()) # Store logits for true class

                confusion_matrix[t_item, p_item] += 1
                total_counts[p_item] += 1
                
                if p_item == t_item:
                    true_positives[p_item] += 1
                else:
                    false_positives[p_item] += 1
                    false_negatives[t_item] += 1
            
            acc1, acc5 = utils.accuracy(similarity, target, topk=(1, 5))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

    # Calculate precision and recall for each class
    precisions = []
    recalls = []

    for cls in range(100):
        # Precision = TP / (TP + FP)
        if true_positives[cls] + false_positives[cls] > 0:
            precision = true_positives[cls] / (true_positives[cls] + false_positives[cls])
        else:
            precision = 0.0
        precisions.append(precision)
        
        # Recall = TP / (TP + FN)
        if true_positives[cls] + false_negatives[cls] > 0:
            recall = true_positives[cls] / (true_positives[cls] + false_negatives[cls])
        else:
            recall = 0.0
        recalls.append(recall)

    print("\n=== Results ===")
    print(f"Total predictions per class: {total_counts}")
    print(f"\n * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}")
    print(f"Total time: {time.time()-start_time:.2f}s")
    
    # Sort classes by total_counts (descending order) for visualization
    sorted_indices = sorted(range(100), key=lambda x: total_counts[x], reverse=True)
    
    print("\n=== Per-class Precision and Recall (sorted by prediction frequency) ===")
    print(f"{'Rank':<6} {'Class':<5} {'Name':<30} {'Predictions':<12} {'Precision':<10} {'Recall':<10}")
    print("-" * 80)
    for rank, cls in enumerate(sorted_indices[:10]):  # Print top 10
        print(f"{rank+1:<6} {cls:<5} {imagenet100_to_name[cls]:<30} {total_counts[cls]:<12} {precisions[cls]:.4f}    {recalls[cls]:.4f}")
    
    print("\n... (showing top 10 only)")

    # At the end of validate_542_clip function
    results = {
        'total_counts': total_counts,
        'precisions': precisions,
        'recalls': recalls,
        'sorted_indices': sorted_indices,
        'class_names': imagenet100_to_name,
        'confusion_matrix': confusion_matrix

    }

    # Generate plot
    plot_precision_recall_histogram(results, save_path='q1_precision_recall.png')
    plot_confusion_matrix(confusion_matrix, imagenet100_to_name, sorted_indices, save_path='q1_confusion_matrix.png')



    centroid_results = analyze_centroid_similarity(class_logits, total_counts, imagenet100_to_name)
    results['centroid_analysis'] = centroid_results

    # 新增：保存 most/least predicted classes
    most_predicted_idx = np.argmax(total_counts)
    least_predicted_idx = np.argmin(total_counts)

    q1_prediction_info = {
        'most_predicted_idx': most_predicted_idx,
        'least_predicted_idx': least_predicted_idx,
        'most_predicted_name': imagenet100_to_name[most_predicted_idx],
        'least_predicted_name': imagenet100_to_name[least_predicted_idx]
    }

    with open('q1_prediction_info.pkl', 'wb') as f:
        pickle.dump(q1_prediction_info, f)


    with open('q1_sorted_indices.pkl', 'wb') as f:
        pickle.dump(results['sorted_indices'], f)

    # print(total_counts)
    # print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
    #           .format(top1=top1, top5=top5))  
    # print('total time: ', time.time()-start_time)
    
    return top1.avg



def validate_542_logit(transform_val, model, criterion, args): 
    # TODO: Implement the visualizations needed here 

    imagenet100_to_1K = {0: 15, 1: 45, 2: 54, 3: 57, 4: 64, 5: 74, 6: 90, 7: 99, 8: 119, 9: 120, 10: 122, 11: 131,
                        12: 137, 13: 151, 14: 155, 15: 157, 16: 158, 17: 166, 18: 167, 19: 169, 20: 176, 21: 180, 
                        22: 209, 23: 211, 24: 222, 25: 228, 26: 234, 27: 236, 28: 242, 29: 246, 30: 267,
                        31: 268, 32: 272, 33: 275, 34: 277, 35: 281, 36: 299, 37: 305, 38: 313, 39: 317, 40: 331, 
                        41: 342, 42: 368, 43: 374, 44: 407, 45: 421, 46: 431, 47: 449, 48: 452, 49: 455, 50: 479,
                        51: 494, 52: 498, 53: 503, 54: 508, 55: 544, 56: 560, 57: 570, 58: 592, 59: 593,
                        60: 599, 61: 606, 62: 608, 63: 619, 64: 620, 65: 653, 66: 659, 67: 662, 68: 665, 69: 667, 
                        70: 674, 71: 682, 72: 703, 73: 708, 74: 717, 75: 724, 76: 748, 77: 758, 78: 765, 79: 766, 
                        80: 772, 81: 775, 82: 796, 83: 798, 84: 830, 85: 854, 86: 857, 87: 858, 88: 872,
                        89: 876, 90: 882, 91: 904, 92: 908, 93: 936, 94: 938, 95: 953, 96: 959, 97: 960, 98: 993, 99: 994} 

    keep_ind = list(imagenet100_to_1K.values())

    imagenet100_to_name = {0: 'robin', 1: 'Gila_monster', 2: 'hognose_snake', 3: 'garter_snake', 4: 'green_mamba', 5: 'garden_spider', 6: 'lorikeet', 7: 'goose', 8: 'rock_crab', 9: 'fiddler_crab', 10: 'American_lobster', 11: 'little_blue_heron', 12: 'American_coot', 13: 'Chihuahua', 14: 'Shih-Tzu', 15: 'papillon', 16: 'toy_terrier', 17: 'Walker_hound', 18: 'English_foxhound', 19: 'borzoi', 20: 'Saluki', 21: 'American_Staffordshire_terrier', 22: 'Chesapeake_Bay_retriever', 23: 'vizsla', 24: 'kuvasz', 25: 'komondor', 26: 'Rottweiler', 27: 'Doberman', 28: 'boxer', 29: 'Great_Dane', 30: 'standard_poodle', 31: 'Mexican_hairless', 32: 'coyote', 33: 'African_hunting_dog', 34: 'red_fox', 35: 'tabby', 36: 'meerkat', 37: 'dung_beetle', 38: 'walking_stick', 39: 'leafhopper', 40: 'hare', 41: 'wild_boar', 42: 'gibbon', 43: 'langur', 44: 'ambulance', 45: 'bannister', 46: 'bassinet', 47: 'boathouse', 48: 'bonnet', 49: 'bottlecap', 50: 'car_wheel', 51: 'chime', 52: 'cinema', 53: 'cocktail_shaker', 54: 'computer_keyboard', 55: 'Dutch_oven', 56: 'football_helmet', 57: 'gasmask', 58: 'hard_disc', 59: 'harmonica', 60: 'honeycomb', 61: 'iron', 62: 'jean', 63: 'lampshade', 64: 'laptop', 65: 'milk_can', 66: 'mixing_bowl', 67: 'modem', 68: 'moped', 69: 'mortarboard', 70: 'mousetrap', 71: 'obelisk', 72: 'park_bench', 73: 'pedestal', 74: 'pickup', 75: 'pirate', 76: 'purse', 77: 'reel', 78: 'rocking_chair', 79: 'rotisserie', 80: 'safety_pin', 81: 'sarong', 82: 'ski_mask', 83: 'slide_rule', 84: 'stretcher', 85: 'theater_curtain', 86: 'throne', 87: 'tile_roof', 88: 'tripod', 89: 'tub', 90: 'vacuum', 91: 'window_screen', 92: 'wing', 93: 'head_cabbage', 94: 'cauliflower', 95: 'pineapple', 96: 'carbonara', 97: 'chocolate_sauce', 98: 'gyromitra', 99: 'stinkhorn'}


    losses = utils.AverageMeter('Loss', ':.4e')
    top1 = utils.AverageMeter('Acc@1', ':6.2f')
    top5 = utils.AverageMeter('Acc@5', ':6.2f')

    # Initialize counters
    true_positives = [0 for _ in range(100)]
    false_positives = [0 for _ in range(100)]
    false_negatives = [0 for _ in range(100)]
    total_counts = [0 for _ in range(100)]
    
    # Initialize confusion matrix
    confusion_matrix = np.zeros((100, 100), dtype=np.int32)

    class_logits = {i: [] for i in range(100)}  # Store logits per class for centroid analysis


    # switch to evaluate mode
    model.eval()

    start_time = time.time()
    with torch.no_grad():
        end = time.time()

        ## Optional: if want to use preloaded val dataset, each loop takes ~20-30sec 
        # in this case, overwrite like:

        valdir = os.path.join(args.data, 'val')
        val_dataset = datasets.ImageFolder(
                valdir, transform=transform_val)
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True) 
        
        # then, enumerate val_loader instead of val_dataset
        # inside the loop, no longer transform, just get images and target



        for _, i in tqdm(enumerate(val_loader)): #optional: _loader

            images = i[0]
            target = i[1]

            images = images.cuda() 
            target = target.cuda() 

            output = model(images)

            output = output[:, keep_ind]

            loss = criterion(output, target)
            probs = torch.softmax(output, dim=-1)

            pred = torch.argmax(output, dim=-1)



            for idx, (p, t) in enumerate(zip(pred, target)):
                p_item = p.item()
                t_item = t.item()

                class_logits[t.item()].append(output[idx].cpu())  # Store logits for true class
                
                confusion_matrix[t_item, p_item] += 1
                total_counts[p_item] += 1
                
                if p_item == t_item:
                    true_positives[p_item] += 1
                else:
                    false_positives[p_item] += 1
                    false_negatives[t_item] += 1


            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

    # Calculate precision and recall
    precisions = []
    recalls = []
    
    for cls in range(100):
        if true_positives[cls] + false_positives[cls] > 0:
            precision = true_positives[cls] / (true_positives[cls] + false_positives[cls])
        else:
            precision = 0.0
        precisions.append(precision)
        
        if true_positives[cls] + false_negatives[cls] > 0:
            recall = true_positives[cls] / (true_positives[cls] + false_negatives[cls])
        else:
            recall = 0.0
        recalls.append(recall)

    print("\n=== Results ===")
    print(f"Total predictions per class: {total_counts}")
    print(f"\n * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f} Loss {losses.avg:.4f}")
    print(f"Total time: {time.time()-start_time:.2f}s")


    # Sort classes by prediction frequency (column sum)
    col_sums = confusion_matrix.sum(axis=0)
    sorted_indices = sorted(range(100), key=lambda x: col_sums[x], reverse=True)
    
    
    # Store results
    results = {
        'total_counts': total_counts,
        'precisions': precisions,
        'recalls': recalls,
        'sorted_indices': sorted_indices,
        'class_names': imagenet100_to_name,
        'confusion_matrix': confusion_matrix
    }
    
    # Generate plots
    plot_precision_recall_histogram(results, save_path='q2f_precision_recall.png')

    if os.path.exists('q1_sorted_indices.pkl'):
        with open('q1_sorted_indices.pkl', 'rb') as f:
            sorted_indices = pickle.load(f)
        print("\n=== Using Q1's class ordering ===")
    else:
        print("\n=== WARNING: Q1 ordering file not found! ===")
        print("Please run Q1 first with --clip flag")
        # Use current ordering
        col_sums = confusion_matrix.sum(axis=0)
        sorted_indices = sorted(range(100), key=lambda x: col_sums[x], reverse=True)
        print("Using current ordering instead.")

    results = {
        'total_counts': total_counts,
        'precisions': precisions,
        'recalls': recalls,
        'sorted_indices': sorted_indices,
        'class_names': imagenet100_to_name,
        'confusion_matrix': confusion_matrix
    }
    
    plot_precision_recall_histogram(results, save_path='q2_precision_recall.png')
    plot_confusion_matrix(confusion_matrix, imagenet100_to_name, sorted_indices, 
                         save_path='q2_confusion_matrix.png')
    

    print("\n" + "="*80)
    print("PART 1: Analyzing Q2's own most/least predicted classes")
    print("="*80)
    centroid_results_q2 = analyze_centroid_similarity(
        class_logits, total_counts, imagenet100_to_name, 
        use_q1_classes=False
    )
    results['centroid_analysis_q2'] = centroid_results_q2

    # ========== Centroid Analysis: Q1's most/least predicted (on Q2 model) ==========
    print("\n" + "="*80)
    print("PART 2: Analyzing Q1's most/least predicted classes (using Q2's logits)")
    print("="*80)
    centroid_results_q1_on_q2 = analyze_centroid_similarity(
        class_logits, total_counts, imagenet100_to_name, 
        use_q1_classes=True
    )
    # results['centroid_analysis_q1_on_q2'] = centroid_results_q1_on_q2

    # print(total_counts)
    # print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f} Loss {loss.avg:.4f}'
    #           .format(top1=top1, top5=top5, loss=losses))  
    # print('total time: ', time.time()-start_time)
    
    return top1.avg



if __name__ == '__main__':
    main()
