import os
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from medpy.metric.binary import dc, hd95
from utils_test import refine_mask

# Image configuration
IMAGE_SIZE = 560

DATASET = ''
CATAGORY = ''

# Path settings
BASE_DIR = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(BASE_DIR, 'results', DATASET, CATAGORY,'masks')  
GROUND_TRUTH_DIR = os.path.join(BASE_DIR, 'dataset', DATASET, CATAGORY, 'target_masks')
time = datetime.now().strftime("%Y%m%d_%H%M")
OUTPUT_DIR = os.path.join(BASE_DIR, 'evaluation', DATASET, CATAGORY, time)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def dice_coefficient(y_true, y_pred):
    """Calculate Dice coefficient between two masks"""
    y_true = y_true.astype(np.bool_)
    y_pred = y_pred.astype(np.bool_)
    return dc(y_pred, y_true)

def hausdorff_distance(y_true, y_pred):
    """Calculate 95% Hausdorff Distance between two masks"""
    y_true = y_true.astype(np.bool_)
    y_pred = y_pred.astype(np.bool_)
    return hd95(y_pred, y_true)

def evaluate_segmentation(results_dir=RESULTS_DIR, ground_truth_dir=GROUND_TRUTH_DIR, output_dir=OUTPUT_DIR):
    result_files = sorted(glob(os.path.join(results_dir, "*.jpg")))
    
    if not result_files:
        raise FileNotFoundError(f"No mask images found in {results_dir}")
    
    results = []
    
    for result_file in tqdm(result_files, desc="Evaluating segmentation results"):
        # Get filename (remove _mask suffix and extension)
        filename = os.path.basename(result_file)
        # Find corresponding ground truth file
        gt_file = os.path.join(ground_truth_dir, filename)
        
        if not os.path.exists(gt_file):
            print(f"Warning: Ground truth not found for {filename}, skipping")
            continue
        
        # Read images
        result_img = cv2.imread(result_file, cv2.IMREAD_GRAYSCALE)
        gt_img = cv2.imread(gt_file, cv2.IMREAD_GRAYSCALE)
        
        # Ensure images are read correctly
        if result_img is None or gt_img is None:
            print(f"Warning: Cannot read images for {filename}, skipping")
            continue
        
        # Resize images to 560x560
        result_img = cv2.resize(result_img, (IMAGE_SIZE,IMAGE_SIZE))
        gt_img = cv2.resize(gt_img, (IMAGE_SIZE,IMAGE_SIZE))
        
        # Binarize images
        if np.max(result_img) > 1:
            result_img = (result_img > 127).astype(np.uint8)
        if np.max(gt_img) > 1:
            gt_img = (gt_img > 127).astype(np.uint8)
            

        result_img = refine_mask(result_img, threshold=0.5)
        gt_img = refine_mask(gt_img, threshold=0.5)

        # Calculate metrics
        dice = dice_coefficient(gt_img, result_img)
        hd_value = hausdorff_distance(gt_img, result_img)
        
        # Store results
        results.append({
            'filename': filename,
            'dice': dice,
            'hd': hd_value
        })
    
    # Create DataFrame
    results_df = pd.DataFrame(results)
    
    # Calculate statistics
    mean_dice = results_df['dice'].mean()
    std_dice = results_df['dice'].std()
    mean_hd = results_df['hd'].mean()
    std_hd = results_df['hd'].std()
    
    print(f"Evaluation completed!")
    print(f"Mean DSC: {mean_dice:.4f} ± {std_dice:.4f}")
    print(f"Mean HD: {mean_hd:.4f} ± {std_hd:.4f}")
    
    # Create subplots for DSC and HD
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot DSC distribution
    ax1.hist(results_df['dice'], bins=20, alpha=0.7, color='blue')
    ax1.axvline(mean_dice, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_dice:.4f}')
    ax1.set_title('DSC Distribution')
    ax1.set_xlabel('DSC Value')
    ax1.set_ylabel('Frequency')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot HD distribution
    ax2.hist(results_df['hd'], bins=20, alpha=0.7, color='green')
    ax2.axvline(mean_hd, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_hd:.4f}')
    ax2.set_title('HD Distribution')
    ax2.set_xlabel('HD Value')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save results
    if output_dir:
        print(f"Saving results to {output_dir}")
        distribution_filename = "distribution.png"
        results_filename = "results.csv"
        summary_filename = "summary.txt"
        plt.savefig(os.path.join(output_dir, distribution_filename), dpi=300, bbox_inches='tight')
        results_df.to_csv(os.path.join(output_dir, results_filename), index=False)
        
        with open(os.path.join(output_dir, summary_filename), 'w') as f:
            f.write(f"Mean DSC: {mean_dice:.4f} ± {std_dice:.4f}\n")
            f.write(f"Mean HD: {mean_hd:.4f} ± {std_hd:.4f}\n")
            f.write(f"Min DSC: {results_df['dice'].min():.4f}\n")
            f.write(f"Max DSC: {results_df['dice'].max():.4f}\n")
            f.write(f"Min HD: {results_df['hd'].min():.4f}\n")
            f.write(f"Max HD: {results_df['hd'].max():.4f}\n")
    
    plt.show()

if __name__ == "__main__":
    evaluate_segmentation()