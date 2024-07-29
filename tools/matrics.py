import os
import numpy as np
from skimage.io import imread
from skimage.metrics import adapted_rand_error, variation_of_information
from skimage.transform import resize
from scipy.spatial.distance import directed_hausdorff
from sklearn.metrics import jaccard_score
from tqdm import tqdm


def dice_coefficient(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection) / (np.sum(y_true) + np.sum(y_pred))


def mean_iou(y_true, y_pred):
    intersection = np.sum((y_true * y_pred), axis=(1, 2))
    union = np.sum(y_true, axis=(1, 2)) + np.sum(y_pred, axis=(1, 2)) - intersection
    return np.mean(intersection / union)


def hausdorff_distance(y_true, y_pred):
    return max(directed_hausdorff(y_true, y_pred)[0], directed_hausdorff(y_pred, y_true)[0])


# Define the directories
true_folder = '/home/shiguangze/code/GBMSeg_v2/data/ISIC_test/masks'
pred_folder = '/home/shiguangze/code/hsnet-main/datasets/test/ISIC2018/1shot_1'

resize_shape = (560, 560)  # Desired resize shape

# List all files in the directories
#true_files = sorted([f for f in os.listdir(true_folder) if os.path.isfile(os.path.join(true_folder, f))])
pred_files = sorted([f for f in os.listdir(pred_folder) if os.path.isfile(os.path.join(pred_folder, f))])

dice_scores = []
iou_scores = []
hausdorff_distances = []

# Iterate over all files
for pred_file in tqdm(pred_files):

    # Read the images
    if os.path.exists(os.path.join(pred_folder, pred_file)):
        #print(os.path.join(true_folder, pred_file[:-4]+'.jpg'))
        y_true = imread(os.path.join(true_folder, pred_file[:-4]+'.jpg'), as_gray=True)
        y_pred = imread(os.path.join(pred_folder, pred_file), as_gray=True)

        # Resize images to the same shape
        y_true_resized = resize(y_true, resize_shape, anti_aliasing=True)
        y_pred_resized = resize(y_pred, resize_shape, anti_aliasing=True)

        # Ensure the images are binary
        y_true_resized = (y_true_resized > 0.5).astype(np.int32)
        y_pred_resized = (y_pred_resized > 0.5).astype(np.int32)

        # Calculate DICE coefficient
        dice = dice_coefficient(y_true_resized, y_pred_resized)
        #print(true_file)
        #print(pred_file)
        #print(dice)
        #if dice>0.50:
        dice_scores.append(dice)
        iou = jaccard_score(y_true_resized.flatten(), y_pred_resized.flatten())
        iou_scores.append(iou)
        hausdorff = hausdorff_distance(y_true_resized, y_pred_resized)
        hausdorff_distances.append(hausdorff)

    #else:
        #print(dice)

    # Calculate mean IoU


    # Calculate Hausdorff distance
    #hausdorff = hausdorff_distance(y_true_resized, y_pred_resized)
    #hausdorff_distances.append(hausdorff)

# Print the results
print(f"Average DICE Coefficient: {np.mean(dice_scores):.4f}")
print(f"Average mIoU: {np.mean(iou_scores):.4f}")
print(f"Average Hausdorff Distance: {np.mean(hausdorff_distances):.4f}")