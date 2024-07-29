import os
import sys
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
import cv2
import warnings
from feature_matching.generate_points import generate, loading_dino, distance_calculate
from segmenter.segment import loading_seg

# Ignore all warnings
warnings.filterwarnings("ignore")

# Set paths for feature matching and segmentation modules
generate_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'feature_matching'))
segment_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'segmenter'))
sys.path.append(segment_path)
sys.path.append(generate_path)

# Function to draw points on an image
def draw_points_on_image(image, points, color):
    """
    Draws a list of points on an image.

    Parameters:
    image (np.array): The image on which to draw the points.
    points (list of tuples): The list of points to draw.
    color (tuple): The color of the points in BGR format.
    """
    image = np.array(image)
    for point in points:
        cv2.circle(image, (point[0], point[1]), radius=5, color=color, thickness=-1)
    return image

# Function to save a PyTorch tensor to a text file
def save_tensor_to_txt(tensor, filename):
    """
    Saves a PyTorch tensor to a text file.

    Parameters:
    tensor (torch.Tensor): The tensor to save.
    filename (str): The path to the text file.
    """
    array = tensor.cpu().numpy()
    np.savetxt(filename, array, fmt='%d')
    print(f"Tensor saved to {filename}")

def main():
    # Hyperparameter setting
    dataset_name = ''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    image_size = 560  # Must be a multiple of 14

    # Loading the DINO model
    model_dino = loading_dino(device)

    # Define directories
    image_prompt_dir = ''
    mask_path = ''
    image_dir = ''
    save_dir = ''

    os.makedirs(save_dir, exist_ok=True)

    # Get the reference image for prompting
    reference_list = os.listdir(image_prompt_dir)
    reference = reference_list[0]

    # Load the reference image and ground truth mask
    image_prompt = Image.open(os.path.join(image_prompt_dir, reference)).resize((image_size, image_size))
    gt_mask = Image.open(os.path.join(mask_path, reference)).resize((image_size, image_size))

    imglist = os.listdir(image_dir)
    dice_list = []

    for name in tqdm(imglist):
        image_path = os.path.join(image_dir, name)
        image = Image.open(image_path).resize((image_size, image_size))

        image_inner = [image_prompt, image]
        features, initial_indices_pos, initial_indices_neg = generate(gt_mask, image_inner, device, model_dino, image_size)

        if len(initial_indices_pos) != 0 and len(initial_indices_neg) != 0:
            feature_pos_distances, feature_cross_distances, physical_pos_distances, physical_cross_distances = distance_calculate(features, initial_indices_pos, initial_indices_neg, image_size)

            torch.save(features, os.path.join(save_dir, name + '_features.pt'))
            torch.save(initial_indices_pos, os.path.join(save_dir, name + '_initial_indices_pos.pt'))
            torch.save(initial_indices_neg, os.path.join(save_dir, name + '_initial_indices_neg.pt'))
        else:
            print(f"No positive or negative indices found for {name}")

if __name__ == "__main__":
    main()
