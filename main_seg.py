import os
import sys
import torch
import cv2
import numpy as np
import warnings
from tqdm import tqdm
from segment_anything import SamPredictor
from segmenter.segment import loading_seg, seg_main

# Ignore all warnings
warnings.filterwarnings("ignore")

# Set paths
segment_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'segmenter'))
segment_path_1 = os.path.abspath(os.path.join(os.path.dirname(__file__), 'segmenter', 'segment_anything'))
sys.path.append(segment_path)
sys.path.append(segment_path_1)

# Map resized coordinates to the original image size
def map_to_original_size(resized_coordinates, original_size, image_size):
    original_height, original_width = original_size
    scale_height = original_height / image_size
    scale_width = original_width / image_size

    if isinstance(resized_coordinates, tuple):
        resized_x, resized_y = resized_coordinates
        original_x = resized_x * scale_width
        original_y = resized_y * scale_height
        return original_x, original_y
    elif isinstance(resized_coordinates, list):
        original_coordinates = [[round(x * scale_width), round(y * scale_height)] for x, y in resized_coordinates]
        return original_coordinates
    else:
        raise ValueError("Unsupported input format. Please provide a tuple or list of coordinates.")

# Calculate center points
def calculate_center_points(indices, size):
    center_points = []
    indices = indices.cpu().numpy()

    for index in indices:
        row = index // (size / 14)
        col = index % (size / 14)
        center_x = col * 14 + 14 // 2
        center_y = row * 14 + 14 // 2
        center_points.append([center_x, center_y])

    return center_points

# Refine the mask
def refine_mask(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    largest_contour = contours[0]
    min_area = 0.6 * cv2.contourArea(largest_contour)
    filtered_contours = [contour for contour in contours if cv2.contourArea(contour) >= min_area]
    contour_mask = np.zeros_like(mask)
    cv2.drawContours(contour_mask, [largest_contour], -1, 255, -1)
    cv2.drawContours(contour_mask, filtered_contours, -1, 255, -1)
    return contour_mask

# Generate positive and negative sample points
def generate_points(positive_indices, negative_indices, image_size):
    positive_points = calculate_center_points(positive_indices, image_size)
    negative_points = calculate_center_points(negative_indices, image_size)

    unique_positive_points = set(tuple(point) for point in positive_points)
    unique_negative_points = set(tuple(point) for point in negative_points)

    mapped_positive_points = map_to_original_size(list(unique_positive_points), [560, 560], image_size)
    mapped_negative_points = map_to_original_size(list(unique_negative_points), [560, 560], image_size)

    return mapped_positive_points, mapped_negative_points

if __name__ == "__main__":
    image_size = 560

    # Define paths
    image_path = ''
    save_path = ''
    prompt_path = ''

    os.makedirs(save_path, exist_ok=True)

    for name in tqdm(os.listdir(image_path)):
        device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
        model_seg = loading_seg('vith', device)
        image = cv2.imread(os.path.join(image_path, name))
        image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if os.path.exists(os.path.join(prompt_path, name + '_initial_indices_pos.pt')):
            positive_indices = torch.load(os.path.join(prompt_path, name + '_initial_indices_pos.pt')).to(device)
            negative_indices = torch.load(os.path.join(prompt_path, name + '_initial_indices_neg.pt')).to(device)
            positive_points, negative_points = generate_points(positive_indices, negative_indices, image_size)
            if len(positive_points) != 0 and len(negative_points) != 0:
                mask = seg_main(image, positive_points, negative_points, device, model_seg)
                mask = refine_mask(mask)
                cv2.imwrite(os.path.join(save_path, name), mask)
