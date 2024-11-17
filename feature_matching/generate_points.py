import os
import sys
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm
from sklearn.cluster import KMeans
from . import hubconf

def find_foreground_patches(mask_np, size):
    """
    Find the foreground and background patches in the mask.

    Parameters:
    mask_np (numpy array): The mask image.
    size (int): The size of the image.

    Returns:
    list: Foreground patches.
    numpy array: Foreground indices.
    list: Background patches.
    numpy array: Background indices.
    """
    fore_patchs = []
    back_patchs = []

    for i in range(0, mask_np.shape[0] - 14, 14):
        for j in range(0, mask_np.shape[1] - 14, 14):
            if np.all(mask_np[i:i + 14, j:j + 14] != [0, 0, 0]):
                fore_patchs.append((i, j))
            if np.all(mask_np[i:i + 14, j:j + 14] == [0, 0, 0]):
                back_patchs.append((i, j))

    fore_index = np.empty((len(fore_patchs), 1), dtype=int)
    back_index = np.empty((len(back_patchs), 1), dtype=int)

    for i in range(len(fore_patchs)):
        row = fore_patchs[i][0] / 14
        col = fore_patchs[i][1] / 14
        fore_index[i] = row * (size / 14) + col

    for i in range(len(back_patchs)):
        row = back_patchs[i][0] / 14
        col = back_patchs[i][1] / 14
        back_index[i] = row * (size / 14) + col

    return fore_patchs, fore_index, back_patchs, back_index

def calculate_center_points(indices, size):
    """
    Calculate the center points of each patch.

    Parameters:
    indices (numpy array): The indices of the patches.
    size (int): The size of the image.

    Returns:
    list: Center points of the patches.
    """
    center_points = []
    indices = indices.cpu().numpy()

    for i in range(len(indices)):
        row_index = indices[i] // (size / 14)
        col_index = indices[i] % (size / 14)
        center_x = col_index * 14 + 14 // 2
        center_y = row_index * 14 + 14 // 2
        center_points.append([center_x, center_y])

    return center_points

def map_to_ori_size(resized_coordinates, original_size, size):
    """
    Map the coordinates back to the original image size.

    Parameters:
    resized_coordinates (list or tuple): The resized coordinates.
    original_size (tuple): The original size of the image.
    size (int): The size of the image.

    Returns:
    list or tuple: The coordinates mapped back to the original size.
    """
    original_height, original_width = original_size
    scale_height = original_height / size
    scale_width = original_width / size

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

def convert_to_rgb(image):
    """
    Convert an image to RGB format if it is in RGBA.

    Parameters:
    image (PIL.Image): The input image.

    Returns:
    PIL.Image: The converted image.
    """
    if image.mode == 'RGBA':
        return image.convert('RGB')
    return image

def forward_matching(images_inner, index, device, dino, size):
    """
    Perform forward matching to get features and indices.

    Parameters:
    images_inner (list of PIL.Image): The list of images.
    index (list): The list of indices.
    device (torch.device): The device to run the model on.
    dino (model): The DINO model.
    size (int): The size of the image.

    Returns:
    list of PIL.Image: The list of images.
    torch.Tensor: The features.
    torch.Tensor: The minimum indices.
    """
    transform = T.Compose([
        T.Resize(size),
        T.CenterCrop(size),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    imgs_tensor = torch.stack([transform(convert_to_rgb(img))[:3] for img in images_inner]).to(device)
    with torch.no_grad():
        features_dict = dino.forward_features(imgs_tensor)
        features = features_dict['x_norm_patchtokens']
    fore_index = torch.tensor(index)
    fore_index  = fore_index.long()
    distances = torch.cdist(features[0][fore_index].squeeze(1), features[1])
    min_values, min_indices = distances.min(dim=1)

    return images_inner, features, min_indices

def loading_dino(device):
    """
    Load the DINO model.

    Parameters:
    device (torch.device): The device to load the model on.

    Returns:
    model: The DINO model.
    """
    dino = hubconf.dinov2_vitg14()
    dino.to(device)
    return dino

def distance_calculate(features, indices_pos, indices_back, size):
    """
    Calculate distances between features and physical points.

    Parameters:
    features (torch.Tensor): The features.
    indices_pos (torch.Tensor): The positive indices.
    indices_back (torch.Tensor): The background indices.
    size (int): The size of the image.

    Returns:
    tuple: Distances between features and physical points.
    """
    final_pos_points = torch.tensor(calculate_center_points(indices_pos, size))
    final_neg_points = torch.tensor(calculate_center_points(indices_back, size))

    feature_pos_distances = torch.cdist(features[1][indices_pos], features[1][indices_pos])
    feature_cross_distances = torch.cdist(features[1][indices_pos], features[1][indices_back])
    physical_pos_distances = torch.cdist(final_pos_points, final_pos_points)
    physical_cross_distances = torch.cdist(final_pos_points, final_neg_points)

    return feature_pos_distances, feature_cross_distances, physical_pos_distances, physical_cross_distances

def points_generate(indices_pos, indices_neg, size, images_inner):
    """
    Generate points and map them back to the original size.

    Parameters:
    indices_pos (torch.Tensor): The positive indices.
    indices_neg (torch.Tensor): The negative indices.
    size (int): The size of the image.
    images_inner (list of PIL.Image): The list of images.

    Returns:
    tuple: The mapped positive and negative points.
    """
    final_pos_points = calculate_center_points(indices_pos, size)
    final_neg_points = calculate_center_points(indices_neg, size)

    final_pos_points = set(tuple(point) for point in final_pos_points)
    final_neg_points = set(tuple(point) for point in final_neg_points)
    image = images_inner[1]
    final_pos_points_map = map_to_ori_size(list(final_pos_points), [image.size[1], image.size[0]], size)
    final_neg_points_map = map_to_ori_size(list(final_neg_points), [image.size[1], image.size[0]], size)

    return final_pos_points_map, final_neg_points_map

def generate(mask, image_inner, device, dino, size):
    """
    Generate initial prompting scheme.

    Parameters:
    mask (PIL.Image): The mask image.
    image_inner (list of PIL.Image): The list of images.
    device (torch.device): The device to run the model on.
    dino (model): The DINO model.
    size (int): The size of the image.

    Returns:
    tuple: Features, initial positive indices, initial background indices.
    """
    mask = np.array(mask)
    mask_np = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    fore_patchs, fore_index, back_patchs, back_index = find_foreground_patches(mask_np, size)

    images_inner, features, initial_indices = forward_matching(image_inner, fore_index, device, dino, size)
    images_inner_back, features_back, initial_indices_back = forward_matching(image_inner, back_index, device, dino, size)

    return features, initial_indices, initial_indices_back
