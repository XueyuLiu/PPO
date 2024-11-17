import os
import torch
import numpy as np
from PIL import Image
import cv2
import networkx as nx
import random
from collections import defaultdict, deque
import time
from datetime import timedelta


# === Utility Functions ===
def calculate_center_points(indices, size):
    """Calculate the center points of blocks based on their indices."""
    center_points = []
    indices = indices.cpu().numpy()

    for index in indices:
        row = index // (size // 14)
        col = index % (size // 14)
        center_x = col * 14 + 7  # Center x-coordinate
        center_y = row * 14 + 7  # Center y-coordinate
        center_points.append([center_x, center_y])

    return center_points


def map_to_original_size(resized_coordinates, original_size, image_size):
    """Map resized coordinates back to the original image size."""
    original_height, original_width = original_size
    scale_height = original_height / image_size
    scale_width = original_width / image_size

    if isinstance(resized_coordinates, tuple):
        resized_x, resized_y = resized_coordinates
        original_x = resized_x * scale_width
        original_y = resized_y * scale_height
        return original_x, original_y
    elif isinstance(resized_coordinates, list):
        return [[round(x * scale_width), round(y * scale_height)] for x, y in resized_coordinates]
    else:
        raise ValueError("Unsupported input format. Expected a tuple or list of coordinates.")


def normalize_distances(distances):
    """Normalize distances to a range of 0 to 1."""
    max_distance = torch.max(distances)
    min_distance = torch.min(distances)
    return (distances - min_distance) / (max_distance - min_distance)


def refine_mask(mask, threshold):
    """Refine the mask by filtering contours based on area size."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    if not contours:
        return np.zeros_like(mask)

    largest_contour = contours[0]
    min_area = threshold * cv2.contourArea(largest_contour)

    filtered_contours = [contour for contour in contours if cv2.contourArea(contour) >= min_area]
    contour_mask = np.zeros_like(mask)
    cv2.drawContours(contour_mask, filtered_contours, -1, 255, -1)

    return contour_mask


def generate_points(positive_indices, negative_indices, image_size):
    """Generate unique positive and negative points mapped to original size."""
    positive_points = calculate_center_points(positive_indices, image_size)
    negative_points = calculate_center_points(negative_indices, image_size)

    unique_positive_points = set(tuple(point) for point in positive_points)
    unique_negative_points = set(tuple(point) for point in negative_points)

    mapped_positive_points = map_to_original_size(list(unique_positive_points), [560, 560], image_size)
    mapped_negative_points = map_to_original_size(list(unique_negative_points), [560, 560], image_size)

    return mapped_positive_points, mapped_negative_points


def calculate_distances(features, positive_indices, negative_indices, image_size, device):
    """Calculate feature and physical distances between nodes."""
    positive_points = torch.tensor(calculate_center_points(positive_indices, image_size), dtype=torch.float).to(device)
    negative_points = torch.tensor(calculate_center_points(negative_indices, image_size), dtype=torch.float).to(device)
    features = features.to(device)

    feature_positive_distances = torch.cdist(features[1][positive_indices], features[1][positive_indices])
    feature_cross_distances = torch.cdist(features[1][positive_indices], features[1][negative_indices])

    physical_positive_distances = torch.cdist(positive_points, positive_points)
    physical_negative_distances = torch.cdist(negative_points, negative_points)
    physical_cross_distances = torch.cdist(positive_points, negative_points)

    return (
        normalize_distances(feature_positive_distances),
        normalize_distances(feature_cross_distances),
        normalize_distances(physical_positive_distances),
        normalize_distances(physical_negative_distances),
        normalize_distances(physical_cross_distances),
    )


def draw_points_on_image(image, points, color, size):
    """Draw points of a given size and color on an image."""
    image = np.array(image)
    for point in points:
        cv2.circle(image, (point[0], point[1]), radius=size, color=color, thickness=-1)
    return image


def convert_to_edges(start_nodes, end_nodes, weights):
    """Convert nodes into edges with weights."""
    assert weights.shape == (len(start_nodes), len(end_nodes)), "Weight matrix shape mismatch"
    start_nodes_expanded = start_nodes.unsqueeze(1).expand(-1, end_nodes.size(0))
    end_nodes_expanded = end_nodes.unsqueeze(0).expand(start_nodes.size(0), -1)
    edges_with_weights_tensor = torch.stack((start_nodes_expanded, end_nodes_expanded, weights), dim=2)
    return edges_with_weights_tensor.view(-1, 3).tolist()


def average_edge_size(graph, weight_name):
    """Calculate the average edge size for a given weight."""
    edges = graph.edges(data=True)
    total_weight = sum(data[weight_name] for _, _, data in edges if weight_name in data)
    edge_count = sum(1 for _, _, data in edges if weight_name in data)
    return total_weight / edge_count if edge_count > 0 else 0


def sample_points(mask_path, new_size=(224, 224), num_positive=10, num_negative=10):
    """Randomly sample positive and negative points from a mask."""
    mask = Image.open(mask_path).convert("L").resize(new_size)
    mask_array = np.array(mask)

    # Extract positive (white) and negative (black) points
    positive_points = np.column_stack(np.where(mask_array == 255))
    negative_points = np.column_stack(np.where(mask_array == 0))

    num_positive = min(num_positive, len(positive_points))
    num_negative = min(num_negative, len(negative_points))

    sampled_positive_points = positive_points[np.random.choice(len(positive_points), num_positive, replace=False)]
    sampled_negative_points = negative_points[np.random.choice(len(negative_points), num_negative, replace=False)]

    positive_indices = calculate_block_index(sampled_positive_points, 560)
    negative_indices = calculate_block_index(sampled_negative_points, 560)

    return mask_array, positive_indices, negative_indices


def calculate_block_index(center_points, size, block_size=14):
    """Calculate block indices based on center point coordinates."""
    indices = []
    for (y, x) in center_points:
        row = y // block_size
        col = x // block_size
        index = row * (size // block_size) + col
        indices.append(index)
    return indices
