import os
import numpy as np
import torch
import cv2
from segment_anything import sam_model_registry, SamPredictor
from tqdm import tqdm
import matplotlib.pyplot as plt

def show_mask(mask, ax, random_color=False):
    """Display mask on the image."""
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([254 / 255, 215 / 255, 26 / 255, 0.8])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=300):
    """Display points on the image."""
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='#40e0d0', marker='o', s=marker_size,edgecolor='white',linewidth=1)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='pink', marker='o', s=marker_size,edgecolor='white',linewidth=1)

def prepare_input(ps_points, ng_points):
    """Prepare input points and labels for the model."""
    if ps_points is not None and ng_points is not None:
        ps_points = np.array(ps_points)
        ng_points = np.array(ng_points)
        input_point = np.vstack((ps_points, ng_points))
        ps_label = np.ones(ps_points.shape[0])
        ng_label = np.zeros(ng_points.shape[0])
        input_label = np.concatenate((ps_label, ng_label))
    else:
        ps_points = np.array(ps_points)
        input_point = ps_points
        input_label = np.ones(ps_points.shape[0])
    return input_point, input_label

def save_max_contour_area(mask):
    """Find and save the largest contour in the mask."""
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    filled_image = np.zeros_like(mask)
    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(filled_image, [max_contour], -1, 255, thickness=cv2.FILLED)
    return filled_image

def refine_mask(mask):
    """Refine the mask by keeping only the largest contours."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    largest_contour = contours[0]
    min_area = 0.3 * cv2.contourArea(largest_contour)
    filtered_contours = [contour for contour in contours if cv2.contourArea(contour) >= min_area]
    contour_mask = np.zeros_like(mask)
    cv2.drawContours(contour_mask, [largest_contour], -1, 255, -1)
    cv2.drawContours(contour_mask, filtered_contours, -1, 255, -1)
    return contour_mask

def process_image(image, ps_points, ng_points, sam, max_contour=False):
    """Process a single image to generate a mask."""
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    predictor = SamPredictor(sam)
    predictor.set_image(image)
    input_point, input_label = prepare_input(ps_points, ng_points)
    #print(input_point)
    #print(input_label)
    masks, _, _ = predictor.predict(point_coords=input_point, point_labels=input_label, multimask_output=False)
    mask_image = (masks[0] * 255).astype(np.uint8)
    if max_contour:
        mask_image = save_max_contour_area(mask_image)
    return mask_image

def loading_seg(model_type, device):
    """Load the segmentation model."""
    if model_type == 'vitb':
        sam_checkpoint = "./segmenter/checkpoint/sam_vit_b_01ec64.pth"
        model_type = 'vit_b'
    elif model_type == 'vitl':
        sam_checkpoint = "./segmenter/checkpoint/sam_vit_l_0b3195.pth"
        model_type = 'vit_l'
    elif model_type == 'vith':
        sam_checkpoint = "./segmenter/checkpoint/sam_vit_h_4b8939.pth"
        model_type = 'vit_h'
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    return sam

def seg_main(image, pos_prompt, neg_prompt, device, sam_model, max_contour=False):
    """Main segmentation function."""
    mask = process_image(image, pos_prompt, neg_prompt, sam_model, max_contour)
    return mask
