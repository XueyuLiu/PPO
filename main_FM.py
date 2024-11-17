import os
import sys
import time
import warnings
import torch
from tqdm import tqdm
from PIL import Image
from segment_anything import sam_model_registry, SamPredictor
from segmenter.segment import process_image, loading_seg, seg_main
from feature_matching.generate_points import generate, loading_dino, distance_calculate
from test_GPOA import test_agent, optimize_nodes
from utils import generate_points, GraphOptimizationEnv, QLearningAgent, calculate_distances, convert_to_edges

# Ignore all warnings
warnings.filterwarnings("ignore")

# Set device (GPU if available, otherwise CPU)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = SIZE

# Define paths
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, 'data', 'DATA_NAME')
REFERENCE_IMAGE_DIR = os.path.join(DATA_DIR, 'references_images')
MASK_DIR = os.path.join(DATA_DIR, 'references_masks')
Q_TABLE_PATH = os.path.join(BASE_DIR, 'MODEL_NAME')
IMAGE_DIR = os.path.join(DATA_DIR, 'images')  # Path for test images
SAVE_DIR = os.path.join(BASE_DIR, 'results')

# Ensure the results directory exists
os.makedirs(SAVE_DIR, exist_ok=True)

# Load models for segmentation and feature generation
def load_models():
    """
    Load the segmentation model and DINO feature extractor.
    """
    try:
        model_seg = loading_seg('vitl', DEVICE)
        model_dino = loading_dino(DEVICE)
        return model_seg, model_dino
    except Exception as e:
        print(f"Error loading models: {e}")
        sys.exit(1)

# Process a single image
def process_single_image(agent, model_dino, model_seg, image_name, reference, mask_dir):
    """
    Process a single image for segmentation and optimization.

    Parameters:
    - agent: Q-learning agent for optimization
    - model_dino: DINO feature extraction model
    - model_seg: SAM model
    - image_name: Name of the image to process
    - reference: Reference image for feature comparison
    - mask_dir: Directory containing ground truth masks
    """
    try:
        # Load input image and reference data
        image_path = os.path.join(IMAGE_DIR, image_name)
        image = Image.open(image_path).resize((IMAGE_SIZE, IMAGE_SIZE))
        reference_image = Image.open(os.path.join(REFERENCE_IMAGE_DIR, reference)).resize((IMAGE_SIZE, IMAGE_SIZE))
        gt_mask = Image.open(os.path.join(mask_dir, reference)).resize((IMAGE_SIZE, IMAGE_SIZE))

        # Generate features and initial positive/negative prompts
        image_inner = [reference_image, image]
        start_time = time.time()
        features, pos_indices, neg_indices = generate(gt_mask, image_inner, DEVICE, model_dino, IMAGE_SIZE)
        end_time = time.time()
        print(f"Time to generate initial prompts: {end_time - start_time:.4f} seconds")

        if len(pos_indices) != 0 and len(neg_indices) != 0:
            # Optimize prompts using Q-learning
            start_time = time.time()
            opt_pos_indices, opt_neg_indices = optimize_nodes(
                agent, pos_indices, neg_indices, features, max_steps=100, device=DEVICE, image_size=IMAGE_SIZE
            )
            end_time = time.time()
            print(f"Time to optimize prompts: {end_time - start_time:.4f} seconds")

            # Generate points and perform segmentation
            pos_points, neg_points = generate_points(opt_pos_indices, opt_neg_indices, IMAGE_SIZE)
            mask = seg_main(image, pos_points, neg_points, DEVICE, model_seg)

            # Save the resulting segmentation mask
            mask = Image.fromarray(mask)
            mask.save(os.path.join(SAVE_DIR, f"{image_name}_mask.png"))
        else:
            print(f"Skipping {image_name}: No positive or negative indices found.")
    except Exception as e:
        print(f"Error processing {image_name}: {e}")

# Main function
if __name__ == "__main__":
    # Load models
    model_seg, model_dino = load_models()

    # Initialize Q-learning agent
    env = None  # Placeholder for the environment
    agent = QLearningAgent(env)
    agent.q_table = torch.load(Q_TABLE_PATH)

    # Get reference image list
    reference_list = os.listdir(REFERENCE_IMAGE_DIR)
    if not reference_list:
        print("No reference images found.")
        sys.exit(1)

    # Use the first reference image
    reference = reference_list[0]

    # Process all images in the test directory
    img_list = os.listdir(IMAGE_DIR)
    for img_name in tqdm(img_list, desc="Processing images"):
        process_single_image(agent, model_dino, model_seg, img_name, reference, MASK_DIR)

    print("Processing complete!")
