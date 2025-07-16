import os
import sys
import time
import warnings
import torch
from tqdm import tqdm
from PIL import Image

# Set paths for feature matching and segmentation modules
generate_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'feature_matching'))
segment_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'segmenter'))
sys.path.append(segment_path)
sys.path.append(generate_path)

# from segment_anything import sam_model_registry, SamPredictor
from segmenter.segment import loading_seg, seg_main
from feature_matching.generate_points import generate, loading_dino
from test_PPO import optimize_nodes
from utils_test import generate_points, GraphOptimizationEnv, QLearningAgent

# Ignore all warnings
warnings.filterwarnings("ignore")

SIZE = 560

DATASET = ''
CATAGORY = ''

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = SIZE

# Define paths
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, 'dataset', DATASET, CATAGORY)
REFERENCE_IMAGE_DIR = os.path.join(DATA_DIR, 'reference_images')
MASK_DIR = os.path.join(DATA_DIR, 'reference_masks')
Q_TABLE_PATH = os.path.join(BASE_DIR, 'model', 'best_q_table.pkl')
IMAGE_DIR = os.path.join(DATA_DIR, 'target_images')  
RESULTS_DIR = os.path.join(BASE_DIR, 'results', DATASET, CATAGORY)
SAVE_DIR = os.path.join(RESULTS_DIR, 'masks')

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
            print(f"len(opt_pos_indices): {len(opt_pos_indices)}, len(opt_neg_indices): {len(opt_neg_indices)}")
            print(f"Time to optimize prompts: {end_time - start_time:.4f} seconds")

            # Generate points and perform segmentation
            pos_points, neg_points = generate_points(opt_pos_indices, opt_neg_indices, IMAGE_SIZE)
            mask = seg_main(image, pos_points, neg_points, DEVICE, model_seg)

            # Save the resulting segmentation mask
            mask = Image.fromarray(mask)
            mask.save(os.path.join(SAVE_DIR, f"{image_name}"))
        else:
            print(f"Skipping {image_name}: No positive or negative indices found.")
    except Exception as e:
        print(f"Error processing {image_name}: {e}")

# Main function
if __name__ == "__main__":
    # Load models
    model_seg, model_dino = load_models()

    # Initialize Q-learning agent
    env = GraphOptimizationEnv
    agent = QLearningAgent(env)
    agent.q_table = torch.load(Q_TABLE_PATH,weights_only=False)

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
