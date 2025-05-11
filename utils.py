import torch
import numpy as np
from PIL import Image
import cv2
import networkx as nx
import random
from collections import defaultdict, deque
from config import *

def calculate_center_points(indices, size):
    """Calculate the center points based on indices for a given size."""
    center_points = []
    
    # Convert indices to numpy array depending on input type
    if hasattr(indices, 'cpu'):  # Check if indices is a torch tensor
        indices = indices.cpu().numpy()
    elif isinstance(indices, list):
        indices = np.array(indices)
    else:
        indices = np.asarray(indices)

    for index in indices:
        row = index // (size // 14)
        col = index % (size // 14)
        center_x = col * 14 + 14 // 2
        center_y = row * 14 + 14 // 2
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
        original_coordinates = [[round(x * scale_width), round(y * scale_height)] for x, y in resized_coordinates]
        return original_coordinates
    else:
        raise ValueError("Unsupported input format. Please provide a tuple or list of coordinates.")

def normalize_distances(distances):
    """Normalize the distances to be between 0 and 1."""
    max_distance = torch.max(distances)
    min_distance = torch.min(distances)
    normalized_distances = (distances - min_distance) / (max_distance - min_distance)
    return normalized_distances


def refine_mask(mask,threshold):

    # Find contours in the mask image
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours by area in descending order
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Find the largest contour
    largest_contour = contours[0]

    # Calculate the minimum contour area that is 20% of the size of the largest contour
    min_area = threshold * cv2.contourArea(largest_contour)

    # Find contours that are at least 20% of the size of the largest contour
    filtered_contours = [contour for contour in contours if cv2.contourArea(contour) >= min_area]

    # Draw the contours on the resized mask image
    contour_mask = np.zeros_like(mask)
    cv2.drawContours(contour_mask, [largest_contour], -1, 255, -1)
    cv2.drawContours(contour_mask, filtered_contours, -1, 255, -1)

    return contour_mask

def generate_points(positive_indices, negative_indices, image_size):
    """Generate positive and negative points mapped to original size."""
    positive_points = calculate_center_points(positive_indices, image_size)
    negative_points = calculate_center_points(negative_indices, image_size)

    unique_positive_points = set(tuple(point) for point in positive_points)
    unique_negative_points = set(tuple(point) for point in negative_points)

    mapped_positive_points = map_to_original_size(list(unique_positive_points), [560, 560], image_size)
    mapped_negative_points = map_to_original_size(list(unique_negative_points), [560, 560], image_size)

    return mapped_positive_points, mapped_negative_points

def calculate_distances(features, positive_indices, negative_indices, image_size, device):
    """Calculate feature and physical distances."""
    positive_points = torch.tensor(calculate_center_points(positive_indices, image_size), dtype=torch.float).to(device)
    negative_points = torch.tensor(calculate_center_points(negative_indices, image_size), dtype=torch.float).to(device)

    features = features.to(device)

    feature_positive_distances = torch.cdist(features[1][positive_indices], features[1][positive_indices])
    feature_cross_distances = torch.cdist(features[1][positive_indices], features[1][negative_indices])

    physical_positive_distances = torch.cdist(positive_points, positive_points)
    physical_negative_distances = torch.cdist(negative_points, negative_points)
    physical_cross_distances = torch.cdist(positive_points, negative_points)

    feature_positive_distances = normalize_distances(feature_positive_distances)
    feature_cross_distances = normalize_distances(feature_cross_distances)
    physical_positive_distances = normalize_distances(physical_positive_distances)
    physical_negative_distances = normalize_distances(physical_negative_distances)
    physical_cross_distances = normalize_distances(physical_cross_distances)

    return feature_positive_distances, feature_cross_distances, physical_positive_distances, physical_negative_distances, physical_cross_distances

def draw_points_on_image(image, points, color, size):
    """Draw points on the image with specified color and size"""
    image = np.array(image)
    for point in points:
        cv2.circle(image, (point[0], point[1]), radius=size, color=color, thickness=-1)
    return image

def convert_to_edges(start_nodes, end_nodes, weights):
    """Convert node pairs to edges with corresponding weights"""
    assert weights.shape == (len(start_nodes), len(end_nodes)), "Weight matrix shape mismatch"
    start_nodes_expanded = start_nodes.unsqueeze(1).expand(-1, end_nodes.size(0))
    end_nodes_expanded = end_nodes.unsqueeze(0).expand(start_nodes.size(0), -1)
    edges_with_weights_tensor = torch.stack((start_nodes_expanded, end_nodes_expanded, weights), dim=2)
    edges_with_weights = edges_with_weights_tensor.view(-1, 3).tolist()
    return edges_with_weights

def average_edge_size(graph, weight_name):
    """Calculate average edge weight for specified weight type"""
    edges = graph.edges(data=True)
    total_weight = sum(data[weight_name] for _, _, data in edges if weight_name in data)
    edge_count = sum(1 for _, _, data in edges if weight_name in data)
    if edge_count == 0:
        return 0
    average_weight = total_weight / edge_count
    return average_weight

class GraphOptimizationEnv:
    def __init__(self, G, max_steps):
        """Initialize graph optimization environment
        Args:
            G: Input graph
            max_steps: Maximum steps allowed
        """
        self.original_G = G.copy()
        self.G = G.copy()
        self.pos_nodes = [node for node, data in self.G.nodes(data=True) if data['category'] == 'pos']
        self.neg_nodes = [node for node, data in self.G.nodes(data=True) if data['category'] == 'neg']
        self.min_nodes = 5  # Minimum nodes required
        self.max_nodes = 20 # Maximum nodes allowed
        self.steps = 0
        self.max_steps = max_steps
        self.removed_nodes = set()
        self.reset()

        # Store initial feature and physical distances
        self.previous_feature_pos_mean = np.mean(list(nx.get_edge_attributes(self.original_G, 'feature_pos').values()))
        self.previous_feature_cross_mean = np.mean(list(nx.get_edge_attributes(self.original_G, 'feature_cross').values()))
        self.previous_physical_pos_mean = np.mean(list(nx.get_edge_attributes(self.original_G, 'physical_pos').values()))
        self.previous_physical_neg_mean = np.mean(list(nx.get_edge_attributes(self.original_G, 'physical_neg').values()))
        self.previous_physical_cross_mean = np.mean(list(nx.get_edge_attributes(self.original_G, 'physical_cross').values()))

        self.previous_pos_num = 0
        self.previous_neg_num = 0

    def reset(self):
        """Reset environment to initial state"""
        self.G = self.original_G.copy()
        self.removed_nodes = set(self.G.nodes())
        self.G.clear()
        self.pos_nodes = []
        self.neg_nodes = []
        return self.get_state()

    def get_state(self):
        """Return current graph state"""
        return self.G

    def step(self, action):
        """Execute action and return new state, reward and completion status
        Args:
            action: Action to take (add/remove node)
        Returns:
            (new_state, reward, done)
        """
        node, operation = action
        if operation == "remove_pos":
            self.remove_node(node, "pos")
        elif operation == "remove_neg":
            self.remove_node(node, "neg")
        elif operation == "restore_pos":
            self.restore_node(node, "pos")
        elif operation == "restore_neg":
            self.restore_node(node, "neg")
        elif operation == "add":
            self.add_node(node)

        reward = self.calculate_reward(operation)
        if self.min_nodes < len(self.pos_nodes) < self.max_nodes and self.min_nodes < len(self.neg_nodes) < self.max_nodes:
            if reward < 0:
                self.revertStep(action)
                reward = 0
            elif reward > 0:
                self.steps += 1
        done = self.is_done()
        print(reward)
        return self.get_state(), reward, done
    
    def revertStep(self, action):
        node, operation = action
        if operation == "remove_pos":
            self.restore_node(node, "pos")
        elif operation == "remove_neg":
            self.restore_node(node, "neg")
        elif operation == "restore_pos":
            self.remove_node(node, "pos")
        elif operation == "restore_neg":
            self.remove_node(node, "neg")

    def remove_node(self, node, category):
        """Remove node from graph and update node lists"""
        if node in self.G.nodes() and self.G.nodes[node]['category'] == category:
            self.G.remove_node(node)
            self.removed_nodes.add(node)
            if node in self.pos_nodes:
                self.pos_nodes.remove(node)
            elif node in self.neg_nodes:
                self.neg_nodes.append(node)

            # Restore edges associated with this node
            for neighbor in self.original_G.neighbors(node):
                if neighbor in self.G.nodes():
                    for edge in self.original_G.edges(node, data=True):
                        if edge[1] == neighbor:
                            self.G.add_edge(edge[0], edge[1], **edge[2])

    def add_node(self, node):
        """Add a new node to the graph."""
        category = 'pos' if random.random() < 0.5 else 'neg'
        self.G.add_node(node, category=category)
        self.original_G.add_node(node, category=category)
        if category == 'pos':
            self.pos_nodes.append(node)
        else:
            self.neg_nodes.append(node)

    def calculate_reward(self, operation):
        """Calculate the reward based on the current state and operation."""
        feature_pos_distances = nx.get_edge_attributes(self.G, 'feature_pos')
        feature_cross_distances = nx.get_edge_attributes(self.G, 'feature_cross')
        physical_pos_distances = nx.get_edge_attributes(self.G, 'physical_pos')
        physical_neg_distances = nx.get_edge_attributes(self.G, 'physical_neg')
        physical_cross_distances = nx.get_edge_attributes(self.G, 'physical_cross')

        mean_feature_pos = np.mean(list(feature_pos_distances.values())) if feature_pos_distances else 0
        mean_feature_cross = np.mean(list(feature_cross_distances.values())) if feature_cross_distances else 0
        mean_physical_pos = np.mean(list(physical_pos_distances.values())) if physical_pos_distances else 0
        mean_physical_neg = np.mean(list(physical_neg_distances.values())) if physical_neg_distances else 0
        mean_physical_cross = np.mean(list(physical_cross_distances.values())) if physical_cross_distances else 0

        reward = 0

        if mean_feature_pos < self.previous_feature_pos_mean:
            reward += 5 * (self.previous_feature_pos_mean - mean_feature_pos)
        else:
            reward -= 5 * (mean_feature_pos - self.previous_feature_pos_mean)

        if mean_feature_cross > self.previous_feature_cross_mean:
            reward += 5 * (mean_feature_cross - self.previous_feature_cross_mean)
        else:
            reward -= 5 * (self.previous_feature_cross_mean - mean_feature_cross)

        if mean_physical_pos > self.previous_physical_pos_mean:
            reward += (mean_physical_pos - self.previous_physical_pos_mean)
        else:
            reward -= (self.previous_physical_pos_mean - mean_physical_pos)

        if mean_physical_neg > self.previous_physical_neg_mean:
            reward += (mean_physical_neg - self.previous_physical_neg_mean)
        else:
            reward -= (self.previous_physical_neg_mean - mean_physical_neg)

        if mean_physical_cross < self.previous_physical_cross_mean:
            reward += (self.previous_physical_cross_mean - mean_physical_cross)
        else:
            reward -= (mean_physical_cross - self.previous_physical_cross_mean)

        if operation == "add":
            if (mean_feature_pos < self.previous_feature_pos_mean and 
                mean_feature_cross > self.previous_feature_cross_mean):
                reward -= 3
            else:
                reward -= 5

        self.previous_pos_num = len(self.pos_nodes)
        self.previous_neg_num = len(self.neg_nodes)
        self.previous_feature_cross_mean = mean_feature_cross
        self.previous_feature_pos_mean = mean_feature_pos
        self.previous_physical_pos_mean = mean_physical_pos
        self.previous_physical_neg_mean = mean_physical_neg
        self.previous_physical_cross_mean = mean_physical_cross

        return reward

    def is_done(self):
        """Check if the maximum steps have been reached."""
        return self.steps >= self.max_steps

class QLearningAgent:
    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.995, memory_size=10000, batch_size=64,reward_threshold=0.1):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.epsilon = epsilon_start
        self.q_table = defaultdict(float)
        self.best_pos = 100
        self.best_cross = 0
        self.best_q_table = None
        self.memory = deque(maxlen=memory_size)
        self.best_memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
        self.reward_threshold = reward_threshold
        self.last_reward = None
        self.best_reward = -float('inf')
        self.best_reward_save = -float('inf')
        self.best_pos_feature_distance = float('inf')
        self.best_cross_feature_distance = float('inf')

    def update_epsilon(self):
        """Update the epsilon value based on a decay factor."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def get_action(self, state):
        """Select an action based on epsilon-greedy policy."""
        actions = self.get_possible_actions(state)

        if random.random() < self.epsilon:
            action = random.choice(actions)
        else:
            # Select the action with the highest Q-value
            q_values = {action: self.q_table[(state, action)] for action in actions}
            max_q = max(q_values.values())
            action = random.choice([action for action, q in q_values.items() if q == max_q])

        return action

    def get_possible_actions(self, state):
        """Get the possible actions for the current state."""
        actions = []
        
        # Get restore and remove actions for positive and negative nodes
        restore_pos_actions = [
            (node, "restore_pos") 
            for node in self.env.removed_nodes
            if node in self.env.original_G and self.env.original_G.nodes[node].get('category') == 'pos'
        ]
        restore_neg_actions = [
            (node, "restore_neg")
            for node in self.env.removed_nodes 
            if node in self.env.original_G and self.env.original_G.nodes[node].get('category') == 'neg'
        ]
        remove_pos_actions = [
            (node, "remove_pos")
            for node in state.nodes()
            if node in self.env.original_G and self.env.original_G.nodes[node].get('category') == 'pos' 
        ]
        remove_neg_actions = [
            (node, "remove_neg")
            for node in state.nodes()
            if node in self.env.original_G and self.env.original_G.nodes[node].get('category') == 'neg'
        ]

        pos_nodes_count = len(self.env.pos_nodes)
        neg_nodes_count = len(self.env.neg_nodes)

        # Only add actions when node counts are within bounds
        if self.env.min_nodes < pos_nodes_count < self.env.max_nodes and self.env.min_nodes < neg_nodes_count < self.env.max_nodes:
            actions.extend(restore_pos_actions)
            actions.extend(restore_neg_actions)
            actions.extend(remove_pos_actions)
            actions.extend(remove_neg_actions)
        else:
            # When out of bounds, only allow restore/remove to get back in bounds
            if pos_nodes_count <= self.env.min_nodes:
                actions.extend(restore_pos_actions)
            elif pos_nodes_count >= self.env.max_nodes:
                actions.extend(remove_pos_actions)

            if neg_nodes_count <= self.env.min_nodes:
                actions.extend(restore_neg_actions)
            elif neg_nodes_count >= self.env.max_nodes:
                actions.extend(remove_neg_actions)

        return actions

    def update_q_table(self, state, action, reward, next_state):
        """Update the Q-table based on the current state, action, reward, and next state."""
        max_next_q = max(
            [self.q_table[(next_state, next_action)] for next_action in self.get_possible_actions(next_state)],
            default=0)
        self.q_table[(state, action)] += self.alpha * (reward + self.gamma * max_next_q - self.q_table[(state, action)])

    def replay(self):
        """Replay experiences from memory to update Q-table."""
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state in batch:
            self.update_q_table(state, action, reward, next_state)

    def replay_best(self):
        """Replay best experiences from memory to update Q-table."""
        if len(self.best_memory) < self.batch_size:
            return

        batch = random.sample(self.best_memory, self.batch_size)
        for state, action, reward, next_state in batch:
            self.update_q_table(state, action, reward, next_state)

def show_mask(mask,ax, random_color=False):

    color = np.array([50/255, 120/255, 255/255, 0.8])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def sample_points(mask_path, new_size=(224, 224), num_positive=10, num_negative=10):
    """Randomly sample positive and negative point coordinates"""
    mask = Image.open(mask_path).convert("L").resize(new_size)
    mask_array = np.array(mask)

    # Get coordinates of white (positive) and black (negative) points
    positive_points = np.column_stack(np.where(mask_array == 255))
    negative_points = np.column_stack(np.where(mask_array == 0))

    # If the number of positive or negative points is less than the specified number, use the actual number
    num_positive = min(num_positive, len(positive_points))
    num_negative = min(num_negative, len(negative_points))

    # Randomly sample positive and negative points
    sampled_positive_points = positive_points[np.random.choice(len(positive_points), num_positive, replace=False)]
    sampled_negative_points = negative_points[np.random.choice(len(negative_points), num_negative, replace=False)]

    positive_indices = calculate_block_index(sampled_positive_points,560)
    negative_indices =calculate_block_index(sampled_negative_points,560)

    return mask_array, positive_indices, negative_indices


def calculate_block_index(center_points, size, block_size=14):
    """Calculate block index based on center point coordinates"""
    indices = []
    for (y, x) in center_points:
        row = y // block_size
        col = x // block_size
        index = row * (size // block_size) + col
        indices.append(index)
    return indices

def is_point_in_box(point, bbox):
    """
    Check if a point is inside a bounding box
    
    Args:
        point (list): Point coordinates in [x, y] format
        bbox (dict): Bounding box information containing min_x, min_y, max_x, max_y
    
    Returns:
        bool: Whether the point is inside the bounding box
    """
    x, y = point
    return (bbox['min_x'] <= x <= bbox['max_x'] and 
            bbox['min_y'] <= y <= bbox['max_y'])

def get_box_node_indices(G, bbox):
    """
    Get node indices inside and outside the bounding box
    
    Args:
        G (networkx.Graph): Graph structure
        bbox (dict): Bounding box information containing min_x, min_y, max_x, max_y
    
    Returns:
        tuple: (inside_indices, outside_indices, 
                inside_pos_indices, outside_pos_indices,
                inside_neg_indices, outside_neg_indices)
    """
    inside_indices = []
    outside_indices = []
    inside_pos_indices = []
    outside_pos_indices = []
    inside_neg_indices = []
    outside_neg_indices = []
    
    for node in G.nodes():
        # Calculate the center point coordinates of the node
        point = calculate_center_points([node], 560)[0]
        
        # Check if the point is inside the bounding box
        if is_point_in_box(point, bbox):
            inside_indices.append(node)
            if G.nodes[node]['category'] == 'pos':
                inside_pos_indices.append(node)
            else:
                inside_neg_indices.append(node)
        else:
            outside_indices.append(node)
            if G.nodes[node]['category'] == 'pos':
                outside_pos_indices.append(node)
            else:
                outside_neg_indices.append(node)
    
    return (inside_indices, outside_indices,
            inside_pos_indices, outside_pos_indices,
            inside_neg_indices, outside_neg_indices)
