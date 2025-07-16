import os
import torch
import numpy as np
from PIL import Image
import cv2
import networkx as nx
import pandas as pd
import random
from collections import defaultdict, deque
import time
from datetime import timedelta


def calculate_center_points(indices, size):
    """Calculate the center points based on indices for a given size."""
    center_points = []
    indices = indices.cpu().numpy()

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
    """Draw points on the image."""
    image = np.array(image)
    for point in points:
        cv2.circle(image, (point[0], point[1]), radius=size, color=color, thickness=-1)
    return image

def convert_to_edges(start_nodes, end_nodes, weights):
    """Convert nodes to edges with weights."""
    assert weights.shape == (len(start_nodes), len(end_nodes)), "Weight matrix shape mismatch"
    start_nodes_expanded = start_nodes.unsqueeze(1).expand(-1, end_nodes.size(0))
    end_nodes_expanded = end_nodes.unsqueeze(0).expand(start_nodes.size(0), -1)
    edges_with_weights_tensor = torch.stack((start_nodes_expanded, end_nodes_expanded, weights), dim=2)
    edges_with_weights = edges_with_weights_tensor.view(-1, 3).tolist()
    return edges_with_weights

def average_edge_size(graph, weight_name):
    """Calculate the average edge size based on the specified weight."""
    edges = graph.edges(data=True)
    total_weight = sum(data[weight_name] for _, _, data in edges if weight_name in data)
    edge_count = sum(1 for _, _, data in edges if weight_name in data)
    if edge_count == 0:
        return 0
    average_weight = total_weight / edge_count
    return average_weight

class GraphOptimizationEnv:
    def __init__(self, G, max_steps):
        """Initialize the graph optimization environment."""
        self.original_G = G.copy()
        self.G = G.copy()
        self.pos_nodes = [node for node, data in self.G.nodes(data=True) if data['category'] == 'pos']
        self.neg_nodes = [node for node, data in self.G.nodes(data=True) if data['category'] == 'neg']
        self.min_nodes = 5
        self.max_nodes = 20
        self.steps = 0
        self.max_steps = max_steps
        self.removed_nodes = set()
        self.reset()

        self.previous_feature_pos_mean = np.mean(list(nx.get_edge_attributes(self.original_G, 'feature_pos').values()))
        self.previous_feature_cross_mean = np.mean(list(nx.get_edge_attributes(self.original_G, 'feature_cross').values()))
        self.previous_physical_pos_mean = np.mean(list(nx.get_edge_attributes(self.original_G, 'physical_pos').values()))
        self.previous_physical_neg_mean = np.mean(list(nx.get_edge_attributes(self.original_G, 'physical_neg').values()))
        self.previous_physical_cross_mean = np.mean(list(nx.get_edge_attributes(self.original_G, 'physical_cross').values()))

        self.previous_pos_num = 0
        self.previous_neg_num = 0

    def reset(self):
        """Reset the environment."""
        self.G = self.original_G.copy()
        self.removed_nodes = set(self.G.nodes())
        self.G.clear()
        self.pos_nodes = []
        self.neg_nodes = []
        return self.get_state()

    def get_state(self):
        """Get the current state of the environment."""
        return self.G

    def step(self, action):
        """Perform an action in the environment."""
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
        self.steps += 1
        done = self.is_done()
        return self.get_state(), reward, done

    def remove_node(self, node, category):
        """Remove a node from the graph."""
        if node in self.G.nodes() and self.G.nodes[node]['category'] == category:
            self.G.remove_node(node)
            self.removed_nodes.add(node)
            if node in self.pos_nodes:
                self.pos_nodes.remove(node)
            elif node in self.neg_nodes:
                self.neg_nodes.remove(node)

    def restore_node(self, node, category):
        """Restore a node to the graph."""
        if node in self.removed_nodes and self.original_G.nodes[node]['category'] == category:
            self.G.add_node(node, **self.original_G.nodes[node])
            self.removed_nodes.remove(node)
            if self.original_G.nodes[node]['category'] == 'pos':
                self.pos_nodes.append(node)
            elif self.original_G.nodes[node]['category'] == 'neg':
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
            reward -= 10  # Add penalty for add operation

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
    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.995, memory_size=10000, batch_size=64,
                 reward_threshold=0.1):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay  # 添加 epsilon_decay 参数
        self.epsilon = epsilon_start
        self.q_table = defaultdict(float)
        self.best_pos = 100
        self.best_cross = 0  # Initialize to inf because we aim to minimize the final evaluation metric
        self.best_q_table = None
        self.memory = deque(maxlen=memory_size)
        self.best_memory = deque(maxlen=memory_size)  # 存储最佳经验的memory
        self.batch_size = batch_size
        self.reward_threshold = reward_threshold  # 添加reward_threshold属性
        self.last_reward = None  # 添加last_reward属性用于存储上一轮奖励
        self.best_reward = -float('inf')  # Initialize best reward
        self.best_reward_save = -float('inf')  # Initialize best reward
        self.best_pos_feature_distance = float('inf')  # Initialize best pos feature distance
        self.best_cross_feature_distance = float('inf')  # Initialize best pos feature distance

    def update_epsilon(self):
        """Update the epsilon value based on a decay factor."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def get_action(self, state):
        """Select an action based on epsilon-greedy policy."""
        actions = self.get_possible_actions(state)

        if random.random() < self.epsilon:
            action = random.choice(actions)
        else:
            q_values = {action: self.q_table[(state, action)] for action in actions}
            max_q = max(q_values.values())
            action = random.choice([action for action, q in q_values.items() if q == max_q])

        #print(self.epsilon, action)
        return action

    def get_possible_actions(self, state):
        """Get the possible actions for the current state."""
        actions = []

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

        #(pos_nodes_count, neg_nodes_count)

        # 只在范围之间添加 add 动作
        if self.env.min_nodes < pos_nodes_count < self.env.max_nodes and self.env.min_nodes < neg_nodes_count < self.env.max_nodes:
            actions.extend(restore_pos_actions)
            actions.extend(restore_neg_actions)
            actions.extend(remove_pos_actions)
            actions.extend(remove_neg_actions)
            actions.extend([(node, "add") for node in range(state.number_of_nodes(), state.number_of_nodes() + 10)])
        else:
            # 超出范围时，只允许 restore 和 remove 操作
            if pos_nodes_count <= self.env.min_nodes:
                actions.extend(restore_pos_actions)
            elif pos_nodes_count >= self.env.max_nodes:
                actions.extend(remove_pos_actions)

            if neg_nodes_count <= self.env.min_nodes:
                actions.extend(restore_neg_actions)
            elif neg_nodes_count >= self.env.max_nodes:
                actions.extend(remove_neg_actions)

        # 保证 remove_pos 和 remove_neg 的概率一致
        if len(remove_pos_actions) < len(remove_neg_actions):
            remove_neg_actions = random.sample(remove_neg_actions, len(remove_pos_actions))
        elif len(remove_neg_actions) < len(remove_pos_actions):
            remove_pos_actions = random.sample(remove_pos_actions, len(remove_neg_actions))

        actions.extend(remove_pos_actions)
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

    def train(self, episodes, output_path, base_dir, file_prefixes, max_steps):
        """Train the Q-learning agent."""
        rewards = []
        image_size = 560
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        os.makedirs(output_path, exist_ok=True)
        start_time = time.time()
        for episode in range(episodes):
            selected_prefix = random.choice(list(file_prefixes))
            print(f"Episode {episode + 1}/{episodes}, Selected file prefix: {selected_prefix}")

            feature_file = os.path.join(base_dir, f"{selected_prefix}.png_features.pt")
            pos_file = os.path.join(base_dir, f"{selected_prefix}.png_initial_indices_pos.pt")
            neg_file = os.path.join(base_dir, f"{selected_prefix}.png_initial_indices_neg.pt")

            if not (os.path.exists(feature_file) and os.path.exists(pos_file) and os.path.exists(neg_file)):
                print(f"Required files not found for prefix {selected_prefix}, skipping this episode.")
                continue

            features = torch.load(feature_file).to(device)
            positive_indices = torch.load(pos_file).to(device)
            negative_indices = torch.load(neg_file).to(device)

            positive_indices = torch.unique(positive_indices).to(device)
            negative_indices = torch.unique(negative_indices).to(device)

            set1 = set(positive_indices.tolist())
            set2 = set(negative_indices.tolist())

            intersection = set1.intersection(set2)
            positive_indices = torch.tensor([x for x in positive_indices.cpu().tolist() if x not in intersection]).cuda()
            negative_indices = torch.tensor([x for x in negative_indices.cpu().tolist() if x not in intersection]).cuda()
            if positive_indices.numel() == 0 or negative_indices.numel() == 0:
                continue
            feature_pos_distances, feature_cross_distances, physical_pos_distances, physical_neg_distances, physical_cross_distances = calculate_distances(
                features, positive_indices, negative_indices, image_size, device)

            feature_pos_edge = convert_to_edges(positive_indices, positive_indices, feature_pos_distances)
            physical_pos_edge = convert_to_edges(positive_indices, positive_indices, physical_pos_distances)
            physical_neg_edge = convert_to_edges(negative_indices, negative_indices, physical_neg_distances)
            feature_cross_edge = convert_to_edges(positive_indices, negative_indices, feature_cross_distances)
            physical_cross_edge = convert_to_edges(positive_indices, negative_indices, physical_cross_distances)

            G = nx.MultiGraph()
            G.add_nodes_from(positive_indices.cpu().numpy(), category='pos')
            G.add_nodes_from(negative_indices.cpu().numpy(), category='neg')

            G.add_weighted_edges_from(feature_pos_edge, weight='feature_pos')
            G.add_weighted_edges_from(physical_pos_edge, weight='physical_pos')
            G.add_weighted_edges_from(physical_neg_edge, weight='physical_neg')
            G.add_weighted_edges_from(feature_cross_edge, weight='feature_cross')
            G.add_weighted_edges_from(physical_cross_edge, weight='physical_cross')

            self.env = GraphOptimizationEnv(G, max_steps)
            state = self.env.reset()
            done = False
            total_reward = 0

            normalized_reward = (self.best_reward - 0) / (5 - 0)

            #print(normalize_distances())

            if self.best_reward<0:
                self.epsilon=self.epsilon_start
            elif self.best_reward>=5:
                self.epsilon=self.epsilon_end
            else:
                self.epsilon = 1-normalized_reward  # 每个 epoch 开始时重置 epsilon
            print(self.epsilon)
            while not done:
                action = self.get_action(state)
                next_state, reward, done = self.env.step(action)
                self.memory.append((state, action, reward, next_state))
                self.update_q_table(state, action, reward, next_state)
                self.replay()
                state = next_state
                total_reward += reward
                self.update_epsilon()  # Update epsilon based on decay factor

            # 使用最佳表现的经验进行回放

            print(total_reward,self.best_reward)

            if total_reward > self.best_reward:

                self.best_reward = total_reward
                self.best_memory = deque(self.memory, maxlen=self.memory.maxlen)  # 更新最佳经验的memory
                self.replay_best()

            rewards.append(total_reward)
            self.save_results(episode, total_reward, output_path, selected_prefix)
            self.last_reward = total_reward  # 更新last_reward

            # Calculate final evaluation metric
            mean_feature_pos = np.mean(list(nx.get_edge_attributes(self.env.G, 'feature_pos').values()))
            mean_feature_cross = np.mean(list(nx.get_edge_attributes(self.env.G, 'feature_cross').values()))
            print(f"Episode {episode + 1}/{episodes}, Reward: {total_reward}, Final pos: {mean_feature_pos}, Final cross: {mean_feature_cross}")

            elapsed_time = time.time() - start_time
            estimated_total_time = elapsed_time * (episodes / (episode + 1))
            remaining_time = estimated_total_time - elapsed_time
            print(f"Elapsed Time: {timedelta(seconds=int(elapsed_time))}, Estimated Total Time: {timedelta(seconds=int(estimated_total_time))}, Remaining Time: {timedelta(seconds=int(remaining_time))}")

            if mean_feature_pos < self.best_pos and mean_feature_cross > self.best_cross:  # Assuming lower is better for evaluation metric
                print('Update!')
                self.best_pos = mean_feature_pos
                self.best_cross = mean_feature_cross
                self.best_q_table = self.q_table.copy()
                self.save_best_q_table(output_path)

            # 保存奖励最大的模型
            if total_reward > self.best_reward_save:
                self.best_reward_save = total_reward
                self.save_best_model(output_path, 'best_reward_model.pkl')

            # 保存pos特征距离最小的模型
            if mean_feature_pos < self.best_pos_feature_distance:
                self.best_pos_feature_distance = mean_feature_pos
                self.save_best_model(output_path, 'best_pos_feature_distance_model.pkl')

            if mean_feature_cross > self.best_cross_feature_distance:
                self.best_cross_feature_distance = mean_feature_cross
                self.save_best_model(output_path, 'best_cross_feature_distance_model.pkl')

            # 输出保留的正点和负点数量
            final_pos_count = len(self.env.pos_nodes)
            final_neg_count = len(self.env.neg_nodes)
            print(f"Episode {episode + 1}: Final positive nodes count: {final_pos_count}, Final negative nodes count: {final_neg_count}")

        return rewards

    def save_results(self, episode, reward, output_path, prefix):
        """Save the results of an episode."""
        G_state = self.env.get_state()
        pos_nodes = [node for node, data in G_state.nodes(data=True) if data['category'] == 'pos']
        neg_nodes = [node for node, data in G_state.nodes(data=True) if data['category'] == 'neg']

        with open(f"{output_path}/{prefix}_rewards.txt", "a") as f:
            f.write(f"Episode {episode}: Reward: {reward}\n")

    def save_best_q_table(self, output_path):
        """Save the best Q-table."""
        with open(f"{output_path}/best_q_table.pkl", "wb") as f:
            torch.save(self.best_q_table, f)

    def save_best_model(self, output_path, filename):
        """Save the best model based on the highest reward."""
        with open(f"{output_path}/{filename}", "wb") as f:
            torch.save(self.q_table, f)
        print(f"Best model saved with reward: {self.best_reward}")

def show_mask(mask,ax, random_color=False):

    color = np.array([50/255, 120/255, 255/255, 0.8])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def sample_points(mask_path, new_size=(224, 224), num_positive=10, num_negative=10):
    """随机采样正点和负点坐标"""
    mask = Image.open(mask_path).convert("L").resize(new_size)
    mask_array = np.array(mask)

    # 获取白色（正点）和黑色（负点）的坐标
    positive_points = np.column_stack(np.where(mask_array == 255))
    negative_points = np.column_stack(np.where(mask_array == 0))

    # 如果正点或负点的数量不足指定的数量，使用实际数量
    num_positive = min(num_positive, len(positive_points))
    num_negative = min(num_negative, len(negative_points))

    # 随机采样正点和负点
    sampled_positive_points = positive_points[np.random.choice(len(positive_points), num_positive, replace=False)]
    sampled_negative_points = negative_points[np.random.choice(len(negative_points), num_negative, replace=False)]

    positive_indices = calculate_block_index(sampled_positive_points,560)
    negative_indices =calculate_block_index(sampled_negative_points,560)

    return mask_array, positive_indices, negative_indices


def calculate_block_index(center_points, size, block_size=14):
    """根据中心点坐标计算块索引"""
    indices = []
    for (y, x) in center_points:
        row = y // block_size
        col = x // block_size
        index = row * (size // block_size) + col
        indices.append(index)
    return indices
