import os
import torch
import numpy as np
import networkx as nx
import random
from collections import deque
import time
from datetime import timedelta, datetime
from utils import GraphOptimizationEnv, QLearningAgent,calculate_distances,convert_to_edges,get_box_node_indices
import json
import matplotlib.pyplot as plt

# Constants
SIZE = 560
DATASET = ''
CATAGORY = ''
BASE_DIR = os.path.dirname(__file__)

def train_agent(agent, episodes, output_path, base_dir, file_prefixes, max_steps):
    """Train Q-learning agent
    Args:
        agent: QLearningAgent instance
        episodes: Number of training episodes
        output_path: Output directory path
        base_dir: Training data directory
        file_prefixes: List of file prefixes
        max_steps: Maximum steps per episode
    Returns:
        rewards: List of rewards per episode
    """
    rewards = []
    image_size = SIZE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_path, exist_ok=True)
    start_time = time.time()

    # Training loop
    for episode in range(episodes):
        # Randomly select a file prefix
        selected_prefix = random.choice(list(file_prefixes))
        print(f"Episode {episode + 1}/{episodes}, Selected file prefix: {selected_prefix}")
        
        # Build full paths for feature and index files
        feature_file = os.path.join(base_dir, f"{selected_prefix}_features.pt")
        pos_file = os.path.join(base_dir, f"{selected_prefix}_initial_indices_pos.pt")
        neg_file = os.path.join(base_dir, f"{selected_prefix}_initial_indices_neg.pt")
        bbox_file = os.path.join(base_dir, f"{selected_prefix}_bbox.json") 
        
        # Check if required files exist
        if not (os.path.exists(feature_file) and os.path.exists(pos_file) and os.path.exists(neg_file) and os.path.exists(bbox_file)):
            print(f"Required files not found for prefix {selected_prefix}, skipping this episode.")
            continue
        
        # Load feature and index data
        features = torch.load(feature_file, weights_only=True).to(device)
        positive_indices = torch.load(pos_file, weights_only=True).to(device)
        negative_indices = torch.load(neg_file, weights_only=True).to(device)
        with open(bbox_file, 'r') as f:
            bbox_data = json.load(f)
            
        positive_indices = torch.unique(positive_indices).to(device)
        negative_indices = torch.unique(negative_indices).to(device)
        
        # Remove intersecting indices
        set1 = set(positive_indices.tolist())
        set2 = set(negative_indices.tolist())
        intersection = set1.intersection(set2)
        if intersection:
            positive_indices = torch.tensor([x for x in positive_indices.cpu().tolist() if x not in intersection]).cuda()
            negative_indices = torch.tensor([x for x in negative_indices.cpu().tolist() if x not in intersection]).cuda()
            
        if positive_indices.numel() == 0 or negative_indices.numel() == 0:
            continue

        print(f"Positive indices: {positive_indices.shape}, Negative indices: {negative_indices.shape}")

        # Calculate distances
        feature_pos_distances, feature_cross_distances, physical_pos_distances, physical_neg_distances, physical_cross_distances = calculate_distances(
            features, positive_indices, negative_indices, image_size, device)

        # Convert to edge representation
        feature_pos_edge = convert_to_edges(positive_indices, positive_indices, feature_pos_distances)
        physical_pos_edge = convert_to_edges(positive_indices, positive_indices, physical_pos_distances)
        physical_neg_edge = convert_to_edges(negative_indices, negative_indices, physical_neg_distances)
        feature_cross_edge = convert_to_edges(positive_indices, negative_indices, feature_cross_distances)
        physical_cross_edge = convert_to_edges(positive_indices, negative_indices, physical_cross_distances)

        # Create graph structure
        G = nx.MultiGraph()
        G.add_nodes_from(positive_indices.cpu().numpy(), category='pos')
        G.add_nodes_from(negative_indices.cpu().numpy(), category='neg')
        G.add_weighted_edges_from(feature_pos_edge, weight='feature_pos')
        G.add_weighted_edges_from(physical_pos_edge, weight='physical_pos')
        G.add_weighted_edges_from(physical_neg_edge, weight='physical_neg')
        G.add_weighted_edges_from(feature_cross_edge, weight='feature_cross')
        G.add_weighted_edges_from(physical_cross_edge, weight='physical_cross')

        # Get node indices
        (inside_indices, outside_indices,
        inside_pos_indices, outside_pos_indices,
        inside_neg_indices, outside_neg_indices) = get_box_node_indices(G, bbox_data)
        
        print(f"Inside bbox - Pos: {len(inside_pos_indices)}, Neg: {len(inside_neg_indices)}")
        print(f"Outside bbox - Pos: {len(outside_pos_indices)}, Neg: {len(outside_neg_indices)}")

        # Initialize environment
        agent.env = GraphOptimizationEnv(G, max_steps)
        state = agent.env.reset()
        done = False
        total_reward = 0

        # Adjust epsilon based on best reward
        normalized_reward = (agent.best_reward - 0) / (5 - 0)
        if agent.best_reward < 0:
            agent.epsilon = agent.epsilon_start
        elif agent.best_reward >= 5:
            agent.epsilon = agent.epsilon_end
        else:
            agent.epsilon = 1 - normalized_reward
        print(f"best_reward:{agent.best_reward},epsilon:{agent.epsilon}")

        # Episode loop
        while not done:
            action = agent.get_action(state)
            next_state, reward, done = agent.env.step(action)
            agent.memory.append((state, action, reward, next_state))
            agent.update_q_table(state, action, reward, next_state)
            agent.replay()
            state = next_state
            total_reward += reward
            agent.update_epsilon()

        # Update best record if current episode performs better
        print(total_reward, agent.best_reward)
        if total_reward > agent.best_reward:
            agent.best_reward = total_reward
            agent.best_memory = deque(agent.memory, maxlen=agent.memory.maxlen)
            agent.replay_best()

        # Save results
        rewards.append(total_reward)
        save_results(agent, episode, total_reward, output_path, selected_prefix)
        agent.last_reward = total_reward

        # Calculate final metrics
        mean_feature_pos = np.mean(list(nx.get_edge_attributes(agent.env.G, 'feature_pos').values()))
        mean_feature_cross = np.mean(list(nx.get_edge_attributes(agent.env.G, 'feature_cross').values()))
        print(f"Episode {episode + 1}/{episodes}, Reward: {total_reward}, Final pos: {mean_feature_pos}, Final cross: {mean_feature_cross}")

        # Calculate and display time information
        elapsed_time = time.time() - start_time
        estimated_total_time = elapsed_time * (episodes / (episode + 1))
        remaining_time = estimated_total_time - elapsed_time
        print(f"Elapsed Time: {timedelta(seconds=int(elapsed_time))}, Estimated Total Time: {timedelta(seconds=int(estimated_total_time))}, Remaining Time: {timedelta(seconds=int(remaining_time))}")

        # Update and save best model
        if mean_feature_pos < agent.best_pos and mean_feature_cross > agent.best_cross:
            print('Update!')
            agent.best_pos = mean_feature_pos
            agent.best_cross = mean_feature_cross
            agent.best_q_table = agent.q_table.copy()
            save_best_q_table(agent, output_path)

        # Save models for different metrics
        if total_reward > agent.best_reward_save:
            agent.best_reward_save = total_reward
            save_best_model(agent, output_path, 'best_reward_model.pkl')

        if mean_feature_pos < agent.best_pos_feature_distance:
            agent.best_pos_feature_distance = mean_feature_pos
            save_best_model(agent, output_path, 'best_pos_feature_distance_model.pkl')

        if mean_feature_cross > agent.best_cross_feature_distance:
            agent.best_cross_feature_distance = mean_feature_cross
            save_best_model(agent, output_path, 'best_cross_feature_distance_model.pkl')

        # Output final node statistics
        final_pos_count = len(agent.env.pos_nodes)
        final_neg_count = len(agent.env.neg_nodes)
        print(f"Episode {episode + 1}: Final positive nodes count: {final_pos_count}, Final negative nodes count: {final_neg_count}")

    return rewards

def save_results(agent, episode, reward, output_path, prefix):
    """Save the results of an episode in txt."""
    G_state = agent.env.get_state()
    pos_nodes = [node for node, data in G_state.nodes(data=True) if data['category'] == 'pos']
    neg_nodes = [node for node, data in G_state.nodes(data=True) if data['category'] == 'neg']

    with open(f"{output_path}/{prefix}_rewards.txt", "a") as f:
        f.write(f"Episode {episode}: Reward: {reward}\n")

def save_best_q_table(agent, output_path):
    """Save the best Q-table."""
    with open(f"{output_path}/best_q_table.pkl", "wb") as f:
        torch.save(agent.best_q_table, f)

def save_best_model(agent, output_path, filename):
    """Save the best model."""
    with open(f"{output_path}/{filename}", "wb") as f:
        torch.save(agent.q_table, f)
    print(f"Best model saved with reward: {agent.best_reward}")

def main():
    """Main function to train the Q-learning agent."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_size = SIZE

    # Set initial prompts data directory
    base_dir = os.path.join(BASE_DIR, 'results', DATASET, CATAGORY, 'initial_prompts')
    
    files = [f for f in os.listdir(base_dir) if os.path.isfile(os.path.join(base_dir, f))]
    # Extract file prefixes (first three underscore-separated parts)
    file_prefixes = set('_'.join(f.split('_')[:3]) for f in files)
    
    # Training parameters
    max_steps = 100
    env = GraphOptimizationEnv
    agent = QLearningAgent(env)
    
    # Create output directory with timestamp
    current_time = datetime.now().strftime("%Y%m%d_%H%M")
    output_path = os.path.join(os.path.dirname(__file__), 'train', current_time)
    
    # Train agent and plot rewards
    rewards = train_agent(agent, episodes=30, output_path=output_path, base_dir=base_dir, file_prefixes=file_prefixes, max_steps=max_steps)
    
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training Rewards Over Episodes')
    plt.savefig(f"{output_path}/rewards_plot.png")
    plt.show()

if __name__ == "__main__":
    main()
