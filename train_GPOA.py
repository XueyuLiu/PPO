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
from utils import GraphOptimizationEnv, QLearningAgent



def main():
    """Main function to train the Q-learning agent."""
    global device, image_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_size = SIZE

    base_dir = ''
    files = [f for f in os.listdir(base_dir) if os.path.isfile(os.path.join(base_dir, f))]
    file_prefixes = set('_'.join(f.split('_')[:2]) for f in files)
    max_steps = 100
    env = None
    agent = QLearningAgent(env)
    output_path = ''
    rewards = agent.train(episodes=100000, output_path=output_path, base_dir=base_dir, file_prefixes=file_prefixes, max_steps=max_steps)

    import matplotlib.pyplot as plt
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training Rewards Over Episodes')
    plt.savefig(f"{output_path}/rewards_plot.png")
    plt.show()

if __name__ == "__main__":
    main()
