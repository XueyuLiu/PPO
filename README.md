<div align="center">

<h1>Plug-and-Play PPO: An Adaptive Point Prompt Optimizer Making SAM Greater</h1>

[![CVPR 2025](https://img.shields.io/badge/CVPR-2025-b31b1b.svg)](https://cvpr.thecvf.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)

<a href="https://youtu.be/LKievqcEsJA">
  <img src="Display/Video.gif" alt="Video Demo" width="80%">
</a>

</div>

## 📢 News

> [!IMPORTANT]
> **Exciting News:** This work has been accepted by **CVPR 2025**! 🚀
>
> **Update:** We have officially released the pre-trained model weights.
> 📥 **Download:** [**The model weights**](https://drive.google.com/file/d/1elAw4iagw4TYHsD0zWjJZnn9vJUHcZ0H/view?usp=sharing)

## 📝 Description

Powered by extensive curated training data, the **Segment Anything Model (SAM)** demonstrates impressive generalization capabilities in open-world scenarios, effectively guided by user-provided prompts. However, the class-agnostic characteristic of SAM renders its segmentation accuracy highly dependent on prompt quality.

In this paper, we propose a novel **Plug-and-Play dual-space Point Prompt Optimizer (PPO)** designed to enhance prompt distribution through **Deep Reinforcement Learning (DRL)**-based heterogeneous graph optimization. PPO optimizes initial prompts for any task without requiring additional training, thereby improving SAM’s downstream segmentation performance.

**Key Features:**
- **Dual-Space Heterogeneous Graph:** Leverages robust feature-matching capabilities of foundation models to create internal feature and physical distance matrices.
- **DRL-Based Optimization:** A policy network iteratively refines the distribution of prompt points.
- **Plug-and-Play:** Optimizes segmentation predictions for diverse tasks without re-training SAM.

## 🛠️ Usage

### Setup
Ensure you have **CUDA 12.7** and **Python 3.9.20** installed.

```bash
pip install -r requirements.txt
