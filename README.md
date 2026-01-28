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
> **🚀 Exciting News:** This work has been accepted by **CVPR 2025**!
>
> **🔥 Update:** We have updated the model weights!
> 📥 **Download:** [**The model weights**](https://drive.google.com/file/d/1elAw4iagw4TYHsD0zWjJZnn9vJUHcZ0H/view?usp=sharing)

## 📝 Description

Powered by extensive curated training data, the **Segment Anything Model (SAM)** demonstrates impressive generalization capabilities in open-world scenarios, effectively guided by user-provided prompts. However, the class-agnostic characteristic of SAM renders its segmentation accuracy highly dependent on prompt quality.

In this paper, we propose a novel **Plug-and-Play dual-space Point Prompt Optimizer (PPO)** designed to enhance prompt distribution through **Deep Reinforcement Learning (DRL)**-based heterogeneous graph optimization. PPO optimizes initial prompts for any task without requiring additional training, thereby improving SAM’s downstream segmentation performance. Specifically, PPO constructs a dual-space heterogeneous graph, leveraging the robust feature-matching capabilities of a foundational pre-trained model to create internal feature and physical distance matrices. A DRL policy network iteratively refines the distribution of prompt points, optimizing segmentation predictions.

In conclusion, PPO redefines the prompt optimization problem as a heterogeneous graph optimization task, using DRL to construct an effective, plug-and-play prompt optimizer. This approach holds potential for broader applications across diverse segmentation tasks and provides a promising solution for point prompt optimization.

## 🛠️ Usage

### Setup
Ensure you have **CUDA 12.7** and **Python 3.9.20** installed.

```bash
# Install dependencies
pip install -r requirements.txt
```

### Datasets
Please organize your data directory as follows:

```text
../                            # parent directory
└── data/                      # data path
    ├── reference_image/       # the one-shot reference image
    ├── reference_mask/        # the one-shot reference mask
    ├── target_image/          # testing images
    ├── initial_indices/       # initial prompt indices
    └── optimized_indices/     # optimized prompt indices
```

### 🚀 Run PPO

You can run the full pipeline using the following commands:

```bash
# 1. Generate initial prompt
python generate_initial_prompts.py

# 2. Train PPO
python train_PPO.py

# 3. Optimize PPO with feature matching
python main_FM.py
```

## 📊 Results

### Optimization results for different datasets
<div align="center">
  <img width="100%" alt="Optimization Results" src="Display/Results.png">
</div>

## 💐 Acknowledgement

We explicitly thank the authors of the following repositories for their valuable contributions, which served as building blocks for PPO:

* [**DINOv2**](https://github.com/facebookresearch/dinov2)
* [**Segment Anything (SAM)**](https://github.com/facebookresearch/segment-anything)
* [**GBMSeg**](https://github.com/SnowRain510/GBMSeg)
