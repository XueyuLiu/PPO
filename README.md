<div align="center">

<h1> Plug-and-Play PPO: An Adaptive Point Prompt Optimizer Making SAM Greater </h1>

</div>



<div align="center">
  <a href="https://youtu.be/LKievqcEsJA">
    <img src="Display/Video.gif" alt="Video Demo">
  </a>
</div>


##  Description
Powered by extensive curated training data, the Segment Anything Model (SAM) demonstrates impressive generalization capabilities in open-world scenarios, effectively guided by user-provided prompts.However, the class-agnostic characteristic of SAM renders its segmentation accuracy highly dependent on prompt quality. In this paper, we propose a novel Plug-and-Play dual-space Point Prompt Optimizer (PPO) designed to enhance prompt distribution through deep reinforcement learning (DRL)-based heterogeneous graph optimization. PPO optimizes initial prompts for any task without requiring additional training, thereby improving SAM’s downstream segmentation performance. Specifically, PPO constructs a dual-space heterogeneous graph, leveraging the robust feature-matching capabilities of a foundational pre-trained model to create internal feature and physical distance matrices. A DRL policy network iteratively refines the distribution of prompt points, optimizing segmentation predictions. In conclusion, PPO redefines the prompt optimization problem as a heterogeneous graph optimization task, using DRL to construct an effective, plug-and-play prompt optimizer. This approach holds potential for broader applications across diverse segmentation tasks and provides a promising solution for point prompt optimization.

## Usage 
### Setup 

- Cuda 12.0
- Python 3.9.18
- PyTorch 2.0.0


### Datasets
    ../                          # parent directory
    ├── ./data                   # data path
    │   ├── reference_image      # the one-shot reference image
    │   ├── reference_mask       # the one-shot reference mask
    │   ├── target_image         # testing images
    │   ├── initial_indices      # initial prompt indices
    │   ├── optimized_indices    # optimized prompt indices


## Setup 
- Cuda 12.0
- Python 3.9.18
- PyTorch 2.0.0

### Generate initial prompt
```
python Generate_initial_prompts.py
```

### Train GPOA
```
python train_GPOA.py
```

### Optimize initial prompt
```
python test_GPOA.py
```

### Segmentation 
```
python main_seg.py
```




## Acknowledgement
Thanks [DINOv2](https://github.com/facebookresearch/dinov2), [SAM](https://github.com/facebookresearch/segment-anything), [GBMSeg](https://github.com/SnowRain510/GBMSeg). for serving as building blocks of GPOA.
