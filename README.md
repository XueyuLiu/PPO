<div align="center">

<h1> GPOA: Enhancing Point Prompt Distribution via Reinforcement Learning-based Heterogeneous Graph Optimization for Segment Anything </h1>

</div>



<div align="center">
  <a href="https://youtu.be/LKievqcEsJA">
    <img src="Display/Video.gif" alt="Video Demo">
  </a>
</div>


##  Description
Powered by extensive curated training data, the Segment Anything Model (SAM) has demonstrated impressive generalization capabilities in open-world scenarios with the guidance of prompts. However, SAM is class-agnostic and heavily relies on user-provided prompts to segment objects of interest, making the quality of segmentation results highly dependent on the quality of these prompts. We propose a novel Graph-based Prompt Optimization Agent (GPOA), designed to enhance point prompt distribution via deep reinforcement learning (DRL)-based heterogeneous graph optimization. Anchored by an agent, GPOA enables the generality of the SAM model across diverse downstream tasks with a training-free paradigm. Specifically, GPOA constructs a dual-space heterogeneous graph by leveraging the robust feature matching capabilities of the foundational pre-trained model, creating an internal feature and a physical distance matrix. It then employs a DRL policy network to iteratively refine the distribution of prompt points, optimizing segmentation predictions. In conclusion, GPOA effectively enhances SAM's segmentation performance through optimized prompt distributions, demonstrating potential for broader application in various segmentation tasks, and providing a promising solution for any point prompt optimization.

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
