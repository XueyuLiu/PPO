import os
import torch
import torchvision.transforms as T
import hubconf
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Load the largest dino model
dino = hubconf.dinov2_vitg14()
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")  # 单GPU或者CPU
dino.to(device)

# Load the images
img_dir = '/Data20T/data20t/data20t/envs/shigz1/code/dinov2-main/dinov2-main/images/predict'
image_files = os.listdir(img_dir)

# Preprocess & convert to tensor
transform = T.Compose([
    T.Resize(560, interpolation=T.InterpolationMode.BICUBIC),
    T.CenterCrop(560),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

# Load and preprocess the two images
image_paths = [os.path.join(img_dir, image_files[i]) for i in range(2)]
images = [Image.open(image_path) for image_path in image_paths]
imgs_tensor = torch.stack([transform(img)[:3] for img in images]).unsqueeze(0).to(device)

# Inference
with torch.no_grad():
    features_dict = dino.forward_features(imgs_tensor)
    features = features_dict['x_norm_patchtokens']

# Compute PCA between the patches of the images
features = features.reshape(2 * 1600, 1536)
pca = PCA(n_components=2)
pca_result = pca.fit_transform(features.cpu())

# Plot PCA results
plt.scatter(pca_result[:1600, 0], pca_result[:1600, 1], label='Image 1')
plt.scatter(pca_result[1600:, 0], pca_result[1600:, 1], label='Image 2')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend()
plt.show()