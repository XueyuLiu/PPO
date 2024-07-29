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
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # 单GPU或者CPU
dino.to(device)
# dino = dino.cuda()

# Load the images
img_dir = '/Data20T/data20t/data20t/envs/shigz1/code/dinov2-main/dinov2-main/images/rabbit'
""" res1_dir = '/Data20T/data20t/data20t/envs/shigz1/code/dinov2-main/dinov2-main/images/result1'
res2_dir = '/Data20T/data20t/data20t/envs/shigz1/code/dinov2-main/dinov2-main/images/result2'
res3_dir = '/Data20T/data20t/data20t/envs/shigz1/code/dinov2-main/dinov2-main/images/result3' """
image_files = os.listdir(img_dir)
images = []
numbers = 0

for image in image_files:
    img = Image.open(os.path.join(img_dir, image))
    img_array = np.array(img)
    images.append(img_array)
    numbers = numbers + 1

# Preprocess & convert to tensor
transform = T.Compose([
    T.Resize(560, interpolation=T.InterpolationMode.BICUBIC),
    T.CenterCrop(560),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

# Use image tensor to load in a batch of images
batch_size = numbers
imgs_tensor = torch.zeros(batch_size, 3, 560, 560)
for i, img in enumerate(images):
    img = Image.fromarray(img)
    imgs_tensor[i] = transform(img)[:3]

# Inference
with torch.no_grad():
    features_dict = dino.forward_features(imgs_tensor.to(device))
    features = features_dict['x_norm_patchtokens']

print(features.shape)

# Compute PCA between the patches of the image
features = features.reshape(numbers * 1600, 1536)
features = features.cpu()

features_1 = features.reshape(0 * 1600, 1536)
features_2 = features.reshape(1 * 1600, 1536)
# 将 PyTorch 张量转换为 NumPy 数组
features_np = features.numpy()

# 保存到 txt 文件
np.savetxt(r'/Data20T/data20t/data20t/envs/shigz1/code/dinov2-main/dinov2-main/images/rabbit/output/features.txt', features_np)
print(features)
""" pca = PCA(n_components=3)
pca.fit(features)
pca_features = pca.transform(features)

# Visualize the first PCA component
for i in range(numbers):
    plt.subplot(1, numbers, i + 1)
    plt.imshow(pca_features[i * 1600: (i + 1) * 1600, 0].reshape(40, 40))
#plt.show()

# Compute PCA between the patches of the image
features = features.reshape(numbers * 1600, 1536)
features = features.cpu()
pca = PCA(n_components=3)
pca.fit(features)
pca_features = pca.transform(features)

# Visualize the first PCA component
for i in range(numbers):
    plt.subplot(1, numbers, i + 1)
    plt.imshow(pca_features[i * 1600: (i + 1) * 1600, 0].reshape(40, 40))
    name = image_files[i].split('\\')[-1].split('.')[0]
    plt.savefig(f"/Data20T/data20t/data20t/envs/shigz1/code/dinov2-main/dinov2-main/images/result2/{name}.png")
#plt.show()

# Remove background
forground = pca_features[:, 0] < 1  # Adjust threshold accordingly
background = ~forground

# Fit PCA
pca.fit(features[forground])
features_forground = pca.transform(features[forground])

# Transform and visualize the first 3 PCA components
for i in range(numbers):
    features_forground[:, i] = (features_forground[:, i] - features_forground[:, i].min()) / (
                features_forground[:, i].max() - features_forground[:, i].min())
rgb = pca_features.copy()
rgb[background] = 0
rgb[forground] = features_forground
rgb = rgb.reshape(numbers, 40, 40, 3)
for i in range(numbers):
    #plt.subplot(1, numbers, i + 1)
    #plt.imshow(rgb[i][..., ::-1])
    name = image_files[i].split('\\')[-1].split('.')[0]
    plt.savefig(f"/Data20T/data20t/data20t/envs/shigz1/code/dinov2-main/dinov2-main/images/result3/{name}.png")
#plt.show() """
