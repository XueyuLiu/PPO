""" import os
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
img_dir = '/Data20T/data20t/data20t/envs/shigz1/code/dinov2-main/dinov2-main/images/predict'
res1_dir = '/Data20T/data20t/data20t/envs/shigz1/code/dinov2-main/dinov2-main/images/result1'
res2_dir = '/Data20T/data20t/data20t/envs/shigz1/code/dinov2-main/dinov2-main/images/result2'
res3_dir = '/Data20T/data20t/data20t/envs/shigz1/code/dinov2-main/dinov2-main/images/result3'
image_files = os.listdir(img_dir)
images = []
numbers = 0
print(len(image_files))

# Preprocess & convert to tensor
transform = T.Compose([
    T.Resize(560, interpolation=T.InterpolationMode.BICUBIC),
    T.CenterCrop(560),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

for image in image_files:
    img = Image.open(os.path.join(img_dir, image))
    img_array = np.array(img)
    images.append(img_array)
    # numbers = numbers + 1
    
    imgs_tensor = torch.zeros(batch_size, 3, 560, 560)
    img = Image.fromarray(img)
    imgs_tensor[i] = transform(img)[:3]

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
features = features.cpu() """

import torch
import torchvision
from torchvision import transforms
# from transformers import VisionPerceiver
import torchvision.transforms as T
import torch.nn.functional as F
import hubconf
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# 加载预训练的DinoV2模型
# dino_model = VisionPerceiver.from_pretrained('Salesforce/dino-vits-base')
dino = hubconf.dinov2_vitg14()
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu") # 单GPU或者CPU
dino_model = dino.to(device)
# 加载两张示例图像并进行预处理
transform = T.Compose([
    T.Resize(560, interpolation=T.InterpolationMode.BICUBIC),
    T.CenterCrop(560),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

""" 
image1 = transform(torchvision.io.read_image('/Data20T/data20t/data20t/envs/shigz1/code/dinov2-main/dinov2-main/images/test_0/001_248921_4.png')).unsqueeze(0)  # 替换为第一张图像的路径
image2 = transform(torchvision.io.read_image('/Data20T/data20t/data20t/envs/shigz1/code/dinov2-main/dinov2-main/images/test_0/001_248921_5.png')).unsqueeze(0)  # 替换为第二张图像的路径 """
image1_path = '/Data20T/data20t/data20t/envs/shigz1/code/dinov2-main/dinov2-main/images/test_0/001_248921_4.png'
image2_path = '/Data20T/data20t/data20t/envs/shigz1/code/dinov2-main/dinov2-main/images/test_0/001_248921_5.png'
image1_pil = Image.open(image1_path).convert('RGB')  # 使用PIL打开图像，并确保转换为RGB格式
image2_pil = Image.open(image2_path).convert('RGB')
image1 = transform(image1_pil).unsqueeze(0)  # 应用预处理并添加批量维度
image2 = transform(image2_pil).unsqueeze(0)  
image1 = image1.to(device)
image2 = image2.to(device)
img1_array = np.array(image1)
img2_array = np.array(image2)

images = []
images.append(img1_array)
images.append(img2_array)
    # numbers = numbers + 1
    
imgs_tensor = torch.zeros(batch_size, 3, 560, 560)
img = Image.fromarray(img)
imgs_tensor[i] = transform(img)[:3]


# 使用DinoV2模型提取图像特征
# 3. 使用DinoV2模型提取图像特征
with torch.no_grad():
    # 将图像特征从您的代码中提取的 features_dict 中取出
    features_dict = dino.forward_features(imgs_tensor.to(device))
    features = features_dict['x_norm_patchtokens'].to(device)
with torch.no_grad():
    features1 = dino_model(image1)  # 调用模型的forward方法获得特征
    features2 = dino_model(image2)

""" # Inference
with torch.no_grad():
    features_dict = dino.forward_features(imgs_tensor.to(device))
    features = features_dict['x_norm_patchtokens'] """

# 获取特征表示
features1 = features1.last_hidden_states
features2 = features2.last_hidden_states

# 将特征展平为一维向量
features1_flat = features1.view(features1.size(0), -1)
features2_flat = features2.view(features2.size(0), -1)

# 计算余弦相似度
similarity = F.cosine_similarity(features1_flat, features2_flat, dim=1)

# 将相似度信息映射到图像上
similarity_map = similarity.view(image1.shape[2], image1.shape[3])  # 重塑为与图像相同的大小

# 创建热力图
plt.imshow(similarity_map, cmap='hot', interpolation='nearest')
plt.axis('off')
plt.colorbar()
plt.show()
plt.savefig('/Data20T/data20t/data20t/envs/shigz1/code/dinov2-main/dinov2-main/images/test_0/output/result.png')

