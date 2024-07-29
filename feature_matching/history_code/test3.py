import os
import torch
import torchvision.transforms as T
import hubconf
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.decomposition import PCA
from skimage import color

# Load the largest dino model
dino = hubconf.dinov2_vitg14()
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")  # 单GPU或者CPU
dino.to(device)

# Load the images
img_dir = '/Data20T/data20t/data20t/envs/shigz1/code/dinov2-main/dinov2-main/images/test_0'
image_files = os.listdir(img_dir)

# Preprocess & convert to tensor
transform = T.Compose([
    T.Resize(560, interpolation=T.InterpolationMode.BICUBIC),
    T.CenterCrop(560),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

# Load and preprocess the two images
# Assuming the first two images in the directory are the ones you want to compare
image_paths = [os.path.join(img_dir, image_files[i]) for i in range(2)]
images = [Image.open(image_path) for image_path in image_paths]
imgs_tensor = torch.stack([transform(img)[:3] for img in images]).to(device)
print(imgs_tensor.shape)

# 假设image1和image2是PIL图像对象
for i in range(len(images)):
    images[i] = images[i].resize((560, 560))
image1_np = np.array(images[0])
image2_np = np.array(images[1])

# 创建两个画布，用于盖在原图像上
overlay1 = np.copy(image1_np)
overlay2 = np.copy(image2_np)

# Inference
with torch.no_grad():
    features_dict = dino.forward_features(imgs_tensor)
    features = features_dict['x_norm_patchtokens']

# Compute feature similarity (e.g., cosine similarity)
# Assuming features is a tensor of shape (2, num_patches, feature_dim)
# You can modify this part based on your specific similarity measure
print(features.shape)

# 初始化存储结果的数组
max_similarity_values = []
max_similarity_indices_features0 = []
max_similarity_indices_features1 = []

# 对于 features[0] 中的每个向量
for i in range(len(features[0])):
    # 计算当前 features[0] 向量与 features[1] 中所有向量的余弦相似度
    similarity_values = torch.cosine_similarity(features[0][i].unsqueeze(0), features[1], dim=1)
    
    # 找到最高相似度及对应索引
    max_sim_value, max_sim_index = torch.max(similarity_values, dim=0)
    
    # 将相似度和索引保存到数组
    max_similarity_values.append(max_sim_value.item())
    max_similarity_indices_features0.append(i)
    max_similarity_indices_features1.append(max_sim_index.item())

image1_index = []
image2_index = []
values_index = []
# 将数组转换为 numpy 数组（如果需要）
max_similarity_values = np.array(max_similarity_values)
max_similarity_indices_features0 = np.array(max_similarity_indices_features0)
max_similarity_indices_features1 = np.array(max_similarity_indices_features1)
for i in range(len(max_similarity_values)):
        if max_similarity_values[i] > 0.9:
            image1_index.append(max_similarity_indices_features0[i])
            image2_index.append(max_similarity_indices_features1[i])
            values_index.append(max_similarity_values[i])

# 使用patch的索引对应回像素点左上角坐标
patch_size = 14
num_patches = len(image2_index)

# 计算每个patch的左上角像素点坐标
patch_set1 = np.empty((num_patches, 2), dtype=int)
patch_set2 = np.empty((num_patches, 2), dtype=int)
for i in range(num_patches):
    # 计算当前patch的行号和列号
    row1 = image1_index[i] // (40)  # 假设40个patches一行
    col1 = image1_index[i] % (40)  # 假设40个patches一行
    row2 = image2_index[i] // (40) 
    col2 = image2_index[i] % (40)  

    # 计算左上角像素点坐标
    x1 = col1 * patch_size
    y1 = row1 * patch_size
    x2 = col2 * patch_size
    y2 = row2 * patch_size

    patch_set1[i] = (x1, y1)
    patch_set2[i] = (x2, y2)

""" fixed_alpha = int(0.3 * 255)  # 固定透明度值
# 生成n个随机颜色
colors = np.random.randint(0, 255, (num_patches, 3))
colors_with_alpha = np.hstack((colors, np.full((num_patches, 1), fixed_alpha))) """

fixed_alpha = int(0.3 * 255)  # 固定透明度值
# 生成n个随机颜色，从红色开始依次变化
colors_hsv = np.zeros((num_patches, 3), dtype=np.uint8)
colors_hsv[:, 0] = np.linspace(0, 179, num_patches, dtype=np.uint8)  # H: 0-179 (red to blue)
colors_hsv[:, 1] = 255  # S: 255 (full saturation)
colors_hsv[:, 2] = 255  # V: 255 (full brightness)

# 转换成 RGB 颜色空间
colors_rgb = color.hsv2rgb(colors_hsv.reshape(1, -1, 3)).reshape(-1, 3) * 255
colors_with_alpha = np.hstack((colors_rgb, np.full((num_patches, 1), fixed_alpha, dtype=np.uint8)))

# 假设patch左上角的坐标
# 创建一个画布
img_height = 560
img_width = 560
""" 
img1 = np.zeros((img_height, img_width, 4), dtype=np.uint8)
img2 = np.zeros((img_height, img_width, 4), dtype=np.uint8)
"""
print(overlay1.shape)

# 将不同颜色的patches盖在对应的图片上
for i, point in enumerate(patch_set1):
    x, y = point
    # color = colors_with_alpha[i % len(colors_with_alpha)]  # 取颜色序列中的颜色
    color_new = colors_rgb[i % len(colors_rgb)]
    # print(np.tile(color, (patch_size, patch_size, 1)).shape)
    overlay1[y:y+patch_size, x:x+patch_size] = np.tile(color_new, (patch_size, patch_size, 1))

for i, point in enumerate(patch_set2):
    x, y = point
    color_new = colors_rgb[i % len(colors_rgb)]
    # color = colors[i % len(colors)]  # 取颜色序列中的颜色
    overlay2[y:y+patch_size, x:x+patch_size] = np.tile(color_new, (patch_size, patch_size, 1))

""" overlay_image1 = Image.fromarray(overlay1)
overlay_image2 = Image.fromarray(overlay2) """

# 将overlay盖在原图像上
alpha = 0.7  # 透明度
img1_np = cv2.addWeighted(np.array(images[0]), alpha, overlay1, 1-alpha, 0)
img2_np = cv2.addWeighted(np.array(images[1]), alpha, overlay2, 1-alpha, 0)

img1 = Image.fromarray(img1_np)
img2 = Image.fromarray(img2_np)

# 保存图片
plt.imsave('/Data20T/data20t/data20t/envs/shigz1/code/dinov2-main/dinov2-main/images/test_0/output/image1.png', img1)
plt.imsave('/Data20T/data20t/data20t/envs/shigz1/code/dinov2-main/dinov2-main/images/test_0/output/image2.png', img2)

""" 
# 为每个patch创建mask并应用颜色
for i in range(num_patches):
    color = colors_with_alpha[i]
    coord = img[i]
    mask = np.zeros((img_height, img_width, 4), dtype=np.uint8)
    mask[coord[0]:coord[0] + 14, coord[1]:coord[1] + 14] = color
    img = np.maximum(img, mask)  # 合并mask

# 去掉 Alpha 通道，只保留 RGB
result_img = img[:, :, :3] """

# 保存图片
# plt.imsave('/Data20T/data20t/data20t/envs/shigz1/code/dinov2-main/dinov2-main/images/test_0/output/image.png', result_img)

""" 
# 指定保存路径和文件名
file_path = '/Data20T/data20t/data20t/envs/shigz1/code/dinov2-main/dinov2-main/images/test_0/output/similarity_data_set.txt'

# 保存 max_similarity_values 到文件
np.savetxt(file_path, values_index, fmt='%0.6f', header='Values', delimiter=',')

# 保存 max_similarity_indices_features0 到文件
with open(file_path, 'ab') as f:
    np.savetxt(f, patch_set1, fmt='%d', header='image1', delimiter=',')

# 保存 max_similarity_indices_features1 到文件
with open(file_path, 'ab') as f:
    np.savetxt(f, patch_set2, fmt='%d', header='image2', delimiter=',') """

""" # 将结果保存到 txt 文件
print(len(max_similarity_values))
file_path = '/Data20T/data20t/data20t/envs/shigz1/code/dinov2-main/dinov2-main/images/test_0/output/similarity_data_09.txt'
with open(file_path, 'w') as f:
    for i in range(len(max_similarity_values)):
        if max_similarity_values[i] > 0.9:
            f.write(f'For features[0] patch {max_similarity_indices_features0[i]}, highest similarity is with features[1] patch {max_similarity_indices_features1[i]}, similarity value: {max_similarity_values[i]}\n')
 """
""" similarity_matrix = torch.cosine_similarity(features[0], features[1], dim=1)
print(similarity_matrix.shape)

# 初始化存储结果的数组
max_similarity_values = []
max_similarity_indices = []

# 对于 features[0] 中的每个 patch
for i in range(len(features[0])):
    # 找到 features[1] 中与 features[0][i] 最相似的 patch 的索引和相似度
    max_sim_value, max_sim_index = torch.max(similarity_matrix[i], dim=0)
    
    # 将相似度和索引保存到数组
    max_similarity_values.append(max_sim_value.item())
    max_similarity_indices.append(max_sim_index.item())

# 将数组转换为 numpy 数组（如果需要）
max_similarity_values = np.array(max_similarity_values)
max_similarity_indices = np.array(max_similarity_indices)

# 将结果保存到 txt 文件
file_path = '/Data20T/data20t/data20t/envs/shigz1/code/dinov2-main/dinov2-main/images/test_0/output/similarity_data.txt'
with open('file_path', 'w') as f:
    for i in range(len(max_similarity_values)):
        f.write(f'Patch {i} in features[0] has highest similarity with patch {max_similarity_indices[i]} in features[1]: {max_similarity_values[i]}\n') """

""" # 找出最大相似度对应的索引
max_similarity_index = torch.argmax(similarity).item()

# 将最大相似度的索引转换为两张图片的位置索引
image1_index = max_similarity_index // 1600  # 假设每张图片有1600个像素块
image2_index = max_similarity_index % 1600  # 假设每张图片有1600个像素块

print("Max similarity at image1 index:", image1_index)
print("Max similarity at image2 index:", image2_index)
print(image2_index.shape) """

""" # 将数组保存到一个字典
data = {
    'max_similarity_values': max_similarity_values,
    'max_similarity_indices': max_similarity_indices,
    'second_image_patch_indices': second_image_patch_indices
}

# 指定保存路径和文件名
file_path = '/Data20T/data20t/data20t/envs/shigz1/code/dinov2-main/dinov2-main/images/test_0/output/similarity_data.txt'

# 保存为txt文件
np.savetxt(file_path, data, fmt='%s')
print('Data saved to', file_path)  """

""" # Visualize similarity as an image
similarity_2d = similarity.view(40, 40)

# 假设 cosine_similarity 是两张图片的余弦相似度矩阵
# 假设它是一个 40x40 的矩阵，表示 40x40 个特征的相似度
cosine_similarity = np.random.rand(40, 40)

# 初始化数组来保存结果
max_similarity_values = []
max_similarity_indices = []
second_image_patch_indices = []

# 对于第一张图片的每个 patch，找到对应的最高相似度和索引
for i in range(40):
    # 对于每个第一张图片的 patch，找到与第二张图片所有 patch 的最高相似度
    max_similarity_value = np.max(cosine_similarity[i])
    max_similarity_index = np.argmax(cosine_similarity[i])
    
    # 将最高相似度及其索引保存到相应数组中
    max_similarity_values.append(max_similarity_value)
    max_similarity_indices.append(max_similarity_index)
    second_image_patch_indices.append(max_similarity_index)  # 第二张图片对应的 patch 索引也保存了最高相似度对应的索引

# 打印结果
print("Max Similarity Values:", max_similarity_values)
print("Max Similarity Indices:", max_similarity_indices)
print("Second Image Patch Indices:", second_image_patch_indices)

# 将数组转换为numpy数组
max_similarity_values = np.array(max_similarity_values)
max_similarity_indices = np.array(max_similarity_indices)
second_image_patch_indices = np.array(second_image_patch_indices)

# 将数组保存到一个字典
data = {
    'max_similarity_values': max_similarity_values,
    'max_similarity_indices': max_similarity_indices,
    'second_image_patch_indices': second_image_patch_indices
}

# 指定保存路径和文件名
file_path = '/Data20T/data20t/data20t/envs/shigz1/code/dinov2-main/dinov2-main/images/test_0/output/similarity_data.txt'

# 保存为txt文件
np.savetxt(file_path, data, fmt='%s')
print('Data saved to', file_path) """

""" plt.imshow(similarity_2d.cpu().numpy(), cmap='hot', interpolation='nearest')
plt.colorbar()
plt.show()
plt.savefig("/Data20T/data20t/data20t/envs/shigz1/code/dinov2-main/dinov2-main/images/test_0/output/result.png") """

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
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu") # 单GPU或者CPU
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
pca = PCA(n_components=3)
pca.fit(features)
pca_features = pca.transform(features)

# Visualize the first PCA component
for i in range(numbers):
    #plt.subplot(1, numbers, i + 1)
    plt.imshow(pca_features[i * 1600: (i + 1) * 1600, 0].reshape(40, 40))
    name = image_files[i].split('\\')[-1].split('.')[0]
    plt.savefig(f"/Data20T/data20t/data20t/envs/shigz1/code/dinov2-main/dinov2-main/images/result1/{name}.png")
#plt.show()

# Compute PCA between the patches of the image
features = features.reshape(numbers * 1600, 1536)
features = features.cpu()
pca = PCA(n_components=3)
pca.fit(features)
pca_features = pca.transform(features)

# Visualize the first PCA component
for i in range(numbers):
    #plt.subplot(1, numbers, i + 1)
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
    plt.imshow(rgb[i][..., ::-1])
    name = image_files[i].split('\\')[-1].split('.')[0]
    plt.savefig(f"/Data20T/data20t/data20t/envs/shigz1/code/dinov2-main/dinov2-main/images/result3/{name}.png")
#plt.show() """
