import os
import torch
import torchvision.transforms as T
import hubconf
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# from skimage import color

# Load the largest dino model
dino = hubconf.dinov2_vitg14()
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")  # 单GPU或者CPU
dino.to(device)

# Load the images
img_dir = '/Data20T/data20t/data20t/envs/shigz1/code/dinov2-main/dinov2-main/images/test1017/input'
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

""" images_ori = [] 

for i in range(len(image_files)):
    print(image_files[i])
    img = Image.open(image_paths[i])
    images_ori.append(img)"""

""" # import copy
images_ori = [Image.open(image_path) for image_path in image_paths]
# images_ori[0].save(f'/Data20T/data20t/data20t/envs/shigz1/code/dinov2-main/dinov2-main/images/test_0/output/image.png')

for i in range(len(images_ori)):
    images_ori[i] = images_ori[i].resize((560, 560)) """

images = [Image.open(image_path) for image_path in image_paths]
for i in range(len(images)):
    images[i] = images[i].convert('RGB')
    images[i] = images[i].resize((560, 560))

imgs_tensor = torch.stack([transform(img)[:3] for img in images]).to(device)
print('imgs_tensor的shape是 ' + str(imgs_tensor.shape))

# Inference
with torch.no_grad():
    features_dict = dino.forward_features(imgs_tensor)
    features = features_dict['x_norm_patchtokens']

# Compute feature similarity (e.g., cosine similarity)
# Assuming features is a tensor of shape (2, num_patches, feature_dim)
# You can modify this part based on your specific similarity measure
print('features的shape是 '  + str(features.shape))

# Initialize lists to store results
min_distance_values = []
min_distance_indices_features0 = []
min_distance_indices_features1 = []

# Loop over each vector in features[0]
for i in range(len(features[0])):   # range(len(features[0])):
    # Calculate Euclidean distance (L2 norm) to all vectors in features[1]
    euclidean_distances = torch.norm(features[0][i] - features[1], dim=1, p=2)
    #print('euclidean_distances是 ' + euclidean_distances)
    #print('euclidean_distances的shape是 ' + euclidean_distances.shape)
    # Find the index with the minimum distance
    min_distance, min_distance_index = torch.min(euclidean_distances, dim=0)

    # Append the minimum distance and corresponding index to the lists
    min_distance_values.append(min_distance.item())
    min_distance_indices_features0.append(i)
    min_distance_indices_features1.append(min_distance_index.item())

min_values = []
min_indexs = []

# 按patch计算欧氏距离
for i in range(len(features[0])):   # range(len(features[0])):
    distance_j = []
    distance_j_index = -1

    # 计算i->j的distance
    for j in range(len(features[1])):
        euclidean_distances = torch.norm(features[0][i] - features[1][j], p=2)
        # print(euclidean_distances.item())
        distance_j.append(euclidean_distances.item())

    # sort i->all j distances
    temp_distance = sorted(distance_j)
    min_distance = temp_distance[0]

    # 遍历数组,找到min值和索引
    for index in distance_j:
        if min_distance == distance_j[i]:
            distance_j_index = index
    
    min_values.append(min_distance)
    min_indexs.append(distance_j_index)
    
"""     
euclidean_distances = torch.norm(features[0][i] - features[1], dim=1, p=2)
    #print('euclidean_distances是 ' + euclidean_distances)
    #print('euclidean_distances的shape是 ' + euclidean_distances.shape)
    # Find the index with the minimum distance
    min_distance, min_distance_index = torch.min(euclidean_distances, dim=0)

    # Append the minimum distance and corresponding index to the lists
    min_distance_values.append(min_distance.item())
    min_distance_indices_features0.append(i)
    min_distance_indices_features1.append(min_distance_index.item()) """

""" 
# 初始化一个空字典
value_counts = {}

# 统计每个值出现的次数
for val in min_distance_values:
    if val in value_counts:
        value_counts[val] += 1
    else:
        value_counts[val] = 1 """

# 输出结果
# print("值及其出现次数：", value_counts)

# 找出第n小的元素及其下标
min_number = 0  # 假设要找第5小的元素
max_number = 15

sorted_smallest_value = sorted(min_values)
# 初始化第n小的元素和其下标
nth_smallest_value = sorted_smallest_value[min_number]   # float('inf')  # 正无穷大
nth_bigest_value = sorted_smallest_value[max_number]   # float('inf')  # 正无穷大

nth_smallest_index = -1
nth_bigest_index = -1

# 遍历数组找第n小的元素
# 遍历数组,找到第n小的值和索引
for i, value in enumerate(min_distance_values):
    if value == nth_smallest_value:
        nth_smallest_index = i
    if value == nth_bigest_value:
        nth_smallest_index = i

# 输出结果
print(f"第{min_number}小的元素是 {nth_smallest_value}，其下标为 {nth_smallest_index}") 
print(f"第{max_number}小的元素是 {nth_bigest_value}，其下标为 {nth_bigest_index}") 

print('min_distance_values的len是:', len(min_distance_values))
print('min_distance_indices_features0的len是:', len(min_distance_indices_features0))
print('min_distance_indices_features1的len是:', len(min_distance_indices_features1))

threshold_min = nth_smallest_value
threshold_max = nth_bigest_value

image1_index = []
image2_index = []
values_index = []

for i in range(len(min_distance_values)):
    if (min_distance_values[i] > threshold_min) and (min_distance_values[i] < threshold_max):
        image1_index.append(min_distance_indices_features0[i])
        image2_index.append(min_distance_indices_features1[i])
        values_index.append(min_distance_values[i])

# 使用patch的索引对应回像素点左上角坐标
patch_size = 14
num_patches = len(values_index)

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

    patch_set1[i] = (x1, y1)# (x1, y1)
    patch_set2[i] = (x2, y2)

print('patch_set1的len是 '  + str(len(patch_set1)))
print('patch_set1是 '  + str(patch_set1))

print('patch_set2的len是 '  + str(len(patch_set2)))
print('patch_set2是 '  + str(patch_set2))

fixed_alpha = int(1 * 255)  # 固定透明度值
# 生成n个随机颜色，从红色开始依次变化
colors_rgb = np.zeros((num_patches, 3), dtype=np.uint8)
colors_rgb[:, 0] = np.linspace(0, 255, num_patches, dtype=np.uint8)
# colors_hsv[:, 0] = np.linspace(0, 179, num_patches, dtype=np.uint8)  # H: 0-179 (red to blue)
colors_rgb[:, 1] = np.linspace(255, 0, num_patches, dtype=np.uint8)  # S: 255 (full saturation)
colors_rgb[:, 2] = np.linspace(180, 0, num_patches, dtype=np.uint8)  # V: 255 (full brightness)

# print('colors_rgb是 '  + str(colors_rgb))
# 转换成 RGB 颜色空间
# colors_rgb = color.hsv2rgb(colors_hsv.reshape(1, -1, 3)).reshape(-1, 3) * 255
# colors_with_alpha = np.hstack((colors_rgb, np.full((num_patches, 1), fixed_alpha, dtype=np.uint8)))
colors_rgb = colors_rgb.astype(int)

# 转换为元组
colors_rgb_tuples = [tuple(row) for row in colors_rgb]

print('colors_rgb_tuples是 '  + str(colors_rgb_tuples))
print('colors_rgb的type是 '  + str(type(colors_rgb)))

# 假设patch左上角的坐标
# 创建一个画布
img_height = 560
img_width = 560

""" image1_np = np.array(images_ori[0])
image2_np = np.array(images_ori[1]) """
image1_np = images[0]
image2_np = images[1]
# print(images_ori[0].getbands())

# 将不同颜色的patches盖在对应的图片上
for patch1, patch2, mask_color in zip(patch_set1, patch_set2, colors_rgb_tuples):
    x1, y1 = patch1
    x2, y2 = patch2
    mask_color_rgb = mask_color  #mask_color[::-1] PIL 使用 RGB 而不是 BGR
    # print('mask_colors_rgb是 ' + str(mask_color_rgb))
    # print('mask_colors_rgb的type是 '  + str(type(mask_color_rgb)))
    # 创建 mask 图像，大小与 patch 区域一致
    mask = Image.new('RGB', (patch_size, patch_size), mask_color_rgb)
    
    # 将 mask 图像粘贴到原图像上
    image1_np.paste(mask, (y1, x1))
    image2_np.paste(mask, (y2, x2))

# 保存图片
image1_np.save('/Data20T/data20t/data20t/envs/shigz1/code/dinov2-main/dinov2-main/images/test1017/output/1.png', format='PNG')
image2_np.save('/Data20T/data20t/data20t/envs/shigz1/code/dinov2-main/dinov2-main/images/test1017/output/2.png', format='PNG')

""" 
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
    if max_sim_value > 0.95:
        # 将相似度和索引保存到数组
        max_similarity_values.append(max_sim_value.item())
        max_similarity_indices_features0.append(i)
        max_similarity_indices_features1.append(max_sim_index.item()) 
    
print('max_similarity_values的len是 '  + str(len(max_similarity_values)))
print('max_similarity_values是 '  + str(max_similarity_values))
print('max_similarity_indices_features0的len是 '  + str(len(max_similarity_indices_features0)))
print('max_similarity_indices_features0是 '  + str(max_similarity_indices_features0))
print('max_similarity_indices_features0的len是 '  + str(len(max_similarity_indices_features1)))
print('max_similarity_indices_features1是 '  + str(max_similarity_indices_features1))

image1_index = []
image2_index = []
values_index = []
# 将数组转换为 numpy 数组（如果需要）
max_similarity_values = np.array(similarity_values)
max_similarity_indices_features0 = np.array(similarity_indices_features0)
max_similarity_indices_features1 = np.array(similarity_indices_features1)

print('max_similarity_values的len是 '  + str(len(max_similarity_values)))
print('max_similarity_values[0]的type是 '  + str(type(max_similarity_values[0])))

for i in range(len(max_similarity_values)):
    if max_similarity_values[i] > 0.98:
        image1_index.append(max_similarity_indices_features0[i])
        image2_index.append(max_similarity_indices_features1[i])
        values_index.append(max_similarity_values[i])
"""