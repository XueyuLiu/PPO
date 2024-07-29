import torch
import torchvision.transforms as T
import hubconf
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os


# 获得mask的前景patch与后景patch
mask_path = '/Data20T/data20t/data20t/envs/shigz1/code/dinov2-main/dinov2-main/images/test1018/Mask/'
mask_files = os.listdir(mask_path)
mask = Image.open(os.path.join(mask_path, mask_files[0]))# ("path_to_your_image.png")
mask = mask.resize((560, 560))

# 转换为NumPy数组
mask_np = np.array(mask)
# 输出数组的shape
print("Array shape:", mask_np.shape)

# 存储符合条件的 i 和 j
background_patchs = []
fore_patchs = []

# 遍历每个像素，注意不超出边界
for i in range(0, mask_np.shape[0] - 14, 14):
    for j in range(0, mask_np.shape[1] - 14, 14):
        # 检查当前像素和14x14区域像素是否都为0,0,0 背景点
        if np.all(mask_np[i:i+14, j:j+14] == [255, 255, 255]):
            #  mask_np[i, j] == [255, 255, 255] and mask_np[i+14, j+14] == [255, 255, 255]:
            fore_patchs.append((i, j))

# 输出符合条件的 i 和 j
print(len(fore_patchs))

def stratified_sampling(arr, n_samples):
    # 计算每层的采样间隔
    interval = len(arr) // n_samples
    
    # 初始化采样结果列表
    samples = []
    
    # 从每层中随机选择一个值作为样本
    for i in range(0, len(arr), interval):
        idx = np.random.randint(i, min(i+interval, len(arr)))
        samples.append(arr[idx])
    
    return samples

n_samples = 10

# 分层抽样
mask_samples = stratified_sampling(fore_patchs, n_samples)

# 输出样本
print("Stratified Samples:", mask_samples)

# 计算每个坐标点对应的patch索引
fore_index = np.empty((len(mask_samples), 1), dtype=int)

for i in range(len(mask_samples)):
    row = mask_samples[i][0] / 14
    col = mask_samples[i][1] / 14
    fore_index[i] = row * 40 + col


# Load the largest dino model
dino = hubconf.dinov2_vitg14()
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")  # 单GPU或者CPU
dino.to(device)

# Load the images
img_dir = '/Data20T/data20t/data20t/envs/shigz1/code/dinov2-main/dinov2-main/images/test1018/GT'
img_prompt = '/Data20T/data20t/data20t/envs/shigz1/code/dinov2-main/dinov2-main/images/test1018/GT_ori'
image_files = os.listdir(img_dir)
img_prompt_file = os.listdir(img_prompt)

# Preprocess & convert to tensor
transform = T.Compose([
    T.Resize(560, interpolation=T.InterpolationMode.BICUBIC),
    T.CenterCrop(560),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

# Load and preprocess the two images
# Assuming the first two images in the directory are the ones you want to compare
image_paths = [os.path.join(img_dir, image_files[i]) for i in range(len(image_files))]
image_prompt_paths = [os.path.join(img_prompt, img_prompt_file[0]) for i in range(len(img_prompt_file))]

images = [Image.open(image_path) for image_path in image_paths]

for i_path in range(0, len(image_paths), 1):   # range(len(images)):
    #创建内循环，每次使用prompt图像与一张子图像进行匹配
    images_inner_path = []
    images_inner_path.append(image_prompt_paths[0])
    images_inner_path.append(image_paths[i_path])

    images_inner = [Image.open(inner_path) for inner_path in images_inner_path]
    for i in range(len(images_inner)):
        images_inner[i] = images_inner[i].resize((560, 560))

    imgs_tensor = torch.stack([transform(img)[:3] for img in images_inner]).to(device)
    # print('imgs_tensor的shape是 ' + str(imgs_tensor.shape))

    # Inference
    with torch.no_grad():
        features_dict = dino.forward_features(imgs_tensor)
        features = features_dict['x_norm_patchtokens']

    # Compute feature similarity (e.g., cosine similarity)
    # Assuming features is a tensor of shape (2, num_patches, feature_dim)
    # You can modify this part based on your specific similarity measure
    # print('features的shape是 '  + str(features.shape))

    min_values = []
    min_indexs = []

    # 按patch计算欧氏距离
    for i in range(len(features[0])):   # range(len(features[0])):
        for i_index in fore_index:
            if i == i_index:
                distance_j = []
                distance_j_index = -1

                # 计算i->j的distance
                for j in range(len(features[1])):
                    euclidean_distances = torch.norm(features[0][i] - features[1][j], p=2)
                    distance_j.append(euclidean_distances.item())

                # sort i->all j distances
                temp_distance = sorted(distance_j)
                min_distance = temp_distance[0]

                # 遍历数组,找到min值和索引
                for index in range(len(distance_j)):
                    if min_distance == distance_j[index]:
                        distance_j_index = index
                
                min_values.append(min_distance)
                min_indexs.append(distance_j_index)
            
        i = i+1

        image1_index = []
        image2_index = []
        values_index = []

    for i in range(len(min_values)):
        image1_index.append(i)
        print(image1_index)
        image2_index.append(min_indexs[i])
        print(image2_index)
        values_index.append(min_values[i])
        print(values_index)

    # 使用patch的索引对应回像素点左上角坐标
    patch_size = 14
    num_patches = len(values_index)

    # 计算每个patch的左上角像素点坐标
    # patch_set1 = np.empty((num_patches, 2), dtype=int)
    patch_set2 = np.empty((num_patches, 2), dtype=int)
    for i in range(num_patches):
        # 计算当前patch的行号和列号
        row2 = image2_index[i] // (40) 
        col2 = image2_index[i] % (40)  

        # 计算左上角像素点坐标
        x2 = row2 * patch_size
        y2 = col2 * patch_size

        patch_set2[i] = (x2, y2)

    """     
    print('patch_set1是 ', mask_samples)
    print('patch_set2的len是 '  + str(len(patch_set2)))
    print('patch_set2是 ', patch_set2) """

    fixed_alpha = int(1 * 255)  # 固定透明度值
    # 生成n个随机颜色，从红色开始依次变化
    colors_rgb = np.zeros((num_patches, 3), dtype=np.uint8)
    colors_rgb[:, 0] = np.linspace(0, 255, num_patches, dtype=np.uint8)
    # colors_hsv[:, 0] = np.linspace(0, 179, num_patches, dtype=np.uint8)  # H: 0-179 (red to blue)
    colors_rgb[:, 1] = np.linspace(255, 0, num_patches, dtype=np.uint8)  # S: 255 (full saturation)
    colors_rgb[:, 2] = np.linspace(180, 0, num_patches, dtype=np.uint8)  # V: 255 (full brightness)

    # 转换成 RGB 颜色空间
    colors_rgb = colors_rgb.astype(int)

    # 转换为元组
    colors_rgb_tuples = [tuple(row) for row in colors_rgb]

    """     
    print('colors_rgb_tuples是 '  + str(colors_rgb_tuples))
    print('colors_rgb的type是 '  + str(type(colors_rgb))) """

    # 假设patch左上角的坐标
    # 创建一个画布
    img_height = 560
    img_width = 560

    """ image1_np = np.array(images_ori[0])
    image2_np = np.array(images_ori[1]) """
    image1_np = images_inner[0]
    image2_np = images_inner[1]
    mask1_np = mask
    # print(images_ori[0].getbands())

    # 将不同颜色的patches盖在对应的图片上
    for patch1, patch2, mask_color in zip(mask_samples, patch_set2, colors_rgb_tuples):
        x1, y1 = patch1
        x2, y2 = patch2
        mask_color_rgb = mask_color  #mask_color[::-1] PIL 使用 RGB 而不是 BGR
        # print('mask_colors_rgb是 ' + str(mask_color_rgb))
        # print('mask_colors_rgb的type是 '  + str(type(mask_color_rgb)))
        # 创建 mask 图像，大小与 patch 区域一致
        mask_ij = Image.new('RGB', (patch_size, patch_size), mask_color_rgb)
        
        # 将 mask 图像粘贴到原图像上
        # image1_np.paste(mask_ij, (y1, x1))
        image2_np.paste(mask_ij, (y2, x2))
        # mask1_np.paste(mask_ij, (y1, x1))

    name = image_files[i_path].split('\\')[-1].split('.')[0]

    # 保存图片
    # image1_np.save('/Data20T/data20t/data20t/envs/shigz1/code/dinov2-main/dinov2-main/images/test1016/Output/1.png', format='PNG')
    image2_np.save(f'/Data20T/data20t/data20t/envs/shigz1/code/dinov2-main/dinov2-main/images/test1018/Output/{name}.png', format='PNG')
    # mask1_np.save('/Data20T/data20t/data20t/envs/shigz1/code/dinov2-main/dinov2-main/images/test1016/Output/mask1.png', format='PNG')