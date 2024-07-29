import os
from PIL import Image

def jpeg_to_png_batch(input_folder, output_folder):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        if filename.endswith('.jpg') or filename.endswith('.jpeg'):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename.replace('.jpg', '.png').replace('.jpeg', '.png'))

            # 打开JPEG图像
            with Image.open(input_path) as img:
                # 将JPEG图像转换为RGBA模式，以保留透明度
                img_rgb = img.convert('RGB')
                img_rgb = img_rgb.resize((560, 560))

                # 保存为PNG图像
                img_rgb.save(output_path)

# 输入JPEG图像所在文件夹和输出PNG图像所在文件夹
input_folder = '/Data20T/data20t/data20t/envs/shigz1/code/dinov2-main/dinov2-main/images/rabbit/input'  # 替换为实际的JPEG图像文件夹路径
output_folder = '/Data20T/data20t/data20t/envs/shigz1/code/dinov2-main/dinov2-main/images/rabbit/input'  # 替换为实际的输出PNG图像文件夹路径

# 调用函数进行批量转换
jpeg_to_png_batch(input_folder, output_folder)
