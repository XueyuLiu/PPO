import cv2
import numpy as np
import os
from PIL import Image

pic_path =r'/Data20T/data20t/data20t/envs/shigz1/code/dinov2-main/dinov2-main/images/GT_png_86' # 分割的图片的位置
pic_target =r'/Data20T/data20t/data20t/envs/shigz1/code/dinov2-main/dinov2-main/images/GT_cut/' # 分割后的图片保存的文件夹

""" def jpeg_to_png_batch(input_folder, output_folder):
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
                # img_rgb = img_rgb.resize((560, 560))

                # 保存为PNG图像
                img_rgb.save(output_path)

# 调用函数进行批量转换
jpeg_to_png_batch(pic_path, pic_target)  """


if not os.path.exists(pic_target):  #判断是否存在文件夹如果不存在则创建为文件夹
    os.makedirs(pic_target)
#要分割后的尺寸
cut_width = 512
cut_length = 512
overlap=0
# 读取要分割的图片，以及其尺寸等数据

pic_list=os.listdir(pic_path)
print(pic_list)
for name in pic_list:
    print(os.path.join(pic_path,name))
    picture = cv2.imread(os.path.join(pic_path,name))
    picture=cv2.resize(picture,(2048,2048))
    (width, length, depth) = picture.shape
# 预处理生成0矩阵
    pic = np.zeros((cut_width, cut_length, depth))
# 计算可以划分的横纵的个数
    num_width = int(width / cut_width)
    num_length = int(length / cut_length)
# for循环迭代生成
    num=0
    for i in range(0, num_width):
        for j in range(0, num_length):
            pic = picture[i*cut_width-i*overlap : (i+1)*cut_width-i*overlap, j*cut_length-j*overlap : (j+1)*cut_length-j*overlap, :]      
            result_path = pic_target +name.split('.')[0]+'_{}.png'.format(num) # name.split('_')[0]+
            # result_path = pic_target + name[:-4]+'_{}.png'.format(num)
            num=num+1
            print(result_path)
            cv2.imwrite(result_path, pic)
 
print("done!!!")