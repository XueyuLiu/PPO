import matplotlib.pyplot as plt
import numpy as np
import cv2
import os



def show_points(coords, labels, ax, marker_size=450):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='blue', marker='o', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='o', s=marker_size, edgecolor='white', linewidth=1.25)


img_dir=r"C:\Users\Administrator\Desktop\Dataset\image"
mask_dir=r"C:\Users\Administrator\Desktop\Dataset\mask"
result_dir=r"C:\Users\Administrator\Desktop\Dataset\results"

for name in os.listdir(img_dir):
    image = cv2.imread(os.path.join(img_dir,name))
    mask_image = cv2.imread(os.path.join(mask_dir,name), cv2.IMREAD_GRAYSCALE)
    mask_image = cv2.resize(mask_image, (image.shape[1], image.shape[0]))

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image)
    show_mask(mask_image/255, ax)

    input_point = input_point.astype(int)
    # print(input_point,input_label)
    show_points(input_point, input_label, ax)


    ax.axis('off')
    result_path = os.path.join(result_dir,'gt_'+name)
    plt.savefig(result_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)