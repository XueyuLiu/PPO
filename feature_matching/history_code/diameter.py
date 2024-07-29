import numpy as np
import os
import math
import skimage
from skimage.morphology import skeletonize
import cv2

def fill(img):
    # 二值化
    ret, img = cv2.threshold((np.uint8(img) * 255), 0, 255, cv2.THRESH_BINARY)

    # 填充外部孔洞
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.RETR_CCOMP)
    area = []

    for p in range(len(contours)):
        area.append(cv2.contourArea(contours[p]))
    max_idx = np.argmax(area)
    contours = list(contours)
    contours.remove(contours[max_idx])

    for q in range(0, len(contours)):
        cv2.fillConvexPoly(img, contours[q], 0)
    #填充内部孔洞
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    area = []

    for p in range(len(contours)):

        area.append(cv2.contourArea(contours[p]))
    max_idx = np.argmax(area)

    contours = list(contours)
    contours.remove(contours[max_idx])

    for q in range(0, len(contours)):
        cv2.fillConvexPoly(img, contours[q], 255)
    return img

def euclidean_distance(x, y):
    distance = math.sqrt(sum([(a - b) ** 2 for a, b in zip(x, y)]))
    return distance

def centerline(img):
    dilated = cv2.GaussianBlur(img, (5, 5), 0)
    skeleton = skeletonize(dilated / 255)
    return skeleton

def calculate(img):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    centerlist = np.where(centerline(img))
    centerlist = list(centerlist)
    dialist = []
    for idx in range(0, len(centerlist[0])):
        tempdist = []
        Cx = centerlist[1][idx]
        Cy = centerlist[0][idx]
        for dot in contours[0]:
            Ex = dot[0][0]
            Ey = dot[0][1]
            tempdist.append(euclidean_distance((Cx, Cy), (Ex, Ey)))
        dialist.append(min(tempdist) / 89)
    return dialist

if __name__ == '__main__':
    img = cv2.imread('/Data20T/data20t/data20t/envs/shigz1/code/dinov2-main/dinov2-main/images/mask/001_264264_60.jpg', 0)
    img=fill(img)
    dialist=(calculate(img))
    print(max(dialist))
