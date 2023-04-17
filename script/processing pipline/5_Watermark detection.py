import cv2
import numpy as np
import os

# 读取水印文件并转换为灰度图
watermark_path = "C:\\Users\\57411\\Desktop\\g1_resized.jpg"
watermark = cv2.imread(watermark_path, cv2.IMREAD_GRAYSCALE)

# 初始化SIFT特征检测器
sift = cv2.xfeatures2d.SIFT_create()

# 遍历cut-dowoRGB_val2017目录下的文件并检测隐式水印
input_dir = "C:\\Users\\57411\\Desktop\\watermark"
output_file = "C:\\Users\\57411\\Desktop\\watermark_result.txt"
with open(output_file, 'w') as f:
    for filename in os.listdir(input_dir):
        # 检查文件名是否以".jpg"结尾
        if not filename.endswith(".jpg"):
            continue

        # 获取文件名和路径
        name = os.path.splitext(filename)[0]
        name = name[:-2]
        img_path_r = os.path.join(input_dir, name + "_r.jpg")
        img_path_g = os.path.join(input_dir, name + "_g.jpg")
        img_path_b = os.path.join(input_dir, name + "_b.jpg")

        # 读取三个通道的图片并转换为灰度图
        img_r = cv2.imread(img_path_r, cv2.IMREAD_GRAYSCALE)
        img_g = cv2.imread(img_path_g, cv2.IMREAD_GRAYSCALE)
        img_b = cv2.imread(img_path_b, cv2.IMREAD_GRAYSCALE)

        # 对三个通道的图片分别进行特征检测
        kp1_r, des1_r = sift.detectAndCompute(img_r, None)
        kp1_g, des1_g = sift.detectAndCompute(img_g, None)
        kp1_b, des1_b = sift.detectAndCompute(img_b, None)
        kp2, des2 = sift.detectAndCompute(watermark, None)

        # 进行特征匹配
        bf = cv2.BFMatcher()
        matches_r = bf.knnMatch(des1_r, des2, k=2)
        matches_g = bf.knnMatch(des1_g, des2, k=2)
        matches_b = bf.knnMatch(des1_b, des2, k=2)

        # 根据Lowe's ratio test进行特征筛选
        good_r = []
        good_g = []
        good_b = []
        for m, n in matches_r:
            if m.distance < 0.75 * n.distance:
                good_r.append([m])
        for m, n in matches_g:
            if m.distance < 0.75 * n.distance:
                good_g.append([m])
        for m, n in matches_b:
            if m.distance < 0.75 * n.distance:
                good_b.append([m])

        # 如果三个通道中至少一个匹配的特征点数大于50%，则认为含有隐式水印
        if len(good_r) > 0.5 or len(good_g) > 0.5 or len(good_b) > 0.5:
            f.write(name + ".jpg 1\n")
        else:
            f.write(name + ".jpg 0\n")
