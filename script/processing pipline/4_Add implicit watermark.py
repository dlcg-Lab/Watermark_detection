import cv2
import numpy as np
import os

import numpy.random
import random

# 读取水印文件并转换为灰度图
watermark_path = "C:\\Users\\57411\\Desktop\\g1_resized.jpg"
watermark = cv2.imread(watermark_path, cv2.IMREAD_GRAYSCALE)

# 遍历cut-dowoRGB_val2017目录下的文件并添加隐式水印
input_dir = "C:\\Users\\57411\\Desktop\\cut-dowoRGB_val2017"
output_dir = "C:\\Users\\57411\\Desktop\\watermark"
for filename in os.listdir(input_dir):
    # 读取图片并转换为灰度图
    img_path = os.path.join(input_dir, filename)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # 对图片进行DCT变换
    dct = cv2.dct(np.float32(img))

    # 将水印嵌入DCT系数
    alpha = random.uniform(0.01, 0.04)
    dct_watermarked = dct + alpha * watermark

    # 对嵌有水印的DCT系数进行IDCT变换
    watermarked = cv2.idct(np.float32(dct_watermarked))

    # 将带有隐式水印的图片输出到目标目录
    output_path = os.path.join(output_dir, filename)
    cv2.imwrite(output_path, watermarked)

# 添加水印
