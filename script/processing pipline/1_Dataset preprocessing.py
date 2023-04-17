import cv2
import os

# 定义输入输出路径
input_path = 'C:/Users/57411/Desktop/val2017/'
output_path = 'C:/Users/57411/Desktop/output_val2017/'

# 创建输出目录
if not os.path.exists(output_path):
    os.makedirs(output_path)

# 遍历输入目录下的所有jpg文件
for file_name in os.listdir(input_path):
    if file_name.endswith('.jpg'):
        # 读取图片
        img = cv2.imread(input_path + file_name)
        # 裁剪为224×224大小
        img = cv2.resize(img, (224, 224))
        # 输出裁剪后的图片
        cv2.imwrite(output_path + file_name, img)

# 图片裁剪