import cv2
import os

# 定义读取路径和输出路径
input_path = r'C:\Users\57411\Desktop\cut-dowo_val2017'
output_path = r'C:\Users\57411\Desktop\cut-dowoRGB_val2017'

# 创建输出目录
if not os.path.exists(output_path):
    os.makedirs(output_path)

# 遍历读取路径下的所有jpg文件
for file_name in os.listdir(input_path):
    if file_name.endswith('.jpg'):
        # 读取原始图片
        img = cv2.imread(os.path.join(input_path, file_name))
        # 将图片分为三个单通道图片
        b, g, r = cv2.split(img)
        # 将三个单通道图片输出到指定目录下
        cv2.imwrite(os.path.join(output_path, file_name[:-4] + '_b.jpg'), b)
        cv2.imwrite(os.path.join(output_path, file_name[:-4] + '_g.jpg'), g)
        cv2.imwrite(os.path.join(output_path, file_name[:-4] + '_r.jpg'), r)

# 图片区分通道
