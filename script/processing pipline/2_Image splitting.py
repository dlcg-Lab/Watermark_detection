import os
import cv2


# 定义函数，将一张图片切分成16份并保存
def cut_image(img_path, save_dir):
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    cut_h, cut_w = h // 4, w // 4
    for i in range(4):
        for j in range(4):
            start_h, end_h = i * cut_h, (i + 1) * cut_h
            start_w, end_w = j * cut_w, (j + 1) * cut_w
            cut_img = img[start_h:end_h, start_w:end_w, :]
            cv2.imwrite(os.path.join(save_dir, f"{os.path.splitext(os.path.basename(img_path))[0]}_{i * 4 + j}.jpg"),
                        cut_img)


# 定义路径
input_dir = r"C:\Users\57411\Desktop\output_val2017"
output_dir = r"C:\Users\57411\Desktop\cut-dowo_val2017"
# 创建输出目录
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
# 遍历所有jpg格式图片并切分
for file_name in os.listdir(input_dir):
    if file_name.endswith(".jpg"):
        img_path = os.path.join(input_dir, file_name)
        cut_image(img_path, output_dir)


# 图片切分