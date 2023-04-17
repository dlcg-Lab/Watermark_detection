import os
import shutil

# 读取watermark_result.txt文件，将文件名和标注存储到字典中
watermark_dict = {}
with open("C:\\Users\\57411\\Desktop\\watermark_result.txt", "r") as f:
    for line in f:
        filename, label = line.strip().split()
        watermark_dict[filename] = int(label)

# 遍历output_val2017目录中的所有jpg文件，将标注为0的文件保存到0目录中，标注为1的文件保存到1目录中
input_dir = "C:\\Users\\57411\\Desktop\\cut-dowo_val2017"
output_dir_0 = "C:\\Users\\57411\\Desktop\\0"
output_dir_1 = "C:\\Users\\57411\\Desktop\\1"
for filename in os.listdir(input_dir):
    if filename.endswith(".jpg"):
        if filename in watermark_dict:
            label = watermark_dict[filename]
            if label == 0:
                shutil.copy(os.path.join(input_dir, filename), os.path.join(output_dir_0, filename))
            elif label == 1:
                shutil.copy(os.path.join(input_dir, filename), os.path.join(output_dir_1, filename))