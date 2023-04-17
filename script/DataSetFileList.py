import os
import random
"""
此文件用于列训练集和测试集表（就是dataset目录下test_data.txt和train_data.txt）
"""
main_dir_0 = r"./dataset/0"
main_dir_1 = r"./dataset/1"
script_dir_0 = r"../dataset/0"
script_dir_1 = r"../dataset/1"
output_file_1 = r"../dataset/test_data.txt"
output_file_2 = r"../dataset/train_data.txt"
# 打开输出文件
with open(output_file_1, "w") as f1:
    with open(output_file_2, "w") as f2:
        count = 0
        # 遍历dir_0下的所有.jpg文件
        for filename in os.listdir(script_dir_0):
            if filename.endswith(".jpg"):
                r = random.random()
                if r <= 0.2:
                    path = main_dir_0 + '/' + filename
                    f1.write("{} 0\n".format(path))
                else:
                    path = main_dir_0 + '/' + filename
                    f2.write("{} 0\n".format(path))
                count = 1 - count
        # 遍历dir_1下的所有.jpg文件
        for filename in os.listdir(script_dir_1):
            if filename.endswith(".jpg"):
                r = random.random()
                if r <= 0.2:
                    path = main_dir_1 + '/' + filename
                    f1.write("{} 1\n".format(path))
                else:
                    path = main_dir_1 + '/' + filename
                    f2.write("{} 1\n".format(path))
                count = 1 - count
