import os
import cv2


def is_three_channel_image(file_path):
    """
    检查给定的图像文件是否是三通道的图像。
    """
    img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    return len(img.shape) == 3 and img.shape[2] == 3


def check_directory(directory_path):
    """
    检查给定目录下的所有图像文件是否都是三通道的图像。
    """
    for file_name in os.listdir(directory_path):
        if file_name.endswith('.jpg') or file_name.endswith('.png'):
            file_path = os.path.join(directory_path, file_name)
            if not is_three_channel_image(file_path):
                print(f"File {file_path} is not a three-channel image.")
                os.remove(file_path)


check_directory('../dataset/1')
