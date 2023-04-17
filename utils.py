import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2
import numpy as np

config = {
    "font.family": 'serif',
    "mathtext.fontset": 'stix',
    "font.serif": ['Times New Roman'],
}

sns.set(style='whitegrid')


def fig_hist(params, loss_hist, metric_hist):
    """
    绘制训练过程中loss值和准确率的变化
    """
    epochs = params["epochs"]

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    sns.lineplot(x=[*range(1, epochs + 1)], y=loss_hist["train"], ax=ax[0], label='loss_hist["train"]')
    sns.lineplot(x=[*range(1, epochs + 1)], y=loss_hist["val"], ax=ax[0], label='loss_hist["val"]')
    sns.lineplot(x=[*range(1, epochs + 1)], y=metric_hist["train"], ax=ax[1], label='metric_hist["train"]')
    sns.lineplot(x=[*range(1, epochs + 1)], y=metric_hist["val"], ax=ax[1], label='metric_hist["val"]')
    plt.title('Convergence History')

    plt.savefig('figs/log.png', dpi=200)


def sub_fig(file_dir_1, file_dir_2, label_1, label_2, label_gt):
    """
    绘制原图和水印图的对比图
    """
    # 读取图片
    image1 = Image.open(file_dir_1)
    image2 = Image.open(file_dir_2)

    # 创建一个有两个子图的窗口
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 5))
    ax1.grid(False)  # 关闭网格线显示
    ax2.grid(False)  # 关闭网格线显示
    ax1.axis('off')  # 隐藏所有坐标轴
    ax2.axis('off')  # 隐藏所有坐标轴

    # 在子图1中显示第一张图片
    ax1.imshow(image1)
    ax1.set_title('Clean Predict outcomes:{}'.format(label_1[0, 0]))

    # 在子图2中显示第二张图片
    ax2.imshow(image2)
    ax2.set_title('Watermark Predict outcomes:{}'.format(label_2[0, 0]))

    # 设置窗口的主标题
    fig.suptitle('Original image & embedded watermark image\n Ground Truth : {} \n'.format(label_gt))

    # 显示窗口
    plt.show()


def img2dct(img):
    # 读取水印图片
    watermark_path = "figs/watermark.png"
    watermark = cv2.imread(watermark_path, cv2.IMREAD_GRAYSCALE)
    watermark = cv2.resize(watermark, (64, 64))

    img = (img * 255).astype('int8')
    # 分离三个通道
    r, g, b = cv2.split(img.transpose(1, 2, 0))

    # 对三个通道做DCT变换
    b_dct = cv2.dct(b.astype(float))
    g_dct = cv2.dct(g.astype(float))
    r_dct = cv2.dct(r.astype(float))

    # 添加隐式水印
    b_dct += watermark * 0.01
    g_dct += watermark * 0.01
    r_dct += watermark * 0.01

    # 反DCT变换
    b_watermarked = cv2.idct(b_dct)
    g_watermarked = cv2.idct(g_dct)
    r_watermarked = cv2.idct(r_dct)

    # 合并三个通道
    watermarked = cv2.merge((r_watermarked, g_watermarked, b_watermarked))

    # 调整rgb图像的形状为（3，64，64）
    rgb_image_resized = watermarked.transpose(2, 0, 1)

    # 将rgb图像转换为numpy数组
    numpy_array = np.array(rgb_image_resized).astype('float') / 255.0

    return numpy_array
