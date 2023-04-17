# 介绍
本项目使用神经网络检测图像中的隐式水印
### 作者
[Yuyao Ge](https://github.com/GeYuYao-hub) : 神经网络代码构建, 神经网络的训练, 架构设计
[Lizhe Chen](https://github.com/574118090) : 图像处理, 水印嵌入, 数据集构建

# 需要安装的库

* numpy 
* opencv-python
* pytorch
* tqdm
* torchsummary
* torchvision
* matplotlib

# 文件说明

### main_gpu.py

核心函数，有三个功能分别是训练，测试，可视化。

* 训练：将clean的图像以及label输入神经网络进行训练，训练结束后日志会可视化保存在figs目录下并运行测试功能，用于检测模型在测试集上的准确率。
* 测试：将测试集（无logo）输入模型进行测试，用来测试模型在测试集上的准确率。
* 可视化：首先将原图输入神经网络进行测试，输出一个label，再将dct变换后的图像输入神经网络进行测试，最后将两个图可视化出来。

### models.py

定义了深度学习模型，模型结构如下（在训练的过程中，也会打印模型结构）。

```txt
       Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [-1, 8, 62, 62]             224
            Conv2d-2           [-1, 16, 29, 29]           1,168
            Conv2d-3           [-1, 32, 12, 12]           4,640
            Conv2d-4             [-1, 64, 4, 4]          18,496
            Linear-5                  [-1, 100]          25,700
            Linear-6                    [-1, 2]             202
================================================================
Total params: 50,430
Trainable params: 50,430
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.05
Forward/backward pass size (MB): 0.38
Params size (MB): 0.19
Estimated Total Size (MB): 0.62
----------------------------------------------------------------
```

### torch_dataset.py

用于读取数据集

### utils.py

一些工具，包括绘图工具和可视化工具，以及dct变换工具

# 目录说明

### dataset

用于保存处理好的数据集，目录下有test_data.txt和train_data.txt表示测试集路径及标签和训练集路径及标签

###  figs

用于保存训练过程中日志可视化图像和水印原图

### script

用于保存项目过程中全部的脚本文件

### tmp

用于存储中间文件,注意每次执行可视化功能时，展示的两个图像就保存在这个目录下。因此这个目录不可以删，但是目录下的img1和img2可以删，每次运行都会再生成。

### weights

用于存储训练好的模型权重文件
# 功能展示

### 训练功能

训练中，打印训练集和测试集样本数量，打印模型结构，显示进度条

![](D:/Project/Watermark_detection/figs/2.png)

训练结束，执行测试，测试结束后打印测试集上的准确率

![](D:/Project/Watermark_detection/figs/3.png)

日志可视化保存在figs目录下

![](D:/Project/Watermark_detection/figs/log.png)

### 测试

测试结果如下

![](D:/Project/Watermark_detection/figs/4.png)

### 可视化

可以在此处选择想要可视化的样本

![](D:/Project/Watermark_detection/figs/5.png)

可视化效果如下

![](D:/Project/Watermark_detection/figs/1.png)

大标题ground truth表示这个图像原本的标签为多少

左侧小标题的outcomes表示干净图像在神经网络中预测的label

右侧小标题的outcomes表示打上logo的图像在神经网络中预测的label