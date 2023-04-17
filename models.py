import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def findConv2dOutShape(hin, win, conv, pool=2):
    """
    计算指定形状的输入经过卷积层后的输出形状
    """
    # get conv arguments
    kernel_size = conv.kernel_size
    stride = conv.stride
    padding = conv.padding
    dilation = conv.dilation

    hout = np.floor((hin + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1)
    wout = np.floor((win + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1)

    if pool:
        hout /= pool
        wout /= pool
    return int(hout), int(wout)


class Network(nn.Module):

    # Network Initialisation
    def __init__(self, params):
        """
         定义模型结构
        """
        super(Network, self).__init__()
        # 定义输入形状
        Cin, Hin, Win = params["shape_in"]
        # 定义卷积核个数
        init_f = params["initial_filters"]
        # 定义全连接层中神经元个数
        num_fc1 = params["num_fc1"]
        # 定义分类数
        num_classes = params["num_classes"]
        # 定义dropout率
        self.dropout_rate = params["dropout_rate"]

        # Convolution Layers
        self.conv1 = nn.Conv2d(Cin, init_f, kernel_size=3)
        h, w = findConv2dOutShape(Hin, Win, self.conv1)
        self.conv2 = nn.Conv2d(init_f, 2 * init_f, kernel_size=3)
        h, w = findConv2dOutShape(h, w, self.conv2)
        self.conv3 = nn.Conv2d(2 * init_f, 4 * init_f, kernel_size=3)
        h, w = findConv2dOutShape(h, w, self.conv3)
        self.conv4 = nn.Conv2d(4 * init_f, 8 * init_f, kernel_size=3)
        h, w = findConv2dOutShape(h, w, self.conv4)

        # compute the flatten size
        self.num_flatten = h * w * 8 * init_f
        # 定义线性变换层
        self.fc1 = nn.Linear(self.num_flatten, num_fc1)
        self.fc2 = nn.Linear(num_fc1, num_classes)

    def forward(self, X):
        X = F.relu(self.conv1(X))
        # 定义最大池化层
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv3(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv4(X))
        X = F.max_pool2d(X, 2, 2)

        # 调整张量形状
        X = X.view(-1, self.num_flatten)
        # 定义激活函数
        X = F.relu(self.fc1(X))
        # 定义dropout函数
        X = F.dropout(X, self.dropout_rate)
        X = self.fc2(X)
        return F.log_softmax(X, dim=1)
