import sys
import numpy as np  # linear algebra
import copy
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn
from tqdm import tqdm
from torch import optim
from torch_dataset import get_dataset
from torch.utils.data import DataLoader
from models import Network
from torchsummary import summary
from utils import fig_hist
from PIL import Image
from utils import sub_fig
from utils import img2dct

# 定义运行设备,如果有gpu环境就在gpu上跑,否则在cpu上跑
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义模型参数
params_model = {
    # 图像形状
    "shape_in": (3, 64, 64),
    # 模型中卷积核个数
    "initial_filters": 8,
    # 线性变换层中神经元个数
    "num_fc1": 100,
    # dropout率
    "dropout_rate": 0.25,
    # 分类数
    "num_classes": 2,
    # 保存路径
    "weight_path": "weights/weights.pt"
}

params_train = {
    # 训练集和测试集
    "train": None,
    "val": None,
    # 训练轮数
    "epochs": 100,
    # 优化器
    "optimiser": None,
    # 定义学习率调度器
    "lr_change": None,
    # 损失函数
    "f_loss": nn.NLLLoss(reduction="sum"),
    # 定义保存路径
    "weight_path": "weights/weights.pt",
}


def get_lr(opt):
    # 获得学习率
    for param_group in opt.param_groups:
        return param_group['lr']


def loss_epoch(model, loss_func, dataset_dl, opt=None):
    # 一个batch一个batch的训练
    run_loss = 0.0
    t_metric = 0.0
    len_data = len(dataset_dl.dataset)
    # internal loop over dataset
    for xb, yb in dataset_dl:
        # move batch to device
        # 转tensor
        xb = torch.as_tensor(xb)
        yb = torch.as_tensor(yb)
        xb = xb.to(device)
        yb = yb.to(device)
        # 输入模型训练
        output = model(xb)  # get model output
        # 这个batch的损失值
        loss_b, metric_b = loss_batch(loss_func, output, yb, opt)  # get loss per batch
        run_loss += loss_b  # update running loss

        if metric_b is not None:  # update running metric
            t_metric += metric_b

    loss = run_loss / float(len_data)  # average loss value
    metric = t_metric / float(len_data)  # average metric value

    return loss, metric


def loss_batch(loss_func, output, target, opt=None):
    loss = loss_func(output, target)  # get loss
    pred = output.argmax(dim=1, keepdim=True)  # Get Output Class
    metric_b = pred.eq(target.view_as(pred)).sum().item()  # get performance metric

    # 计算损失
    loss = loss_func(output, target)

    # 获取预测的类别
    pred = output.argmax(dim=1, keepdim=True)

    # 计算性能指标
    metric_b = pred.eq(target.view_as(pred)).sum().item()

    # 如果提供了优化器，则进行梯度下降
    if opt is not None:
        opt.zero_grad()
        loss.backward()
        opt.step()

    return loss.item(), metric_b


def train_val(model, params, verbose=False):
    # Get the parameters
    epochs = params["epochs"]
    loss_func = params["f_loss"]
    opt = params["optimiser"]
    train_dl = params["train"]
    val_dl = params["val"]
    lr_scheduler = params["lr_change"]
    weight_path = params["weight_path"]

    # loss_history和metric_history用于绘图
    loss_history = {"train": [], "val": []}
    metric_history = {"train": [], "val": []}
    # 创建一个模型的状态字典的深拷贝，以便在训练神经网络时进行参数更新。
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    ''' Train Model n_epochs '''

    for epoch in tqdm(range(epochs)):

        ''' Get the Learning Rate '''
        current_lr = get_lr(opt)
        if verbose:
            print('Epoch {}/{}, current lr={}'.format(epoch, epochs - 1, current_lr))

        '''

        Train Model Process

        '''

        model.train()
        train_loss, train_metric = loss_epoch(model, loss_func, train_dl, opt)

        # collect losses
        loss_history["train"].append(train_loss)
        metric_history["train"].append(train_metric)

        '''

        Evaluate Model Process

        '''

        model.eval()
        with torch.no_grad():
            val_loss, val_metric = loss_epoch(model, loss_func, val_dl)

        # store best model
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())

            # store weights into a local file
            torch.save(model.state_dict(), weight_path)
            if verbose:
                print("Copied best model weights!")

        # collect loss and metric for validation dataset
        loss_history["val"].append(val_loss)
        metric_history["val"].append(val_metric)

        # learning rate schedule
        lr_scheduler.step(val_loss)
        if current_lr != get_lr(opt):
            if verbose:
                print("Loading best model weights!")
            model.load_state_dict(best_model_wts)

        if verbose:
            print(f"train loss: {train_loss:.6f}, dev loss: {val_loss:.6f}, accuracy: {100 * val_metric:.2f}")
            print("-" * 10)

            # load best model weights
    model.load_state_dict(best_model_wts)

    return model, loss_history, metric_history


def inference(model, dataset, device, num_classes=2):
    # 样本数
    len_data = len(dataset)
    # 存放输出结果
    y_out = torch.zeros(len_data, num_classes)
    # 存放真实值
    y_gt = np.zeros(len_data, dtype="uint8")
    # 存放输出的标号
    y_pred = np.zeros(len_data, dtype="uint8")
    model = model.to(device)  # move model to device

    # 不反向传播
    with torch.no_grad():
        for i in tqdm(range(len_data)):
            # 取第i个样本
            x, y = dataset[i]
            # y是真实值
            y_gt[i] = y
            # 输入模型预测
            y_out[i] = model(x.unsqueeze(0).to(device))
            # 找出第1维度上最大值的索引(分类结果)
            pred = y_out[i].argmax(dim=0, keepdim=True)  # Get Output Class
            y_pred[i] = pred.numpy()
    return y_pred, y_gt


def main_training():
    # 训练函数
    # 获得数据集
    train_set, val_set = get_dataset(params_model=params_model)

    # 训练集的dataloader
    train_dl = DataLoader(train_set,
                          batch_size=32,
                          shuffle=True)

    # 测试集的dataloader
    val_dl = DataLoader(val_set,
                        batch_size=32,
                        shuffle=True)

    # 定义模型结构
    cnn_model = Network(params_model).to(device=device)
    # 定义优化器
    opt = optim.Adam(cnn_model.parameters(), lr=3e-4)
    params_train['train'] = train_dl
    params_train['val'] = val_dl
    params_train['optimiser'] = optim.Adam(cnn_model.parameters(),
                                           lr=3e-4)
    params_train['lr_change'] = ReduceLROnPlateau(opt,
                                                  mode='min',
                                                  factor=0.5,
                                                  patience=20,
                                                  verbose=False)
    # 打印模型结构
    summary(cnn_model, input_size=params_model['shape_in'], device=device.type)
    # 开始训练
    cnn_model, loss_hist, metric_hist = train_val(cnn_model, params_train)
    # 绘图并保存在figs目录下
    fig_hist(params=params_train, loss_hist=loss_hist, metric_hist=metric_hist)
    # 测试刚刚训练的模型在测试集上的准确率
    main_testing()


def main_testing():
    # 测试函数
    train_set, val_set = get_dataset(params_model=params_model)
    print(len(val_set), 'samples found')

    # 定义模型结构
    cnn_model = Network(params_model)
    # 加载权重文件
    cnn_model.load_state_dict(torch.load(params_model["weight_path"]))

    # 预测,返回预测结果和真实值
    y_pred, y_gt = inference(model=cnn_model, dataset=val_set, device=device, num_classes=2)
    acc_num = 0
    for index in range(len(val_set)):
        # 如果两者相等,正确的个数+1
        if y_pred[index] == y_gt[index]:
            acc_num += 1
    # 正确的个数除以总的样本数,等于准确率
    print('Test accuracy on adversarial examples: %0.4f%%\n' % ((acc_num * 100) / len(val_set)))


def n2w(index=0):
    # 获得数据集
    train_set, val_set = get_dataset(params_model=params_model)
    # 设置展示第几个样本

    # 定义网络结构
    cnn_model = Network(params_model)
    # 加载权重文件
    cnn_model.load_state_dict(torch.load(params_model["weight_path"]))

    model = cnn_model.to(device)  # move model to device
    # 取第index个样本
    img1, _ = val_set[index]
    img1 = np.array(img1)
    # 设置不反向传播
    with torch.no_grad():
        x, y = val_set[index]
        y_gt = y
        y_out = model(x.unsqueeze(0).to(device))
        # 找出第1维度上最大值的索引(分类结果)
        pred = y_out.argmax(dim=1, keepdim=True)  # Get Output Class
        y_pred_0 = pred.cpu().numpy()

    # 给图片加上水印
    img2 = img2dct(img1)

    # 设置不反向传播
    with torch.no_grad():
        y_out = model(torch.tensor(img2).to(device).to(torch.float))
        # 找出第1维度上最大值的索引(分类结果)
        pred = y_out.argmax(dim=1, keepdim=True)  # Get Output Class
        y_pred_1 = pred.cpu().numpy()

    # png图片保存路径
    file_dir_1 = './tmp/img1.png'
    file_dir_2 = './tmp/img2.png'

    img1 = (img1 * 255).astype('uint8')
    img2 = (img2 * 255).astype('uint8')
    pil_img1 = Image.fromarray(np.transpose(img1, (1, 2, 0)))
    pil_img2 = Image.fromarray(np.transpose(img2, (1, 2, 0)))

    # 保存图片
    pil_img1.save(file_dir_1)
    pil_img2.save(file_dir_2)

    # 将两张图片以子图的形式展示
    sub_fig(file_dir_1, file_dir_2, label_1=y_pred_0, label_2=y_pred_1, label_gt=y_gt)


if __name__ == "__main__":
    if sys.argv[1] == 'train':
        main_training()
    elif sys.argv[1] == 'test':
        main_testing()
    elif sys.argv[1] == 'n2w':
        n2w(index=100)
