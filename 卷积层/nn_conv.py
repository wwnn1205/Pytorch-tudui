import torch
import torchvision
from torch import nn
from torch.nn import Conv2d, ReLU
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 下载并准备数据
dataset = torchvision.datasets.CIFAR10("../data/dataset", train=False, transform=torchvision.transforms.ToTensor(), download=True)
dataloader = DataLoader(dataset, batch_size=64)

# 构建神经网络
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)
        self.relu1 = ReLU()  # 添加 ReLU 激活函数

    def forward(self, x):
        x = self.conv1(x)
        x_relu = self.relu1(x)  # relu 激活后的结果
        return x, x_relu         # 同时返回 ReLU 前和后的输出

# 初始化模型与 TensorBoard
tudui = Tudui()
writer = SummaryWriter("logs_relu")
step = 0

# 遍历数据集
for data in dataloader:
    imgs, targets = data
    out_raw, out_relu = tudui(imgs)

    # 原始图像
    writer.add_images("input", imgs, step)

    # 卷积后的特征图（未加激活）
    output_raw = torch.reshape(out_raw, (-1, 3, 30, 30))  # 将6通道变3通道以便可视化
    writer.add_images("conv_only", output_raw, step)

    # 加了 ReLU 后的特征图
    output_relu = torch.reshape(out_relu, (-1, 3, 30, 30))
    writer.add_images("conv_relu", output_relu, step)

    step += 1
    if step == 10:  # 可视化前10组样本就够了
        break

writer.close()
