import torch
import torchvision
from torch import nn
from torch.nn import Linear
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("../data/dataset",train=False,transform=torchvision.transforms.ToTensor(),download=True)
dataloader = DataLoader(dataset, batch_size=64,drop_last=True)

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.linear1 = Linear(196608,10)

    def forward(self, input):
        output = self.linear1(input)
        return output

tudui = Tudui()
writer = SummaryWriter("logs")
step = 0

for data in dataloader:
    imgs, targets = data
    print(imgs.shape)
    writer.add_images("input", imgs, step)
    output = torch.reshape(imgs,(1,1,1,-1)) # 方法一：拉平
    print(output.shape)
    output = tudui(output)
    print(output.shape)
    writer.add_images("output", output, step)
    step = step + 1