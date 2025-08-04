# 导入必要的库
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim

# 定义简单的模型
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(784, 10)

    def forward(self, x):
        return self.fc(x)

# 创建一个模型实例
model = SimpleModel()

# 创建一个优化器
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 创建一个损失函数
criterion = nn.CrossEntropyLoss()

# 创建 TensorBoardX 的 SummaryWriter
writer = SummaryWriter(log_dir='./runs/experiment')

# 模拟一些训练过程
for epoch in range(10):
    # 假设这里有训练数据
    inputs = torch.randn(32, 784)  # 假设32个样本，784维的向量（例如MNIST）
    targets = torch.randint(0, 10, (32,))  # 目标是0到9的标签

    # 清零梯度
    optimizer.zero_grad()

    # 前向传播
    outputs = model(inputs)

    # 计算损失
    loss = criterion(outputs, targets)

    # 反向传播
    loss.backward()

    # 更新参数
    optimizer.step()

    # 记录训练的损失值
    writer.add_scalar('Loss/train', loss.item(), epoch)

    # 记录模型的参数分布
    for name, param in model.named_parameters():
        writer.add_histogram(name, param, epoch)

    print(f"Epoch [{epoch+1}/10], Loss: {loss.item():.4f}")

# 关闭 writer
writer.close()
