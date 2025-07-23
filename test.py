from torchvision import transforms
from PIL import Image
import torch

# 加载图像
image = Image.open('./dataset/train/ants_image/5650366_e22b7e1065.jpg')

# 将图像转换为 tensor
transform = transforms.ToTensor()
tensor_image = transform(image)



print("--- Tensor 信息 ---")
print("Tensor 形状 (Shape):", tensor_image.shape)
print("Tensor 数据类型 (Dtype):", tensor_image.dtype)
print("Tensor 设备 (Device):", tensor_image.device)
print("Tensor 最小值 (Min Value):", tensor_image.min())
print("Tensor 最大值 (Max Value):", tensor_image.max())

# 计算每个通道的均值和标准差
mean_per_channel = tensor_image.mean(dim=[1, 2])
std_per_channel = tensor_image.std(dim=[1, 2])
print("每个通道的均值 (Mean per channel):", mean_per_channel)
print("每个通道的标准差 (Std per channel):", std_per_channel)
print("Tensor 总元素数量 (Numel):", tensor_image.numel())

