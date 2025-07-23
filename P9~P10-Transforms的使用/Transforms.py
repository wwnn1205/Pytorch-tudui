from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from torchvision import transforms

class MyData(Dataset):

    def __init__(self, root_dir, image_dir, label_dir, transform=None):
        # 构造函数，在创建数据集实例时被调用

        self.root_dir = root_dir # 数据集的根目录，例如 "dataset/train"
        self.image_dir = image_dir # 图像文件夹相对于 root_dir 的路径，例如 "ants_image"
        self.label_dir = label_dir # 标签文件夹相对于 root_dir 的路径，例如 "ants_label"

        # 构建完整的标签文件路径和图像文件路径
        self.label_path = os.path.join(self.root_dir, self.label_dir)
        self.image_path = os.path.join(self.root_dir, self.image_dir)

        # 获取图像文件夹和标签文件夹中所有文件的列表
        self.image_list = os.listdir(self.image_path)
        self.label_list = os.listdir(self.label_path)

        self.transform = transform # 存储图像变换（transforms），可选参数

        # 因为label 和 Image文件名相同，进行一样的排序，可以保证取出的数据和label是一一对应的
        # 这一步非常重要！确保图像和其对应的标签能够正确匹配。
        self.image_list.sort()
        self.label_list.sort()

    def __getitem__(self, idx):
        # 这个方法定义了如何根据索引 (idx) 获取单个数据样本
        # 当你使用 dataset[idx] 这样的方式访问时，这个方法会被调用

        img_name = self.image_list[idx]     # 获取当前索引对应的图像文件名
        label_name = self.label_list[idx]   # 获取当前索引对应的标签文件名（假设与图像文件名相同）

        # 构建完整的图像和标签文件的路径
        img_item_path = os.path.join(self.root_dir, self.image_dir, img_name)
        label_item_path = os.path.join(self.root_dir, self.label_dir, label_name)

        # 使用 PIL 库打开图像文件
        img = Image.open(img_item_path)

        # 打开标签文件并读取第一行内容作为标签
        # 这里假设标签文件是文本文件，每行包含一个标签
        with open(label_item_path, 'r') as f:
            label = f.readline()

        # 如果在初始化时传入了 transform，则对图像应用这些变换
        if self.transform:
            img = self.transform(img) # 注意这里是 self.transform，而不是函数外的 transform

        # 返回处理后的图像和标签
        return img, label

    def __len__(self):
        # 这个方法定义了数据集的长度，即数据集中样本的总数
        # 当你使用 len(dataset) 时，这个方法会被调用

        # 断言（assert）确保图像列表和标签列表的长度相同，
        # 如果不相同则程序会报错，提示数据不一致
        assert len(self.image_list) == len(self.label_list)
        return len(self.image_list) # 返回图像（或标签）列表的长度


transform = transforms.Compose([transforms.Resize(400), transforms.ToTensor()])
root_dir = "../dataset/train"
image_ants = "ants_image"
label_ants = "ants_label"
ants_dataset = MyData(root_dir, image_ants, label_ants, transform=transform)
image_bees = "bees_image"
label_bees = "bees_label"
bees_dataset = MyData(root_dir, image_bees, label_bees, transform=transform)