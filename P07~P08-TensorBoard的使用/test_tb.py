from torch.utils.tensorboard import SummaryWriter
import numpy_learn as np
from PIL import Image

writer = SummaryWriter("logs")
# img_path = "D:\\code\\python\\土堆Pytorch\\dataset\\train\\ants_image\\0013035.jpg"
# img_path = "D:\\code\\python\\土堆Pytorch\\dataset\\train\\bees_image\\16838648_415acd9e3f.jpg"
img_path = "../dataset/train/bees_image/150013791_969d9a968b.jpg"
img_PIL = Image.open(img_path)
img_array = np.array(img_PIL)

writer.add_image("train",img_array,1,dataformats='HWC')
# writer.add_image()
# y=2x
for i in range(100):
    writer.add_scalar("y=2x",3*i,i)

writer.close()

