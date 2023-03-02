# 图像分类

from d2lzh import Label_to_File as ltf, Show_Fashion as sf
# data 包读取数据
from mxnet.gluon import data as gdata
import sys
import time
import matplotlib.pyplot as plt

# 通过参数 train 来指定获取训练数据集或测试数据集
# 测试数据只用来评价模型的表现 并不能用来训练模型
# FashionMNIST 只是一个数据集
mnist_train = gdata.vision.FashionMNIST(train=True)
mnist_test = gdata.vision.FashionMNIST(train=False)

# 因为有 10 个类别 训练集和测试集的每个类别的图像数分别为 6000 和 1000
print(len(mnist_train), len(mnist_test))

# 可以通过方括号 [] 来访问任意一个样本
# 获取第一个样本的图像和标签
feature, label = mnist_train[0]

# feature 对应高和宽均为 28 像素的图像 使用三维的NDArray 存储 最后一位是通道数 因为数据集中是灰度图像 所以通道数为 1
print(feature.shape, feature.dtype)

# 图像的标签使用 Numpy 的标量表示 它的类型为 32 为整数
print(label, type(label),  label.dtype)

# 显示数据集中前 9 个样本的图像内容和文本标签
x, y = mnist_train[0 : 9]
sf.show_fashion_mnist(x, ltf.get_fashion_mnist_labels(y))
plt.show()

# 读取小批量
batch_size = 256

# DataLoader 允许使用多进程来加速数据读取 (暂时不支持 Windows 操作系统)
transformer = gdata.vision.transforms.ToTensor
if sys.platform.startswith('win'):
    # 通过参数num_workers 来设置 4 个进程读取数据
    # 0 表示不用额外的进程来加速读取数据
    num_workers = 0
else:
    num_workers = 4

train_iter = gdata.DataLoader(mnist_train.transform_first(transformer), batch_size, shuffle=True)
test_iter = gdata.DataLoader(mnist_test.transform_first(transformer), batch_size, shuffle=False,
                             num_workers=num_workers)
