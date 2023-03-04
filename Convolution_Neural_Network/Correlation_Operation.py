# 卷积神经二维互相关运算

from mxnet import autograd, nd
from mxnet.gluon import nn

def corr2d(x, k):
    h, w = k.shape

    y = nd.zeros((x.shape[0] - h + 1, x.shape[1] - w + 1))
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            y[i, j] = (x[i: i + h, j: j + w] * k).sum()

    return y


# 自定义二维卷积层
# 二维卷积层将输入和卷积核做互相关运算 并加上一个标量偏差来得到输出
# 卷积模型参数包括卷积核核标量偏差
class Conv2D(nn.Block):
    def __init__(self, kernel_size, **kwargs):
        super(Conv2D, self).__init__(**kwargs)
        self.weight = self.params.get('weight', shape=kernel_size)
        self.bias = self.params.get('bias', shape=(1,))

    def forward(self, x):
        return corr2d(x, self.weight.data()) + self.bias.data()


x = nd.array([[0, 1, 2],
              [3, 4, 5],
              [6, 7, 8]])
k = nd.array([[0, 1], [2, 3]])

print(corr2d(x, k))

# 图像中物体边缘检测
# 构造一张 6 * 8 的图像
x = nd.ones((6, 8))
x[:, 2 : 6] = 0
print(x)

# 构造一个高和宽分别为 1 和 2 的卷积核
# 横向相邻元素相同 输出为 0 否则输出为 1
k = nd.array([[1, -1]])
y = corr2d(x, k)
# 卷积层可以通过重复使用卷积核有效地表征局部空间
print(y)

# 通过数据学习核数组
# 构造一个卷积层 将其卷积核初始化为随机数组
# 使用平方差误差来比较 y 和卷积层的输出
# 构造一个输出通道数为 1 核数组形状是 (1, 2) 的二维卷积层
# 虽然之前构造一个 Conv2D 类 但是用于 corr2d 使用了对单个元素赋值的操作因而无法自动求梯度
# 下面使用 Gluon 提供的 Conv2D 类来实现这个例子
conv2d = nn.Conv2D(1, kernel_size=(1, 2))
conv2d.initialize()

# 二维卷积层使用了 4 维输入输出 格式为 (样本, 通道, 高, 宽)
# 这里批量大小 (批量中的样本数) 和通道数均为 1
x = x.reshape((1, 1, 6, 8))
y = y.reshape((1, 1, 6, 7))

for i in range(10):
    with autograd.record():
        y_hat = conv2d(x)
        l = (y_hat - y) ** 2
    l.backward()
    # 简单起见 这里忽略了偏差
    conv2d.weight.data()[:] -= 3e-2 * conv2d.weight.grad()
    if (i + 1) % 2 == 0:
        print('batch %d, loss %.3f' % (i + 1, l.sum().asscalar()))

# 学习到的核数组
print(conv2d.weight.data().reshape((1, 2)))