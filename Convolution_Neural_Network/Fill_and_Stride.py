# 填充和步幅

from mxnet import nd
from mxnet.gluon import nn

# 定义一个函数来计算卷积层 它初始化卷积层权重 并对输入和输出做相应的升维和降维
def comp_conv2d(conv2d, x):
    conv2d.initialize()
    # (1, 1) 代表批量大小和通道数均为 1
    x = x.reshape((1, 1) + x.shape)
    y = conv2d(x)
    # 排除不关心的前两维 批量和通量
    return y.reshape(y.shape[2:])


# 注意这里是两侧分别填充 1 行或列 所以在两侧一共填充 2 行或列
conv2d = nn.Conv2D(1, kernel_size=3, padding=1)
x = nd.random.uniform(shape=(8, 8))
print(comp_conv2d(conv2d, x).shape)

# 当卷积核的高和宽不同时 可以通过设置高和宽上不同的填充数使输出和输入具有相同的高和宽
# 使用高为 5 宽为 3 的卷积核 在高和宽两侧的填充分别为 2 和 1
conv2d = nn.Conv2D(1, kernel_size=(5, 3), padding=(2, 1))
print(comp_conv2d(conv2d, x).shape)

# 步幅 每次滑动的行数和列数
# 令高和宽的步幅均为 2 使输入的高和宽减半
conv2d = nn.Conv2D(1, kernel_size=3, padding=1, strides=2)
print(comp_conv2d(conv2d, x).shape)

# 复杂一点的例子
conv2d = nn.Conv2D(1, kernel_size=(3, 5), padding=(0, 1), strides=(3, 4))
print(comp_conv2d(conv2d, x).shape)