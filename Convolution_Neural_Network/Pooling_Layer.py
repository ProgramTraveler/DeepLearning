# 池化层

from mxnet import nd
from mxnet.gluon import nn


def pool2d(X, pool_size, mode='max'):
    p_h, p_w = pool_size
    Y = nd.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w + 1))

    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j] = X[i: i + p_h, j: j + p_w].max()
            elif mode == 'avg':
                Y[i, j] = X[i: i + p_h, j: j + p_w].mean()

    return Y


# 构造二维数组来验证二维最大池化层的输出
x = nd.array([[0, 1, 2],
              [3, 4, 5],
              [6, 7, 8]])
# 默认是求最大池化层输出
print(pool2d(x, (2, 2)))

# 实验一下平均池化层
print(pool2d(x, (2, 2), 'avg'))

# 填充和步幅
# 池化层也可以在输入的高和宽两侧的填充并调整窗口的移动步幅来改变输出的形状
# 前两个分别是批量和通道
x = nd.arange(16).reshape((1, 1, 4, 4))
print(x)

# 通过 nn 模块里的二维最大池化层 MaxPool2D 来演示池化层填充和步幅的工作机制
# 默认情况下 MaxPool2D 实例里步幅和池化窗口形状相同
# 下面使用 (3, 3) 的池化窗口 默认获得形状为 (3, 3) 的实例
pool2d = nn.MaxPool2D(3)
print(pool2d(x))    # 因为池化层没有模型参数 所以不需要调用参数初始化函数



