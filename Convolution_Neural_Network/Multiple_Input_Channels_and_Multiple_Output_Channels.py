# 多输入通道和多输出通道
from mxnet import nd
from d2lzh import Corr2d as c2

# 多个输入通道的互相关运算
def corr2d_multi_in(X, K):
    # 首先沿着 x 和 k 的第 0 维 (通道维) 遍历 然后使用 * 将结果列表变为 add_n 函数的位置参数
    # (positional argument) 来进行相加
    return nd.add_n(*[c2.corr2d(x, k) for x, k in zip(X, K)])


# 互相关运算来计算多个通道的输出
def corr2d_multi_in_out(X, K):
    # 对 k 的第 0 维遍历 每次同输入 x 做互相关计算 所有结果使用 stack 函数合并在一起
    return nd.stack(*[corr2d_multi_in(X, k) for k in K])

# 使用全连接层中的矩阵乘法来实现 1 * 1 卷积
def corr2d_multi_in_out_1x1(x, k):
    c_i, h, w = x.shape
    c_o = k.shape[0]
    x = x.reshape((c_i, h * w))
    k = k.reshape((c_o, c_i))
    # 全连接层的矩阵乘法
    y = nd.dot(k, x)
    return y.reshape((c_o, h, w))


# x = nd.array([[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
#              [[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
# k = nd.array([[[0, 1], [2, 3]], [[1, 2], [3, 4]]])
#
# print(corr2d_multi_in(x, k))
#
# k = nd.stack(k, k + 1, k + 2)
# print(k.shape)
#
# # 对输入数组 x 和 核数组 k 做互相关运算 此时的输出含有 3 个通道
# print(corr2d_multi_in_out(x, k))

# 1 * 1 卷积层
X = nd.random.uniform(shape=(3, 3, 3))
K = nd.random.uniform(shape=(2, 3, 1, 1))
y1 = corr2d_multi_in_out_1x1(X, K)
y2 = corr2d_multi_in_out(X, K)

print((y1 - y2).norm().asscalar() < 1e-6)