# 定义模型 dot() 函数做矩阵乘法
# 在 python 中的 函数定义好像和 c++ 中不太一样 在哪儿都能定义

from mxnet import nd


def linreg(X, w, b):
    return nd.dot(X, w) + b