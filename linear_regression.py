# 线性回归从零开始
# matplotlib 用于作图

from IPython import display
from matplotlib import pyplot as plt
from mxnet import autograd, nd
import random

# 输入的特征数为 2
num_inputs = 2
# 训练数据集样本1数为 1000
num_example = 1000
# 使用线性回归模型真实权重
true_w = [2, -3, 4]
# 偏差 b = 4.2
true_b = 4.2
# 每一行是一个长度为 2 的向量
features = nd.random.normal(scale=1, shape=(num_example, num_inputs))
# 每一行是一个长度为 1 的向量(标量)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b








