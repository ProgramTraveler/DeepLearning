# 多项式函数的拟合实验

from mxnet import autograd, gluon, nd
from mxnet.gluon import data as gdata, loss as gloss, nn
from d2lzh import Fit_and_Plot as fp

# 使用一个三阶多项式函数来来生成该样本的标签

# 训练数据集和测试数据集的样本数都设为 100
n_train, n_test, true_w, true_b = 100, 100, [1.2, -3.4, 5.6], 5

features = nd.random.normal(shape=(n_train + n_test, 1))
poly_features = nd.concat(features, nd.power(features, 2), nd.power(features, 3))
# 三阶多项式
labels = (true_w[0] * poly_features[:, 0] + true_w[1] * poly_features[:, 1] + true_w[2] * poly_features[:, 2] + true_b)
# 噪声项服从均值为 0 标准差为 0.1 的正态分布
labels += nd.random.normal(scale=0.1, shape=labels.shape)

# 生成的数据集的前两个样本
# print(features[:2], poly_features[:2], labels[:2])

# 定义 训练和测试模型
# 虽然图像正常显示 但是和书上的图像不一样 train 和 test 是反的
# 三阶多项式函数拟合(正常)
fp.fit_and_plot(poly_features[:n_train, :], poly_features[n_train:, :], labels[:n_train], labels[n_train:])

# 线性函数拟合(欠拟合)
fp.fit_and_plot(features[:n_train, :], features[n_train:, :], labels[:n_train], labels[n_train:])

# 训练样本不足(过拟合) 只使用两个样本来训练模型
fp.fit_and_plot(poly_features[0:2, :], poly_features[n_train:, :], labels[0:2], labels[n_train:])