# 权重衰减处理过拟合问题

from mxnet import autograd, gluon, init, nd
from mxnet.gluon import data as gdata, loss as gloss, nn

# 初始化模型参数
def init_param():
        w = nd.random.normal(scale=1, shape=(num_inputs, 1))
        b = nd.zeros(shape=(1, ))
        w.attach_grad()
        b.attach_grad()
        return [w, b]


# 定义 L2 范数惩罚项
def l2_penalty(w):
    # w 是一个矩阵
    return (w ** 2).sum / 2


# 考虑高维线性回归问题 设维度为 200 并特意将训练集设低为 20
n_train, n_test, num_inputs = 20, 100, 200

true_w, true_b = nd.ones((num_inputs, 1)) * 0.01, 0.05

features = nd.random.normal(shape=(n_train + n_test, num_inputs))
labels = nd.dot(features, true_w) + true_b
labels += nd.random.normal(scale=0.01, shape=labels.shape)

train_features, test_features = features[:n_train, :], features[n_train:, :]
train_labels, test_labels = labels[:n_train], labels[n_train:]
