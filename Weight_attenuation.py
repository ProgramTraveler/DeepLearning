# 权重衰减处理过拟合问题

from mxnet import autograd, gluon, init, nd
from mxnet.gluon import data as gdata, loss as gloss, nn
from d2lzh import MatrixMultiplication as ma, Loss as lo, Optimization as op, Semilogy as se

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
    return (w ** 2).sum() / 2

def fit_and_plot(lambd):
    w, b = init_param()
    train_ls, test_ls = [], []
    for _ in range(num_epochs):
        for X, y in train_iter:
            with autograd.record():
                # 添加了 L2 范数的惩罚项 广播机制使其变成长度为 batch_size 的向量
                l = loss(net(X, w, b), y) + lambd * l2_penalty(w)
            l.backward()
            op.sgd([w, b], lr, batch_size)

        train_ls.append(loss(net(train_features, w, b), train_labels).mean().asscalar())
        test_ls.append(loss(net(test_features, w, b), test_labels).mean().asscalar())

    se.semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'loss',
                range(1, num_epochs + 1), test_ls, ['train', 'test'])

    print('L2 norm of w:', w.norm().asscalar())


# 考虑高维线性回归问题 设维度为 200 并特意将训练集设低为 20
n_train, n_test, num_inputs = 20, 100, 200

true_w, true_b = nd.ones((num_inputs, 1)) * 0.01, 0.05

features = nd.random.normal(shape=(n_train + n_test, num_inputs))
labels = nd.dot(features, true_w) + true_b
labels += nd.random.normal(scale=0.01, shape=labels.shape)

train_features, test_features = features[:n_train, :], features[n_train:, :]
train_labels, test_labels = labels[:n_train], labels[n_train:]

# 定义训练和测试 这里计算最终损失函数时添加了 L2 范数惩罚项
batch_size, num_epochs, lr = 1, 100, 0.003

net, loss = ma.linreg, lo.squared_loss

train_iter = gdata.DataLoader(gdata.ArrayDataset(train_features, train_labels), batch_size, shuffle=True)

# 当 lambd 设为 0 时 没有使用权重衰减 过拟合
fit_and_plot(lambd=0)

