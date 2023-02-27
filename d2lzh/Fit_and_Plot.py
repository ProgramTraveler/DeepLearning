# 拟合中的模型定义

from mxnet import autograd, gluon
from mxnet.gluon import data as gdata, loss as gloss, nn
from d2lzh import Semilogy as se

num_epochs, loss = 100, gloss.L2Loss()

def fit_and_plot(train_features, test_features, train_labels, test_labels):
    net = nn.Sequential()
    net.add(nn.Dense(1))
    net.initialize()
    batch_size = min(10, train_labels.shape[0])
    train_iter = gdata.DataLoader(gdata.ArrayDataset(train_features, train_labels), batch_size, shuffle=True)
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate' : 0.01})
    train_ls, test_ls = [], []
    for _ in range(num_epochs):
        for x, y in train_iter:
            with autograd.record():
                l = loss(net(x), y)
            l.backward()
            trainer.step(batch_size)
        train_ls.append(loss(net(train_features), train_labels).mean().asscalar())
        test_ls.append(loss(net(test_features), test_labels).mean().asscalar())
    print('final epoch: train loss', train_ls[-1], 'test loss')
    se.semilogy(range(1, num_epochs + 1), train_ls, 'epoch', 'loss', range(1, num_epochs + 1), test_ls, ['train', 'test'])
    print('weight', net[0].weight.data().asnumpy(), '\nbias:', net[0].bias.data().asnumpy)
