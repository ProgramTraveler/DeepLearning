import numpy as np
from mxnet import autograd
from d2lzh import Optimization as op, Evaluate_Accuracy as ea

def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, params=None, lr=None
              , trainer=None):
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for x, y in train_iter:
            with autograd.record():
                y_hat = net(x)
                l = loss(y_hat, y).sum()
            l.backward()

            if trainer is None:
                op.sgd(params, lr, batch_size)
            else:
                trainer.step(batch_size)

            # astype 敲错了 搞了半天
            y = y.astype('float32')
            train_l_sum += l.asscalar()
            train_acc_sum += (y_hat.argmax(axis=1) == y).sum().asscalar()
            n += y.size

        test_acc = ea.evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))