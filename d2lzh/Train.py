import mxnet as mx
from mxnet import autograd
from mxnet.gluon import utils as gutils
import time
from d2lzh import Evaluate_Accuracy as ea

# 将小批量数据样本 batch 划分并复制到 ctx 变量指定的各个显存
def _get_batch(batch, ctx):
    features, labels = batch
    if labels.dtype != features.dtype:
        labels = labels.astype(features.dtype)
    
    return (gutils.split_and_load(features, ctx),
            gutils.split_and_load(labels, ctx), features.shape[0])
# 定义 train 函数使用多 GPU 训练并评价模型

def train(train_iter, test_iter, net, loss, trainer, ctx, num_epochs):
    print('training on', ctx)
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, m, start = 0.0, 0.0, 0, 0, time.time()
        for i, batch in enumerate(train_iter):
            Xs, ys, batch_size = _get_batch(batch, ctx)
            with autograd.record():
                y_hats = [net(x) for x in Xs]
                ls = [loss(y_hat, y) for y_hat, y in zip(y_hats, ys)]
            for l in ls:
                l.backward()
            trainer.step(batch_size)
            train_l_sum += sum([l.sum().asscalar() for l in ls])
            n += sum([l.size for l in ls])
            train_acc_sum += sum([(y_hat.argmax(axis=1 == y).sum().asscalar()
                                   for y_hat, y in zip(y_hats, ys))])
            m += sum([y.size for y in ys])
        test_acc = ea.evaluate_accuracy(test_iter, net, ctx)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f,'
              'time % .1f sec'
              % (epoch + 1, train_l_sum / n, train_acc_sum / m, test_acc, time.time() - start))