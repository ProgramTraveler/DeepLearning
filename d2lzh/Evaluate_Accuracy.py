from mxnet import nd
from mxnet.gluon import data as gdata, loss as gloss, utils as gutils
import mxnet as mx
# def evaluate_accuracy(data_iter, net):
#     acc_sum, n = 0.0, 0
#
#     for x, y in data_iter:
#         y = y.astype('float32')
#         acc_sum += (net(x).argmax(axis=1) == y).sum().asscalar()
#         n += y.size
#
#         return acc_sum / n

# 该函数将被逐步改进
# def evaluate_accuracy(data_iter, net, ctx):
#     acc_sum, n = nd.array([0], ctx=ctx), 0
#     for X, y in data_iter:
#         # 如果 ctx 代表 GPU 及相应的显存 将数据复制到显存上
#         X, y = X.as_in_context(ctx), y.as_in_context(ctx).astype('float32')
#         acc_sum += (net(X).argmax(axis=1) == y).sum()
#         n += y.size
#
#     return acc_sum.asscalar() / n

# 将小批量数据样本 batch 划分并复制到 ctx 变量指定的各个显存
def _get_batch(batch, ctx):
    features, labels = batch
    if labels.dtype != features.dtype:
        labels = labels.astype(features.dtype)
    
    return (gutils.split_and_load(features, ctx),
            gutils.split_and_load((labels, ctx), features.shape[0]))


# 通过辅助函数 _get_batch 使用 ctx 变量包含的所有 GPU 来评价模型
def evaluate_accuracy(data_iter, net, ctx=[mx.cpu()]):
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    acc_sum, n = nd.array([0]), 0
    for batch in data_iter:
        features, labels, _ = _get_batch(batch, ctx)
        for X, y in zip(features, labels):
            y = y.astype('float32')
            acc_sum += (net(X).argmax(axis=1) == y).sum().copyto(mx.cpu())
            n += y.size
        acc_sum.wait_to_read()
    
    return acc_sum.asscalar() / n
