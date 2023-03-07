from mxnet import nd
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
def evaluate_accuracy(data_iter, net, ctx):
    acc_sum, n = nd.array([0], ctx=ctx), 0
    for X, y in data_iter:
        # 如果 ctx 代表 GPU 及相应的显存 将数据复制到显存上
        X, y = X.as_in_context(ctx), y.as_in_context(ctx).astype('float32')
        acc_sum += (net(X).argmax(axis=1) == y).sum()
        n += y.size

    return acc_sum.asscalar() / n