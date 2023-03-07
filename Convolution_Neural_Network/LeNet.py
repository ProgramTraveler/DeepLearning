# 卷积神经网络 LeNet
# 调用os，sys模块

import sys
sys.path.append('/root/Deep_Pro/')


import mxnet as mx
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import loss as gloss, nn
import time
from d2lzh import Load_Data_Fashion_Mnist as ld

# 卷积神经网络计算比多层感知机要复杂
# 建议使用 GPU 来加速计算
# 我们尝试在 gpu(0) 上创建 NDArray 如果成功则使用 gpu(0) 否则任然使用CPU
def try_gpu():
    try:
        ctx = mx.gpu()
        _ = nd.zeros((1, ), ctx=ctx)
    except mx.base.MXNetError:
        ctx = mx.cpu()

    return ctx


# 数据刚开始是存在 CPU 使用的内存上 当 ctx 变量代表 GPU 及相应的显存时...
# 通过 as_in_context 函数将数据复制到显存上 例如 gpu(0)
def evaluate_accuracy(data_iter, net, ctx):
    acc_sum, n = nd.array([0], ctx=ctx), 0
    for X, y in data_iter:
        # 如果 ctx 代表 GPU 及相应的显存 将数据复制到显存上
        X, y = X.as_in_context(ctx), y.as_in_context(ctx).astype('float32')
        acc_sum += (net(X).argmax(axis=1) == y).sum()
        n += y.size
        
    return acc_sum.asscalar() / n


# 对 train_ch3 函数略作修改 确保计算使用的数据和模型同在内存或显存上
def train_ch5(net, train_iter, test_iter, batch_size, trainer, ctx, num_epochs):
    print('training on', ctx)
    loss = gloss.SoftmaxCrossEntropyLoss()
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        for X, y in train_iter:
            X, y = X.as_in_context(ctx), y.as_in_context(ctx)
            with autograd.record():
                y_hat = net(X)
                l = loss(y_hat, y).sum()
            l.backward()
            trainer.step(batch_size)
            y = y.astype('float32')
            train_l_sum += l.asscalar()
            train_acc_sum += (y_hat.argmax(axis=1) == y).sum().asscalar()
            n += y.size
        test_acc = evaluate_accuracy(test_iter, net, ctx)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f,' 'time %.1f sec'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc,
                 time.time() - start))


# 通过 Sequential 类来实现 LeNet 模型
net = nn.Sequential()

net.add(nn.Conv2D(channels=6, kernel_size=5, activation='sigmoid'),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Conv2D(channels=16, kernel_size=5, activation='sigmoid'),
        nn.MaxPool2D(pool_size=2, strides=2),
        # Dense 会默认将 （批量大小, 通道, 高, 宽） 形状的输入转换成...
        # (批量大小, 通道 * 高 * 宽) 形状的输入
        nn.Dense(120, activation='sigmoid'),
        nn.Dense(84, activation='sigmoid'),
        nn.Dense(10))

# 接下来构造一个高和宽均为 28 的单通道数据样本
# 并逐层进行前向计算来查看每个层的输出形状
x = nd.random.uniform(shape=(1, 1, 28, 28))
net.initialize()
for layer in net:
        x = layer(x)
        print(layer.name, 'output shape\t', x.shape)

# 训练模型
batch_size = 256
train_iter, test_iter = ld.load_data_fashion_mnist(batch_size=batch_size)

# 我没有使用 GPU 所以只会是返回 cpu
ctx = try_gpu()
print(ctx)

# 重新将模型参数初始化到设备变量 ctx 之上
# 使用 Xavier 随机初始化
# 损失函数和训练算法依然使用交叉熵损失函数和小批量随机梯度下降
lr, num_epochs = 0.9, 5
net.initialize(force_reinit=True, ctx=ctx, init=init.Xavier())
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate' : lr})
train_ch5(net, train_iter, test_iter, batch_size, trainer, ctx, num_epochs)