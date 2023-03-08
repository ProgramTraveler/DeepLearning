# 深度卷积神经网络 AlexNet
# AlexNet 引入了大量的图像增广 如翻转 裁剪和颜色变化

import sys
sys.path.append('/root/Deep_Pro/')
from mxnet import gluon, init, nd, autograd
from mxnet.gluon import data as gdata, nn, loss as gloss
import os
from d2lzh import Try_Gpu as tg
import time

def load_fashion_mnist(batch_size, resize=None, root= os.path.join('-', '.mxnet', 'datasets', 'fashion-mnist')):
    # 展开用户路径 '~'
    root = os.path.expanduser(root)
    transformer = []
    if resize:
        transformer += [gdata.vision.transforms.Resize(resize)]
    transformer += [gdata.vision.transforms.ToTensor()]
    transformer = gdata.vision.transforms.Compose(transformer)
    
    mnist_train = gdata.vision.FashionMNIST(root=root, train=True)
    mnist_test = gdata.vision.FashionMNIST(root=root, train=False)
    
    num_workers = 0 if sys.platform.startswith('win32') else 4
    train_iter = gdata.DataLoader(mnist_train.transform_first(transformer), batch_size, shuffle=True,
                                  num_workers=num_workers)
    test_iter = gdata.DataLoader(mnist_test.transform_first(transformer), batch_size, shuffle=False,
                                 num_workers=num_workers)
    
    return train_iter, test_iter

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
        print("###")
        for X, y in train_iter:
            # print("-----")
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
    
    
net = nn.Sequential()
# 使用较大的 11 * 11 窗口来捕获物体
# 同时使用步幅 4 来较大幅度减小输出高和宽
# 这里使用的输出通道数比 LeNet 也要大很多
net.add(nn.Conv2D(96, kernel_size=11, strides=4, activation='relu'),
        nn.MaxPool2D(pool_size=3, strides=2),
        # 减小卷积窗口 使用填充为 2 来使得输入与输出的高和宽一致 且增大输出通道数
        nn.Conv2D(256, kernel_size=5, padding=2, activation='relu'),
        nn.MaxPool2D(pool_size=3, strides=2),
        # 连续 3 个卷积层 且使用更小的卷积窗口 除了最后的卷积层外 进一步增大了输出通道数
        # 前两个卷积层后不使用池化层来减小输入的高和宽
        nn.Conv2D(384, kernel_size=3, padding=1, activation='relu'),
        nn.Conv2D(384, kernel_size=3, padding=1, activation='relu'),
        nn.Conv2D(256, kernel_size=3, padding=1, activation='relu'),
        nn.MaxPool2D(pool_size=3, strides=2),
        # 这里全连接层的输出个数比 LeNet 中的大数倍 使用丢弃层来缓解过拟合
        nn.Dense(4096, activation="relu"), nn.Dropout(0.5),
        nn.Dense(4096, activation="relu"), nn.Dropout(0.5),
        # 输出层 这里使用 Fashion-MNIST 所以用类别数为 10
        nn.Dense(10))

# 构造一个高和宽均为 224 的单通道数据样本来观察每一层的输出形状
x = nd.random.uniform(shape=(1, 1, 224, 224))
net.initialize()
for layer in net:
    x = layer(x)
    # print(layer.name, 'output shape:\t', x.shape)
    
# 读取数据集
# batch_size = 128
batch_size = 128
# 如果出现 out of memory 的报错信息 可减小 batch_size 或者 resize
# resize=224
train_iter, test_iter = load_fashion_mnist(batch_size, resize=224)

# 训练模型 会跑很久
lr, num_epoch, ctx = 0.01, 5, tg.try_gpu()
net.initialize(force_reinit=True, ctx=ctx, init=init.Xavier())
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate' : lr})
train_ch5(net, train_iter, test_iter, batch_size, trainer, ctx, num_epoch)