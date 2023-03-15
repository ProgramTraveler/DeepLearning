# 图像增广训练模型
# 使用 CIFAR-10 数据集
import mxnet as mx
from mxnet import autograd, gluon, image, init, nd
from mxnet.gluon import data as gdata, loss as gloss, utils as gutils
import sys
sys.path.append('/root/Deep_Pro/')
import time
from d2lzh import Set_Figsize as sf, Show_Images as si, Resnet18 as rs18
import matplotlib.pyplot as plt


# 定义一个辅助函数来方便读取图像并应用图像增广
def load_cifar10(is_train, augs, batch_size):
    # 允许使用多进程来加速数据读取
    return gdata.DataLoader(gdata.vision.CIFAR10(train=is_train).transform_first(augs),
                            batch_size=batch_size, shuffle=is_train, num_workers=num_workers)


# 定义 try_all_gpus 函数 从而可以获取所有可用的 GPU
def try_all_gpus():
    ctxes = []
    try:
        for i in range(16):  # 假设一台机器上 GPU 的数量不超过 16
            ctx = mx.gpu(i)
            _ = nd.array([0], ctx=ctx)
            ctxes.append(ctx)
    except mx.base.MXNetError:
        pass
    if not ctxes:
        ctxes = [mx.cpu()]
    
    return ctxes


# 将小批量数据样本 batch 划分并复制到 ctx 变量指定的各个显存
def _get_batch(batch, ctx):
    features, labels = batch
    if labels.dtype != features.dtype:
        labels = labels.astype(features.dtype)
    
    return (gutils.split_and_load(features, ctx),
            gutils.split_and_load(labels, ctx), features.shape[0])


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
            train_acc_sum += sum([(y_hat.argmax(axis=1) == y).sum().asscalar()
                                  for y_hat, y in zip(y_hats, ys)])
            m += sum([y.size for y in ys])
        test_acc = evaluate_accuracy(test_iter, net, ctx)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f,'
              'time % .1f sec'
              % (epoch + 1, train_l_sum / n, train_acc_sum / m, test_acc, time.time() - start))
            

# 定义 train_with_data_aug 函数来使用图像增广来训练模型
# 该函数获取了可用的 GPU 并将 Adam 算法作为训练使用的优化算法
# 然后将图像增广应用于训练数据集之上
def train_with_data_aug(train_augs, test_augs, lr=0.001):
    batch_size, ctx, net = 256, try_all_gpus(), rs18.resnet18(10)
    net.initialize(ctx=ctx, init=init.Xavier())
    trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate' : lr})
    loss = gloss.SoftmaxCrossEntropyLoss()
    train_iter = load_cifar10(True, train_augs, batch_size)
    test_iter = load_cifar10(False, test_augs, batch_size)
    train(train_iter, test_iter, net, loss, trainer, ctx, num_epochs=10)
    
    
si.show_images(gdata.vision.CIFAR10(train=True)[0:32][0], 4, 8, scale=0.8)
# 显示图像
plt.show()

# 通常只将图像增广应用在训练样本上
# 这里只使用最简单的随机左右翻转
# ToTensor 实例将小批量图像转换成 MXNet 需要的格式 即 (批量大小, 通道数, 高, 宽) 域值在 0 到 1 之间且类型为 32 位浮点数
flip_aug = gdata.vision.transforms.Compose([gdata.vision.transforms.RandomFlipLeftRight(),
                                            gdata.vision.transforms.ToTensor()])
no_aug = gdata.vision.transforms.Compose([gdata.vision.transforms.ToTensor()])

num_workers = 0 if sys.platform.startswith('win32') else 4   # 根据 GPU 性能而设定

# 使用多 GPU 训练模型
# 使用随机左右翻转的图像增广来训练模型
train_with_data_aug(flip_aug, no_aug)