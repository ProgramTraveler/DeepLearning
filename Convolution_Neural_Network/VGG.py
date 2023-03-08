# 使用重复元素的网络 VGG
import sys
sys.path.append('/root/Deep_Pro/')
from mxnet import gluon, init, nd
from mxnet.gluon import nn
from d2lzh import Try_Gpu as tg, Train_Ch5 as tc5, Load_Data_Fashion_Mnist as ld

def vgg_block(num_convs, num_channels):
    blk = nn.Sequential()
    for _ in range(num_convs):
        blk.add(nn.Conv2D(num_channels, kernel_size=3, padding=1, activation='relu'))
    blk.add(nn.MaxPool2D(pool_size=2, strides=2))
    
    return blk


# 实现 VGG-11
def vgg(conv_arch):
    net = nn.Sequential()
    # 卷积层部分
    for (num_convs, num_channels) in conv_arch:
        net.add(vgg_block(num_convs, num_channels))
    # 全连接层部分
    net.add(nn.Dense(4096, activation='relu'), nn.Dropout(0.5),
           nn.Dense(4096, activation='relu'), nn.Dropout(0.5),
           nn.Dense(10))
     
    return net


conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))

# 构造一个 高和宽均为 224 的单通道数据样本来观察每一层的输出形状
net = vgg(conv_arch)
net.initialize()
x = nd.random.uniform(shape=(1, 1, 224, 224))
for blk in net:
    x = blk(x)
    # print(blk.name, 'output shape:\t', x.shape)

# 训练模型
ratio = 4
small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]
net = vgg(small_conv_arch)

lr, num_epochs, batch_size, ctx = 0.05, 5, 128, tg.try_gpu()
net.initialize(ctx=ctx, init=init.Xavier())
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate' : lr})
train_iter, test_iter = ld.load_data_fashion_mnist(batch_size, resize=224)
tc5.train_ch5(net, train_iter, test_iter, batch_size, trainer, ctx, num_epochs)