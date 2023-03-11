# NiN 网络中的网络
import sys
sys.path.append('/home/cowa/wjm/Deep_Pro')

from mxnet import gluon, init, nd
from mxnet.gluon import nn
from d2lzh import Try_Gpu as tg, Load_Data_Fashion_Mnist as ld, Train_Ch5 as tc5

def nin_block(num_channels, kernel_size, strides, padding):
    blk = nn.Sequential()
    blk.add(nn.Conv2D(num_channels, kernel_size,
                      strides, padding, activation='relu'),
            nn.Conv2D(num_channels, kernel_size=1, activation='relu'),
            nn.Conv2D(num_channels, kernel_size=1, activation='relu'))
    
    return blk


net = nn.Sequential()
net.add(nin_block(96, kernel_size=11, strides=4, padding=0),
        nn.MaxPool2D(pool_size=3, strides=2),
        nin_block(256, kernel_size=5, strides=1, padding=2),
        nn.MaxPool2D(pool_size=3, strides=2),
        nin_block(384, kernel_size=3, strides=1, padding=1),
        nn.MaxPool2D(pool_size=3, strides=1), nn.Dropout(0.5),
        # 标签类别是 0
        nin_block(10, kernel_size=3, strides=1, padding=1),
        # 全局平均池化池层将形状自动设置成输入的高和宽
        nn.GlobalAvgPool2D(),
        # 将四维的输出转成二维的输出 其形状为 (批量大小, 10)
        nn.Flatten())

# 构建一个数据样本来查看每一层的输出形状
x = nd.random.uniform(shape=(1, 1, 224, 224))
net.initialize()
for layer in net:
    x = layer(x)
    print(layer.name, 'output shape:\t', x.shape)
 
# 训练模型
lr, num_epochs, batch_size, ctx = 0.1, 5, 128, tg.try_gpu()
net.initialize(force_reinit=True, ctx=ctx, init=init.Xavier())
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate' : lr})
train_iter, test_iter = ld.load_data_fashion_mnist(batch_size, resize=224)
tc5.train_ch5(net, train_iter, test_iter, batch_size, trainer, ctx, num_epochs)