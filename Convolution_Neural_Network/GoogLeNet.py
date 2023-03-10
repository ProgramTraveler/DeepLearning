# 含并行连接的网络 GoogLeNet
from mxnet import gluon, init, nd
from mxnet.gluon import nn
from d2lzh import Try_Gpu as tg, Load_Data_Fashion_Mnist as ld, Train_Ch5 as tc5

class Inception(nn.Block):
    # c1 - c4 为每条线路里的层的输出通道数
    def __init__(self, c1, c2, c3, c4, **kwargs):
        super(Inception, self).__init__(**kwargs)
        # 线路 1  -> 单 1 * 1 卷积层
        self.p1_1 = nn.Conv2D(c1, kernel_size=1, activation='relu')
        # 线路 2 -> 1 * 1 卷积层后接 3 * 3 卷积层
        self.p2_1 = nn.Conv2D(c2[0], kernel_size=1, activation='relu')
        self.p2_2 = nn.Conv2D(c2[1], kernel_size=3, padding=1, activation='relu')
        # 线路 3 -> 1 * 1 卷积层后接 5 * 5 卷积层
        self.p3_1 = nn.Conv2D(c3[0], kernel_size=1, activation='relu')
        self.p3_2 = nn.Conv2D(c3[1], kernel_size=5, padding=2, activation='relu')
        # 线路 4 -> 3 * 3 最大池化层后接 1 * 1 卷积层
        self.p4_1 = nn.MaxPool2D(pool_size=3, strides=1, padding=1)
        self.p4_2 = nn.Conv2D(c4, kernel_size=1, activation='relu')
        
    def forward(self, x):
        p1 = self.p1_1(x)
        p2 = self.p2_2(self.p2_1(x))
        p3 = self.p3_2(self.p3_1(x))
        p4 = self.p4_2(self.p4_1(x))
        return nd.concat(p1, p2, p3, p4, dim=1)  # 在通道维上连结输出
    

# 第一个模块使用一个 64 通道的 7 * 7卷积层
b1 = nn.Sequential()
b1.add(nn.Conv2D(64, kernel_size=7, strides=2, padding=3, activation='relu'),
       nn.MaxPool2D(pool_size=3, strides=2, padding=1))

# 第二个模块使用 2 个卷积层
# 首先是 64 通道的 1 * 1 卷积层’
# 然后是将通道增大 3 倍的 3 * 3卷积层
# 它对应 Inception 块中的第二条线路
b2 = nn.Sequential()
b2.add(nn.Conv2D(64, kernel_size=1, activation='relu'),
       nn.MaxPool2D(pool_size=3, strides=2, padding=1))

# 第 3 个模块串联 2 个完整的 Inception 块
b3 = nn.Sequential()
b3.add(Inception(64, (96, 128), (16, 32), 32),
       Inception(128, (128, 129), (32, 96), 64),
       nn.MaxPool2D(pool_size=3, strides=2, padding=1))

# 第四个模块串联了 5 个 Inception 块
b4 = nn.Sequential()
b4.add(Inception(192, (96, 208), (16, 48), 64),
       Inception(160, (112, 224), (24, 64), 64),
       Inception(128, (128, 256), (24, 64), 64),
       Inception(112, (144, 288), (32, 64), 64),
       Inception(256, (160, 320), (32, 128), 128),
       nn.MaxPool2D(pool_size=3, strides=2, padding=1))

# 第 5 个模块
b5 = nn.Sequential()
b5.add(Inception(256, (160, 320), (32, 128), 128),
       Inception(384, (192, 384), (48, 128), 128),
       nn.GlobalAvgPool2D())

net = nn.Sequential()
net.add(b1, b2, b3, b4, b5, nn.Dense(10))

# GoogLeNet 模型计算复杂 而且不如 VGG 那样便于修改通道数
# 将输入的高和宽从 224 降到 96 来简化计算
X = nd.random.uniform(shape=(1, 1, 96, 96))
net.initialize()
# 演示各个模块之间的输出的形状变化
for layer in net:
    x = layer(X)
    print(layer.name, 'output shape\t', x.shape)

# 训练模型
# 使用高和宽均为 96 像素的图像来训练模型
lr, num_epochs, batch_size, ctx = 0.1, 5, 128, tg.try_gpu()
net.initialize(force_reinit=True, ctx=ctx, init=init.Xavier())
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate' : lr})
train_iter, test_iter = ld.load_data_fashion_mnist(batch_size, resize=96)
tc5.train_ch5(net, train_iter, test_iter, batch_size, trainer, ctx, num_epochs)