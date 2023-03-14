# 残差网络
import sys
# sys.path.append('/root/Deep_Pro/')
# sys.path.append('/home/cowa/wjm/Deep_Pro')
sys.path.append('/root/Deep/')
from mxnet import gluon, init, nd
from mxnet.gluon import nn
from d2lzh import Try_Gpu as tg, Load_Data_Fashion_Mnist as ld, Train_Ch5 as tc5

class Residual(nn.Block):
    def __init__(self, num_channels, use_1X1conv=False, strides=1, **kwargs):
        super(Residual, self).__init__(**kwargs)
        
        self.conv1 = nn.Conv2D(num_channels, kernel_size=3, padding=1, strides=strides)
        self.conv2 = nn.Conv2D(num_channels, kernel_size=3, padding=1)
        
        if use_1X1conv:
            self.conv3 = nn.Conv2D(num_channels, kernel_size=1, strides=strides)
            
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm()
        self.bn2 = nn.BatchNorm()
        
    def forward(self, X):
        Y = nd.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        
        if self.conv3:
            X = self.conv3(X)
        
        return nd.relu(Y + X)
    

def resnet_block(num_channels, num_residuals, first_block=False):
    blk = nn.Sequential()
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.add(Residual(num_channels, use_1X1conv=True, strides=2))
        else:
            blk.add(Residual(num_channels))
            
    return blk
    

# 查看输入和输出形状一致的情况
blk = Residual(3)
blk.initialize()
X = nd.random.uniform(shape=(4, 3, 6, 6))
blk(X).shape

# 我们可以在增加输出通道数的同时减半输出的高和宽
blk = Residual(6, use_1X1conv=True, strides=2)
blk.initialize()
blk(X).shape

net = nn.Sequential()
net.add(nn.Conv2D(64, kernel_size=7, strides=2, padding=3),
        nn.BatchNorm(), nn.Activation('relu'),
        nn.MaxPool2D(pool_size=3, strides=2, padding=1))

# 为 ResNet 加入所有残差块 每个模块使用两个残差块
net.add(resnet_block(64, 2, first_block=True),
        resnet_block(128, 2),
        resnet_block(256, 2),
        resnet_block(512, 2))

# 加入全局平均池化层后接上全连接层输出
net.add(nn.GlobalAvgPool2D(), nn.Dense(10))
# 观察输入形状在 ResNet 不同模块之间的变化
X = nd.random.uniform(shape=(1, 1, 224, 224))
net.initialize()
for layer in net:
    X = layer(X)
    print(layer.name, 'output shape:\t', X.shape)
    
# 训练模型
lr, num_epochs, batch_size, ctx = 0.05, 5, 256, tg.try_gpu()
net.initialize(force_reinit=True, ctx=ctx, init=init.Xavier())
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate' : lr})
train_iter, test_iter = ld.load_data_fashion_mnist(batch_size, resize=96)
tc5.train_ch5(net, train_iter, test_iter, batch_size, trainer, ctx, num_epochs)