# 残差块 可以设定输出通道数 是否使用额外的 1*1 卷积层来修改通道数以及卷积层的步幅
from mxnet.gluon import nn
from mxnet import nd

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