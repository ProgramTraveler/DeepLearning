from mxnet.gluon import nn
from d2lzh import ResIdual as ri

def resnet18(num_classes):
    
    def resnet_block(num_channels, num_residuals, first_block=False):
        blk = nn.Sequential()
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.add(ri.Residual(num_channels, use_1X1conv=True, strides=2))
            else:
                blk.add(ri.Residual(num_channels))
                
            return blk
        
    net = nn.Sequential()
    # 这里使用了较小的卷积核 步幅和填充
    # 并去掉了最大池化层
    net.add(nn.Conv2D(64, kernel_size=3, strides=1, padding=1),
            nn.BatchNorm(), nn.Activation('relu'))
    net.add(resnet_block(64, 2, first_block=True),
            resnet_block(128, 2),
            resnet_block(256, 2),
            resnet_block(512, 2))
    
    net.add(nn.GlobalAvgPool2D(), nn.Dense(num_classes))
    
    return net