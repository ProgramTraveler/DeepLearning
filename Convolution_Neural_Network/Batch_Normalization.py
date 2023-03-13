# 批量归一化
import sys
# sys.path.append('/root/Deep_Pro/')
sys.path.append('/home/cowa/wjm/Deep_Pro')
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import nn
from d2lzh import Try_Gpu as tg, Load_Data_Fashion_Mnist as ld, Train_Ch5 as tc5

def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    # 通过 autograd 来判断当前模式是训练模式还是预测模式
    if not autograd.is_training():
        # 如果是在预测模式下 直接使用传入的移动平均所得的均值和方差
        X_hat = (X - moving_mean) / nd.sqrt(moving_var + eps)
    else:
        assert len(X.shape) in (2, 4)
        
        if len(X.shape) == 2:
            # 使用全连接层的情况 计算特征维上的均值和方差
            mean = X.mean(axis=0)
            var = ((X - mean) ** 2).mean(axis=0)
            
        else:
            # 使用二维卷积层的情况 计算通道维上 (axis=1) 的均值和方差
            # 这里我们需要保持 X 的形状以便后面可以做广播运算
            mean = X.mean(axis=(0, 2, 3), keepdims=True)
            var = ((X - mean) ** 2).mean(axis=(0, 2, 3), keepdims=True)
        
        # 训练模式下用当前的均值和方差做标准化
        X_hat = (X - mean) / nd.sqrt(var + eps)
        # 更新移动平均的均值和方差
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var
        
    Y = gamma * X_hat + beta     # 拉伸和偏移
    
    return Y, moving_mean, moving_var

# 自定义一个 BatchNorm 层
# 保护参与求梯度和迭代的拉伸参数 gamma 和偏移参数 beta
# 同时也维护移动平均得到的均值和方差
class BatchNorm(nn.Block):
    def __init__(self, num_features, num_dims, **kwargs):
        super(BatchNorm, self).__init__(**kwargs)
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        
        # 参与梯度和迭代的拉伸和偏移参数 分别初始化为 0 和 1
        self.gamma = self.params.get('gamma', shape=shape, init=init.One())
        self.beta = self.params.get('beta', shape=shape, init=init.Zero())
        # 不参与求梯度和迭代的变量 全在内存上初始化为 0
        self.moving_mean = nd.zeros(shape)
        self.moving_var = nd.zeros(shape)
        
    def forward(self, X):
        # 如果 X 不在内存上 将 moving_mean 和 moving_var 复制到 X 所在显存上
        if self.moving_mean.context != X.context:
            self.moving_mean = self.moving_mean.copyto(X.context)
            self.moving_var = self.moving_var.copyto(X.context)
        # 保存更新过的 moving_mean 和 moving_var
        Y, self.moving_mean, self.moving_var = batch_norm(X, self.gamma.data(), self.beta.data(),
                                                          self.moving_mean, self.moving_var, eps=1e-5,
                                                          momentum=0.9)
            
        return Y
        
        
# 使用批量归一化层的 LeNet
# 在所有的卷积层或全连接层之后 激活层之前加入批量归一化层
net = nn.Sequential()
net.add(nn.Conv2D(6, kernel_size=5),
        BatchNorm(6, num_dims=4),
        nn.Activation('sigmoid'),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Conv2D(16, kernel_size=5),
        BatchNorm(16, num_dims=4),
        nn.Activation('sigmoid'),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Dense(120),
        BatchNorm(120, num_dims=2),
        nn.Activation('sigmoid'),
        nn.Dense(84),
        BatchNorm(84, num_dims=2),
        nn.Activation('sigmoid'),
        nn.Dense(10))

# 训练修改后的模型
lr, num_epochs, batch_size, ctx = 1.0, 5, 256, tg.try_gpu()
net.initialize(ctx=ctx, init=init.Xavier())
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate' : lr})
train_iter, test_iter = ld.load_data_fashion_mnist(batch_size)
tc5.train_ch5(net, train_iter, test_iter, batch_size, trainer, ctx, num_epochs)

# 简洁实现
net = nn.Sequential()
net.add(nn.Conv2D(6, kernel_size=5),
        nn.BatchNorm(),
        nn.Activation('sigmoid'),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Conv2D(16, kernel_size=5),
        nn.BatchNorm(),
        nn.Activation('sigmoid'),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Dense(120),
        nn.BatchNorm(),
        nn.Activation('sigmoid'),
        nn.Dense(84),
        nn.BatchNorm(),
        nn.Activation('sigmoid'),
        nn.Dense(10))

net.initialize(ctx=ctx, init=init.Xavier())
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate' : lr})
tc5.train_ch5(net, train_iter, test_iter, batch_size, trainer, ctx, num_epochs)