# 模型参数的延后初始化
from mxnet import init, nd
from mxnet.gluon import nn

class MyInit(init.Initializer):
    def _init_weight(self, name, data):
        print('Init', name, data.shape)
        # 实际的初始化在这里省略


net = nn.Sequential()
net.add(nn.Dense(256, activation='relu'),
        nn.Dense(10))

# 并没有真正的初始化参数
net.initialize(init=MyInit())

# 只有当输入层的形状给出后 才能推出隐藏层的形状 才能初始化
x = nd.random.uniform(shape=(2, 20))
y = net(x)
# 这个初始化只会在第一次前向计算时才会被调用
# 再次运算前向计算时 不会在产生 Init 实例的输出
y = net(x)

# 避免延后初始化
# 对已初始化的模型重新初始化
net.initialize(init=MyInit(), force_reinit=True)

# 在创建层的时候就指定了它的输入个数
# 通过 in_units 来指定每个全连接层的输入个数
net = nn.Sequential()
net.add(nn.Dense(256, in_units=20, activation='relu'))
net.add(nn.Dense(10, in_units=256))

net.initialize(init=MyInit())