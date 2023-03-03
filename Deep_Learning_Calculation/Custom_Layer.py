# 自定义层
from mxnet import gluon, nd
from mxnet.gluon import nn

class CenteredLayer(nn.Block):
    def __init__(self, **kwargs):
        super(CenteredLayer, self).__init__(**kwargs)

    def forward(self, x):
        return x - x.mean()


# 把 layer 看成 net 好理解一些
layer = CenteredLayer()
print(layer(nd.array([1, 2, 3, 4, 5])))

# 我们也可以构造更复杂的模型
net = nn.Sequential()
net.add(nn.Dense(128), CenteredLayer())
net.initialize()
y = net(nd.random.uniform(shape=(4, 8)))
print(y.mean().asscalar())

# 含参数模型的自定义层
params = gluon.ParameterDict()
params.get('param2', shape=(2, 3))
print(params)

# 实现一个含权重参数和偏差参数的全连接层
# 使用 ReLU 函数作为激活函数
# in_units 和 units 分别代表输入个数和输出个数
class MyDense(nn.Block):
    # units 为该层的输出个数 in_units 为该层的输入个数
    def __init__(self, units, in_units, **kwargs):
        super(MyDense, self).__init__(**kwargs)
        self.weight = self.params.get('weight', shape=(in_units, units))
        self.bias = self.params.get('bias', shape=(units, ))

    def forward(self, x):
        linear = nd.dot(x, self.weight.data()) + self.bias.data()
        return nd.relu(linear)


dense = MyDense(units=3, in_units=5)
print(dense.params)

# 使用自定义层做前向计算
dense.initialize()
print(dense(nd.random.uniform(shape=(2, 5))))

# 也可以使用自定义层构造模型 它和 Gluon 的其他层在使用上很类似
net = nn.Sequential()
net.add(MyDense(8, in_units=64),
        MyDense(1, in_units=8))
net.initialize()
print(net(nd.random.uniform(shape=(2, 64))))