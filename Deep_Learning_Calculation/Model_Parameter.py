# 模型参数的访问 初始化和共享

from mxnet import init, nd
from mxnet.gluon import nn

net = nn.Sequential()
net.add(nn.Dense(256, activation='relu'))
net.add(nn.Dense(10))
net.initialize()    # 使用默认的初始化方法

x = nd.random.uniform(shape=(2, 20))
y = net(x)  # 前向计算

# 访问模型参数 一个由参数名称映射到参数实例的字典 权重参数名称为 dense_weight
print(net[0].params, type(net[0].params))

# 访问特定参数 我们可以通过名字来访问字典里的元素 也可以直接使用它的变量名
print(net[0].params['dense0_weight'], net[0].weight)

# Gluon 里参数类型为 Parameter 类 包含参数和梯度的数值 也可以分别通过 data 函数 和 grad 函数来访问
# 因为随机初始化了权重 所有的权重参数是一个由随机数组成的形状为 (256, 20) 的 NDArray
print(net[0].weight.data())

# 权重梯度的形状和权重的形状一样 因为我们还没有进行反向传播计算 所以梯度的值全为 0
print(net[0].weight.grad())

# 也可以访问其他层的参数
# 输出层的偏差值
print(net[1].bias.data())

# 可以使用 collect_params 函数来获取 net 变量所有的嵌套(例如通过 add 函数嵌套)层的所有参数
# 返回同样是一个由参数名称到参数实例的字典
print(net.collect_params())

# 这个函数可以通过正则表达式来匹配参数名 从而筛选需要的参数
print(net.collect_params('.*weight'))

# 初始化模型参数
# 非首次对模型初始化需要指定 force_reinit 为真
# 将权重参数初始化成均量为 0 标准差为 0.01 的正态分布随机数 并依然将偏差参数清零
net.initialize(init=init.Normal(sigma=0.01), force_reinit=True)
print(net[0].weight.data()[0])

# 使用常数来初始化权重参数
net.initialize(init=init.Constant(1), force_reinit=True)
print(net[0].weight.data()[0])

# 对某个特定参数进行初始化
# 对隐藏层的权重使用 Xavier 随机初始化方法
net[0].weight.initialize(init=init.Xavier(), force_reinit=True)
print(net[0].weight.data()[0])

# 自定义初始化方法
class MyInit(init.Initializer):
    # 令权重有一般概率初始化为 0 有另一半概率初始化为 [-10, -5] [5, 10] 两个区间里均匀分布的随机数
    def _init_weight(self, name, data):
        print('Init', name, data.shape)
        data[:] = nd.random.uniform(low=-10, high=10, shape=data.shape)
        data *= data.abs() >= 5


net.initialize(MyInit(), force_reinit=True)
print(net[0].weight.data()[0])

# Init dense0_weight(256, 20)
# Init dense1_weight(10, 256)

# 还可以通过 Parameter 类的 set_data 函数来直接改写模型参数
# 将隐藏层参数在现有的基础上加 1
net[0].weight.set_data(net[0].weight.data() + 1)
print(net[0].weight.data()[0])

# 共享模型参数
# 在多个层直接共享模型参数
# 如果不同层使用同一份参数 那么它们在前向计算和反向传播时都会共享相同的参数
net = nn.Sequential()
shared = nn.Dense(8, activation='relu')
net.add(nn.Dense(8, activation='relu'),
        shared,
        nn.Dense(8, activation='relu', params=shared.params),
        nn.Dense(10))
net.initialize()

# 在构造第三隐藏层时 我们通过 params 来指定它使用第二隐藏层参数
# 因为模型包含了梯度 所以在反向传播计算时 第二隐藏层和第三隐藏层的梯度都会被累加在 shared.params.grad() 里
x = nd.random.uniform(shape=(2, 20))
net(x)
print(net[1].weight.data()[0] == net[2].weight.data()[0])