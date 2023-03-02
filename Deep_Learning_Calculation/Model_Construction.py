# 模型构造

from mxnet import nd
from mxnet.gluon import nn

# 继承 Block 类来构造模型
class MLP(nn.Block):
    # 声明带有模型参数的层 这里声明两个全连接层

    def __init__(self, **kwargs):
        # 调用 MLP 父类 Block 的构造函数来进行必要的初始化 这样在构造实列时还可以指定其他函数
        # 参数
        super(MLP, self).__init__(**kwargs)
        # 输出节点为 256 个 激活函数为 relu
        self.hidden = nn.Dense(256, activation='relu')
        # 输出函数
        self.output = nn.Dense(10)

    # 定义模型的前向计算 即如何根据输入 x 计算返回所需要的模型输出
    def forward(self, x):
        return self.output(self.hidden(x))


x = nd.random.uniform(shape=(2, 20))
net = MLP()
# 初始化模型参数
net.initialize()
print(net(x))

# Sequential 类继承自 Block 类
class MySequential(nn.Block):
    def __init__(self, **kwargs):
        super(MySequential, self).__init__(**kwargs)

    def add(self, block):
        # block 是一个 Block 子类实例 假设它有一个独一无二的名字 将它保存在 Block 类的...
        # 成员变量 _children 里 其类型是 OrderedDict 当 MySequential 实例调用...
        # initialize 函数时 系统会自动对 _children 里所有成员初始化
        self._children[block.name] = block

    def forward(self, x):
        # OrderedDict 保证会按照成员添加时的顺序遍历成员
        for block in self._children.values():
            x = block(x)

        return x


net = MySequential()
net.add(nn.Dense(256, activation='relu'))
net.add(nn.Dense(10))
net.initialize()
print(net(x))

# 构造复杂的模型
class FancyMLP(nn.Block):
    def __init__(self, **kwargs):
        super(FancyMLP, self).__init__(**kwargs)
        # 使用 get_constant 创建的随机参数不会在训练中被迭代 (即常数参数)
        self.rand_weight = self.params.get_constant('rand_weight', nd.random.uniform(shape=(20, 20)))
        self.dense = nn.Dense(20, activation='relu')

    def forward(self, x):
        x = self.dense(x)
        # 使用创建的常熟参数 以及 NDArray 的 relu 函数和 dot 函数
        x = nd.relu(nd.dot(x, self.rand_weight.data()) + 1)
        # 复用全连接层 等价于两个全连接层共享参数
        x = self.dense(x)
        # 控制流 这里需要调用 asscalar 函数来返回标量进行比较
        while x.norm().asscalar() > 1:
            x /= 7
        if x.norm().asscalar() < 0.8:
            x *= 10

        return x.sum()


# 在 FancyMLP 模型中 使用了常数权重 rand_weight (注意它不是模型参数) 做了矩阵乘法 (dot) 并重复使用了相同的 Dense 层
net = FancyMLP()
net.initialize()
print(net(x))

# FancyMLP 和 Sequential 类都是 Block 类的子类 可以嵌套调用它们
class NestMLP(nn.Block):
    def __init__(self, **kwargs):
        super(NestMLP, self).__init__(**kwargs)
        self.net = nn.Sequential()
        self.net.add(nn.Dense(64, activation='relu'),
                     nn.Dense(32, activation='relu'))
        self.dense = nn.Dense(16, activation='relu')

    def forward(self, x):
        return self.dense(self.net(x))


net = nn.Sequential()
net.add(NestMLP(), nn.Dense(20), FancyMLP())
net.initialize()
print(net(x))