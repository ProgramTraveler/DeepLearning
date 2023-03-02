# 模型构造

from mxnet import nd
from mxnet.gluon import nn

# 继承 Block 类来构造模型
class MLP (nn.Block):
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