# 使用丢弃法来应对过拟合问题

from mxnet import autograd, gluon, init, nd
from mxnet.gluon import loss as gloss, nn
from d2lzh import Load_Data_Fashion_Mnist as ld, Train_Ch3 as tc3


# 该函数将以 drop_prod 的概率丢弃 NDArray 输入 x 中的元素
def dropout(x, drop_prod):
    assert 0 <= drop_prod <= 1
    keep_prod = 1 - drop_prod

    # 这种情况下把元素全部丢弃
    if keep_prod == 0:
        return x.zeros_like()
    mask = nd.random.uniform(0, 1, x.shape) < keep_prod
    return mask * x / keep_prod


def net(X):
    X = X.reshape((-1, num_inputs))
    H1 = (nd.dot(X, w1) + b1).relu()

    # 只在训练模型时使用丢弃法
    if autograd.is_training():
        # 只在第一层全连接后添加丢弃层
        H1 = dropout(H1, drop_prob1)

    H2 = (nd.dot(H1, w2) + b2).relu()
    if autograd.is_training():
        # 只在第二层全连接后添加丢弃层
        H2 = dropout(H2, drop_prob2)

    return nd.dot(H2, w3) + b3


# 对 dropout 进行测试 丢弃概率分别为 0, 0.5, 1
x = nd.arange(16).reshape((2, 8))
dropout(x, 0)

dropout(x, 0.5)

dropout(x, 1)

# 定义模型参数
# 依然使用 Fashion-MNIST 数据集 定义一个包含两个隐藏层的多层感知机 两个隐藏层的个数都是 256
num_inputs, num_outputs, num_hidden1, num_hidden2 = 784, 10, 256, 256

w1 = nd.random.normal(scale=0.01, shape=(num_inputs, num_hidden1))
b1 = nd.zeros(num_hidden1)
w2 = nd.random.normal(scale=0.01, shape=(num_hidden1, num_hidden2))
b2 = nd.zeros(num_hidden2)
w3 = nd.random.normal(scale=0.01, shape=(num_hidden2, num_outputs))
b3 = nd.zeros(num_outputs)

params = [w1, b1, w2, b2, w3, b3]

for param in params:
    param.attach_grad()

# 定义模型
# 将全连接层和激活函数 ReLU 串起来 并对每个激活函数的输出使用丢弃法
# 第一个隐藏层的丢弃概率设为 0.2 第二个隐藏层的丢弃概率设为 0.5
drop_prob1, drop_prob2 = 0.2, 0.5

# 训练和测试模型
num_epochs, lr, batch_size = 5, 0.5, 256
loss = gloss.SoftmaxCrossEntropyLoss()
train_iter, test_iter = ld.load_data_fashion_mnist(batch_size)
tc3.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, params, lr)

# 简洁实现
# 在 gluon 中 我们只需要在全连接层后添加 Dropout 层并指定丢弃概率
# 在训练模型时 Dropout 层将以指定的丢弃率随机丢弃上一层的输出元素
# 在测试模型中 Dropout 层并不发挥作用

net = nn.Sequential()
net.add(nn.Dense(256, activation="relu"),
        nn.Dropout(drop_prob1),     # 在第一个全连接层后添加丢弃层
        nn.Dense(256, activation="relu"),
        nn.Dropout(drop_prob2),     # 在第二个全连接层后添加丢弃层
        nn.Dense(10))

net.initialize(init.Normal(sigma=0.01))

# 注意拼写
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate' : lr})
tc3.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, trainer)