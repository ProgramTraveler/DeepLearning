# 线性回归的简洁实现

from mxnet import nd, autograd
# Gluon 提供了 data 包来读取数据
from mxnet.gluon import data as gdata, nn, loss as gloss
from mxnet import init, gluon

# 输入的特征数为 2
num_inputs = 2
# 训练数据集样本1数为 1000
num_examples = 1000
# 使用线性回归模型真实权重
true_w = [2, -3.4]
# 偏差 b = 4.2
true_b = 4.2
# 每一行是一个长度为 2 的向量
features = nd.random.normal(scale=1, shape=(num_examples, num_inputs))
# 每一行是一个长度为 1 的向量(标量)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += nd.random.normal(scale=0.01, shape=labels.shape)

# 读取数据集
# 随机读取包10个数据样条的小批量
batch_size = 10
# 将训练数据的特征和标签组合
dataset = gdata.ArrayDataset(features, labels)
# 随机读取小批量
data_iter = gdata.DataLoader(dataset, batch_size, shuffle=True)

# 定义模型
# Sequential 实列可以看作是一个串联各个层的容器
net = nn.Sequential()
# 线性回归是一个单层神经网络而且输出层和输入层是完全连接 我们定义该层输出个数为 1 全连接是一个 Dense 实例
net.add(nn.Dense(1))

# 初始化模型参数
# 指定权重参数每个元素将在初始化时随机采样于均值为 0 标准差为 0.01 的正态分布
net.initialize(init.Normal(sigma=0.01))

# 定义损失函数
# 平方损失又称 L2 范数损失
loss = gloss.L2Loss()

# 定义优化算法
# 指定学习率为 0.03 的小批量随机梯度下降 (sgd) 为优化算法 该优化算法将用来迭代 net 实列所有通过 add 函数嵌套的层所包含的全部参数
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate' : 0.03})

# 训练模型
num_epochs = 3
for epoch in range(1, num_epochs + 1):
    for x, y in data_iter:
        with autograd.record():
            l = loss(net(x), y)
        l.backward()
        trainer.step(batch_size)
    l = loss(net(features), labels)
    print('epoch %d, loss: %f' % (epoch, l.mean().asnumpy()))

dense = net[0]

print(true_w, dense.weight.data())
print(true_b, dense.bias.data())