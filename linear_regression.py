# 线性回归从零开始
# matplotlib 用于作图

from IPython import display
from matplotlib import pyplot as plt
from mxnet import autograd, nd
import random
from ipykernel.pylab.config import InlineBackend
from d2lzh import ReadData as Rd, MatrixMultiplication as Mm, Loss as lo, Optimization as Op


# 函数要空两行
def use_svg_display():
    # 用矢量图显示
    display.set_matplotlib_formats('svg')


def set_figSize(figSize=(3.5, 2.5)):
    use_svg_display()
    # 设置图的尺寸
    # figure.figsize 是rcParams 中的固定搭配 亏我还找怀疑半天人生 pycharm 的格式控制是真的烦 一会把你关了
    plt.rcParams['figure.figsize'] = figSize


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

set_figSize()

print("图像显示")
# 加分号只显示图 分号加不加好像没什么区别
plt.scatter(features[:, 1].asnumpy(), labels.asnumpy(), 1)
plt.show()

# 读取第一个小批量数据样本并打印 每个批量的特征形状为 (10, 2) 分别对应批量的大小和输入个数 标签的形状为批量大小
batch_size = 10
for x, y in Rd.data_iter(batch_size, features, labels):
    print(x, y)
    break

# 初始化模型参数
w = nd.random.normal(scale=0.01, shape=(num_inputs, 1))
b = nd.zeros(shape=(1, ))

# 在之后的模型训练中 需要对这些参数求梯度来迭代参数的值 创建它们的梯度
w.attach_grad()
b.attach_grad()

# 训练模型
# 不太懂学习率是怎样决定出来的
lr = 0.03
num_epochs = 3
# 定义模型
net = Mm.linreg
# 损失函数
loss = lo.squared_loss

# 训练模型一共需要 num_epochs 个迭代周期
for epoch in range(num_epochs):
    # 在每一个迭代周期中 会使用训练集中所有样本一次(假设样本数能够被批量大小整除)
    # x 和 y 分别是小批量样本的特征和标签
    for X, y in Rd.data_iter(batch_size, features, labels):
        with autograd.record():
            # l 是有关小批量的损失
            l = loss(net(X, w, b), y)
        # 小批量的损失对模型参数求梯度1
        l.backward()
        # 使用小批量随机梯度下降迭代模型参数
        Op.sgd([w, b], lr, batch_size)
    # 这是？
    train_l = loss(net(features, w, b), labels)
    print('epoch %d, loss %f' % (epoch + 1,train_l.mean().asnumpy()))

print(true_w, w)
print(true_b, b)