# 线性回归从零开始
# matplotlib 用于作图

from IPython import display
from matplotlib import pyplot as plt
from mxnet import autograd, nd
import random
from ipykernel.pylab.config import InlineBackend


# 函数要空两行
def use_svg_display():
    # 用矢量图显示
    display.set_matplotlib_formats('svg')


def set_figSize(figSize=(3.5, 2.5)):
    use_svg_display()
    # 设置图的尺寸
    # figure.figsize 是rcParams 中的固定搭配 亏我还找怀疑半天人生 pycharm 的格式控制是真的烦 一会把你关了
    plt.rcParams['figure.figsize'] = figSize


def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # 样品的读取顺序是随机的
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        j = nd.array(indices[i: min(i + batch_size, num_examples)])


# 输入的特征数为 2
num_inputs = 2
# 训练数据集样本1数为 1000
num_examples = 1000
# 使用线性回归模型真实权重
true_w = [2, -3, 4]
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
