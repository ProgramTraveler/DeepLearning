# 读取数据集

import random
from mxnet import nd


def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # 样品的读取顺序是随机的
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        j = nd.array(indices[i: min(i + batch_size, num_examples)])
        # take 函数根据索引返回对应元素
        yield features.take(j), labels.take(j)
