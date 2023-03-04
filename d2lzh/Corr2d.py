
from mxnet import nd

def corr2d(x, k):
    h, w = k.shape

    y = nd.zeros((x.shape[0] - h + 1, x.shape[1] - w + 1))
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            y[i, j] = (x[i: i + h, j: j + w] * k).sum()

    return y