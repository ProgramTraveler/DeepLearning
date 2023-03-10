
from mxnet import nd

x = nd.arange(16).reshape(1, 1, 4 , 4)

print(x.shape)

x[1, 1] = 8
# x.reshape(4, 4)
