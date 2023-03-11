
from mxnet import nd
from mxnet.gluon import nn

x = nd.arange(16).reshape(1, 1, 4 , 4)

print(x.shape)
print(x)
# x.reshape((4, 4))
x[0][0][1, 1] = 18
print(x)

pool2d = nn.MaxPool2D(3, padding=1, strides=2)
print(pool2d(x))