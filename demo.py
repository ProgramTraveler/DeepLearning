
from mxnet import nd
from mxnet.gluon import nn

def fun() :
    return 1, 2


x = fun()
print(*x)