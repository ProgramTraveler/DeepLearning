import mxnet as mx
from mxnet import nd
# 建议使用 GPU 来加速计算
# 我们尝试在 gpu(0) 上创建 NDArray 如果成功则使用 gpu(0) 否则任然使用CPU
def try_gpu():
    try:
        ctx = mx.gpu()
        _ = nd.zeros((1, ), ctx=ctx)
    except mx.base.MXNetError:
        ctx = mx.cpu()

    return ctx