import mxnet as mx
from mxnet import nd

def try_all_gpus():
    ctxes = []
    try:
        for i in range(16):  # 假设一台机器上 GPU 的数量不超过 16
            ctx = mx.gpu(i)
            _ = nd.array([0], ctx=ctx)
            ctxes.append(ctx)
    except mx.base.MXNetError:
        pass
    if not ctxes:
        ctxes = [mx.cpu()]
    
    return ctxes