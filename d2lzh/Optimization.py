# 定义优化算法
# 在该函数中实现的是小批量随机梯度下降算法
def sgd(params, lr, batch_size):
    # 通过不断迭代模型参数来优化损失函数 这里自动求梯度模块计算的来的梯度是一个批量的梯度和 将它除以批量的大小来得到平均值
    for param in params:
        param[:] = param - lr * param.grad / batch_size