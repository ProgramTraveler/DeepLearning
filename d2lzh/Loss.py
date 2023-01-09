# 定义损失函数
def squared_loss(y_hat, y):
    # 这一步就是正常的线性回归的求损失函数的方法
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2