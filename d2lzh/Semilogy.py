# 拟合函数中的画图函数
# 其中 y 轴使用了对数尺度

from matplotlib import pyplot as plt
from d2lzh import Set_Figsize as sf

def semilogy(x_vals, y_vals, x_label, y_label, x2_vals=None, y2_vals=None, legend=None, figsize=(3.5, 2.5)):
    sf.set_figsize(figsize)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.semilogy(x_vals, y_vals)

    if x2_vals and y2_vals:
        plt.semilogy(x2_vals, y2_vals, linestyle=':')
        plt.legend(legend)
        # 显示图像
        plt.show()