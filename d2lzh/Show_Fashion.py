# 在一行画出多张图像和对应标签

import matplotlib.pyplot as plt

from Deep_Learning.d2lzh.Use_Svg_Display import use_svg_display


def show_fashion_mnist(images, labels):
    use_svg_display()
    # 这里的 _ 表示我们忽略 (不使用) 的变量
    _, figs = plt.subplots(1, len(images), figsize=(12, 12))

    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.reshape((28, 28)).asnumpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
