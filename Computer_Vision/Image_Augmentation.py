# 图像增广
import mxnet as mx
from mxnet import autograd, gluon, image, init, nd
from mxnet.gluon import data as gdata, loss as gloss, utils as gutils
import sys
sys.path.append('/root/Deep_Pro/')
import time
from d2lzh import Set_Figsize as sf
import matplotlib.pyplot as plt

def show_images(imgs, num_rows, num_cols, scale=2):
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    for i in range(num_rows):
        for j in range(num_cols):
            axes[i][j].imshow(imgs[i * num_cols + j].asnumpy())
            axes[i][j].axes.get_xaxis().set_visible(False)
            axes[i][j].axes.get_yaxis().set_visible(False)
        return axes
  
    
# 辅助函数 对输入的图像 img 多次运行图像增广方法 aug 并展示所有结果
def apply(img, aug, num_rows=2, num_cols=4, scale=1.5):
    # 按增广方法绘制 num_rows * num_cols 个图像
    Y = [aug(img) for _ in range(num_rows * num_cols)]
    show_images(Y, num_rows, num_cols, scale)
    plt.show()
    
    
# 读取一个形状为 400 * 500 的图像作为实验的样例
sf.set_figsize()
img = image.imread('../img/cat1.jpg')
# 转换为 NumPy 实例
plt.imshow(img.asnumpy())
print(plt.imshow(img.asnumpy()))
# 显示图片
plt.show()

# 左右翻转
# 左右翻转通常不改变物体的类别
# 实现一半概率的图像左右翻转
apply(img, gdata.vision.transforms.RandomFlipLeftRight())

# 上下翻转不如左右翻转通用
apply(img, gdata.vision.transforms.RandomFlipTopBottom())

# 切割
# 通过对图像随机裁剪来让物体以不同的比例出现在图像的不同位置
# 这可以降低模型对目标位置的敏感度
shape_aug = gdata.vision.transforms.RandomResizedCrop((200, 200), scale=(0.1, 1), ratio=(0.5, 2))
apply(img, shape_aug)

# 变化颜色
# 一般从四个方面来改变图像颜色 亮度 对比度 饱和度 色调
# 随机变化图片亮度
apply(img, gdata.vision.transforms.RandomBrightness(0.5))

# 随机变化图像的色调
apply(img, gdata.vision.transforms.RandomHue(0.5))

# 可以同时设置如何随机变化图像的亮度 对比度 饱和度 色调
color_agu = gdata.vision.transforms.RandomColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
apply(img, color_agu)

# 叠加多个图像增广的方法
augs = gdata.vision.transforms.Compose([gdata.vision.transforms.RandomFlipLeftRight(),
                                        color_agu, shape_aug])
apply(img, augs)