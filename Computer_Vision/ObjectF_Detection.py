# 目标检测和边界框
import sys
sys.path.append('/root/Deep_Pro/')
from mxnet import image
from d2lzh import Set_Figsize as sf
import matplotlib.pyplot as plt


def bbox_to_rect(bbox, color):
    # 将边界框 (左上 x, 左上 y, 右下 x, 右下 y) 格式装换为 matplotlib 格式
    # ((左上 x, 左上 y), 宽, 高)
    return plt.Rectangle(
        xy=(bbox[0], bbox[1]), width=bbox[2] - bbox[0], height=bbox[3] - bbox[1],
        fill=False, edgecolor=color, linewidth=2
    )

    
sf.set_figsize()
img = image.imread('../img/cat_dog.jpg')
# plt.imshow(img)    # 加分号只显示图
plt.imshow(img.asnumpy())
plt.show()

# 没找到猫和狗的图片
# 随便凑合一下看下效果
dog_bbox, cat_bbox = [60, 45, 378, 516], [400, 112, 255, 493]

# fig = plt.imshow(img)
fig = plt.imshow(img.asnumpy())
fig.axes.add_patch(bbox_to_rect(dog_bbox, 'blue'))
fig.axes.add_patch(bbox_to_rect(cat_bbox, 'red'))

plt.show()

