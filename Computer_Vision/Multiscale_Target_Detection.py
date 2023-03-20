# 多尺度目标检测
import sys
sys.path.append('/root/Deep_Pro/')
from mxnet import contrib, image, nd
from d2lzh import Set_Figsize as sf, Show_Bboxes as sb
import matplotlib.pyplot as plt

def display_anchors(fmap_w, fmap_h, s):
    fmap = nd.zeros((1, 10, fmap_w, fmap_h))     # 前两维的取值不影响输出结果
    anchors = contrib.nd.MultiBoxPrior(fmap, sizes=s, ratios=[1, 2, 0.5])
    bbox_scale = nd.array((w, h, w, h))
    sb.show_bboxes(plt.imshow(img.asnumpy()).axes,
                   anchors[0] * bbox_scale)
    
    
img = image.imread('../img/cat_dog.jpg')
h, w = img.shape[0:2]
print(h, w)

sf.set_figsize()

display_anchors(fmap_w=4, fmap_h=4, s=[0.15])
plt.show()
display_anchors(fmap_w=2, fmap_h=2, s=[0.4])
plt.show()
display_anchors(fmap_w=1, fmap_h=1, s=[0.8])
plt.show()