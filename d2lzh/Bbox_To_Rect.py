import matplotlib.pyplot as plt

def bbox_to_rect(bbox, color):
    # 将边界框 (左上 x, 左上 y, 右下 x, 右下 y) 格式装换为 matplotlib 格式
    # ((左上 x, 左上 y), 宽, 高)
    return plt.Rectangle(
        xy=(bbox[0], bbox[1]), width=bbox[2] - bbox[0], height=bbox[3] - bbox[1],
        fill=False, edgecolor=color, linewidth=2
    )