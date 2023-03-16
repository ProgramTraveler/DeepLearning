# 微调
import sys
sys.path.append('/root/Deep_Pro/')
from mxnet import gluon, init, nd
from mxnet.gluon import data as gdata, loss as gloss, model_zoo
from mxnet.gluon import utils as gutils
import os
import zipfile
from d2lzh import Show_Images as si, Try_All_Gpus as tag, Train as t
import matplotlib.pyplot as plt


# 定义一个使用微调的训练模型
def train_fine_tuning(net, learning_rate, batch_size=128, num_epochs=5):
    train_iter = gdata.DataLoader(train_imgs.transform_first(train_augs), batch_size, shuffle=True)
    test_iter = gdata.DataLoader(test_imgs.transform_first(test_augs), batch_size, shuffle=True)
    ctx = tag.try_all_gpus()
    net.collect_params().reset_ctx(ctx)
    net.hybridize()
    loss = gloss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate' : learning_rate, 'wd' : 0.001})
    t.train(train_iter, test_iter, net, loss, trainer, ctx, num_epochs)
    

# 获取数据集
data_dir = '/root/Deep_Pro/data'
# base_url = 'https://apache-mxnet.s3-accelerate.amazonaws.com/'
# fname = gutils.download(base_url + 'gluon/dataset/hotdog.zip',
#                         path=data_dir, sha1_hash='fba480ffa8aa7e0febbb511d181409f899b9baa5')
# with zipfile.ZipFile(fname, 'r') as z:
#     z.extractall(data_dir)
    
# 创建两个实例来分别读取训练数据集和测试数据集
train_imgs = gdata.vision.ImageFolderDataset(os.path.join(data_dir, 'hotdog/train'))
test_imgs = gdata.vision.ImageFolderDataset(os.path.join(data_dir, 'hotdog/test'))
# train_imgs = gdata.vision.ImageFolderDataset('../root/Deep_Por/Computer_Vision/data/hotdog/train/')
# test_imgs = gdata.vision.ImageFolderDataset('../root/Deep_Por/Computer_Vision/data/hotdog/test/')


# 画出前 8 张正类图像和最后 8 张负类图像
hotdogs = [train_imgs[i][0] for i in range(8)]
not_hotdogs = [train_imgs[-i - 1][0] for i in range(8)]
si.show_images(hotdogs + not_hotdogs, 2, 8, scale=1.4)
plt.show()

# 指定 RGB 三个通道的均值和方差来将图像通道归一化
normalize = gdata.vision.transforms.Normalize(
    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

train_augs = gdata.vision.transforms.Compose([
    gdata.vision.transforms.RandomResizedCrop(224),
    gdata.vision.transforms.RandomFlipLeftRight(),
    gdata.vision.transforms.ToTensor(),
    normalize
])

test_augs = gdata.vision.transforms.Compose([
    gdata.vision.transforms.Resize(256),
    gdata.vision.transforms.CenterCrop(224),
    gdata.vision.transforms.ToTensor(),
    normalize
])

# 定义和初始化模型
# 指定 pretrained=True 来自动下载并加载预训练
# 第一次使用需要联网下载模型参数
pretrained_net = model_zoo.vision.resnet18_v2(pretrained=True)

# 预训练的源模型实列含有两个成员变量 即 features 和 output
# 前者包含模型除输出层以外的所有层 后者为模型的输出层
# 这样划分主要是为了方便微调除了输出层以外的所有层的模型参数
print(pretrained_net.output)    # 作为一个全连接层 它将 ResNet 最终的全局平均池化层输出变换成 ImageNet 数据集上的 1000 类的输出

# 新建一个神经网络来作为目标模型
# 定义与预训练的源模型一样 但最后的输出个数等于目标数据集的类别数
# 目标模型实例 finetune_net 的成员变量 features 中的模型参数被初始化为源模型相应层的模型参数
# 成员变量 output 中的模型参数采用随机初始化 一般需要更大的学习率从头训练
finetune_net = model_zoo.vision.resnet18_v2(classes=2)
finetune_net.features = pretrained_net.features
finetune_net.output.initialize(init.Xavier())   # 对最后一层做随机初始化
# output 中的模型参数将在迭代中使用 10 倍大的学习率
finetune_net.output.collect_params().setattr('lr_mult', 10)

# 微调模型
# 我们将 Trainer 实例中的学习率设得小一点以便微调预训练来得到模型参数
# 将以 10 倍学习率从头训练目标模型的输出层参数
train_fine_tuning(finetune_net, 0.01)

# 作为对比
# 可以定义一个相同的模型
# 但将它的所有模型参数都初始化为随机值
# 由于整个模型都需要从头训练 使用较大的学习率
scratch_net = model_zoo.vision.resnet18_v2(classes=2)
scratch_net.initialize(init=init.Xavier())
train_fine_tuning(scratch_net, 0.1)