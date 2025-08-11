import random
import os
import PIL.Image
import cv2
import numpy as np
import torch
import torch.utils.data as data
from matplotlib import pyplot as plt
from torchvision import transforms


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')       # CUDA或者CPU跑程序

def make_dataset1(root1, root2, root3):     # root1为原图地址，root2为标签地址
    imgs = []                        # 遍历文件夹，添加图片和标签图片路径到列表
    n = len(os.listdir(root1))
    for i in range(1, n + 1):
        # img = os.path.join(root1, "%d.tif" % i)    # 后缀要根据数据集的情况进行修改
        # mask = os.path.join(root2, "%d.tif" % i)   # os.path.join()只起连接文件的作用，拼接路径。详见https://blog.csdn.net/MclarenSenna/article/details/117046027
        # img = os.path.join(root1, "%d.bmp" % i)  # 后缀要根据数据集的情况进行修改
        # mask = os.path.join(root2, "%d.bmp" % i)
        # edge = os.path.join(root3, "%d.bmp" % i)
        # img = os.path.join(root1, "%d.tif" % i)  # 后缀要根据数据集的情况进行修改
        # mask = os.path.join(root2, "%d.tif" % i)
        # edge = os.path.join(root3, "%d.tif" % i)
        # img = os.path.join(root1, "%d.png" % i)  # 后缀要根据数据集的情况进行修改
        # mask = os.path.join(root2, "%d.png" % i)
        # edge = os.path.join(root3, "%d.png" % i)
        img = os.path.join(root1, "%d.tif" % i)  # 后缀要根据数据集的情况进行修改
        mask = os.path.join(root2, "%d.png" % i)
        edge = os.path.join(root3, "%d.png" % i)
        imgs.append((img, mask, edge))
    return imgs


class MyDataset(data.Dataset):     # 定义自己的数据集
    '''
    transform: 对src做归一化和标准差处理, 数据最后转换成tensor
    target_transform: 不做处理, label为0/1/2/3(long型)..., 数据最后转换成tensor
    '''

    def __init__(self, root1, root2, root3, transform =None, target_transform=None):   # 类的初始化执行函数，读入原图、标签，传入两种转换的目录。
        imgs = make_dataset1(root1, root2, root3)
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform      # 不做处理，label

    def __getitem__(self, index):      # 在给定索引上加载并返回数据集中的图片。基于该索引，它识别图像文件位置、转为张量，调用适用的转换功能等。
        image_path, label_path, edge_path = self.imgs[index]           # 将imgs获取的root1和root2地址传入
        image = PIL.Image.open(image_path)
        label = PIL.Image.open(label_path)
        edge = PIL.Image.open(edge_path)

        p1 = np.random.choice([0, 1])  # 在0，1二者中随机取一个，作为一个概率值
        p2 = np.random.choice([0, 1])
        p3 = np.random.choice([0, 0.5])
        transform = transforms.Compose([                                        # transforms.Compose将transforms组合在一起，形成一个transform的序列，一起打包执行。
            transforms.Resize((256, 256)),                                    # 按照比例把图像最小的一个边长放缩到256，另一边按照相同比例放缩。会损失太多信息，所以不使用。
            # transforms.RandomResizedCrop(size=256, scale=(0.08, 1)),          # 随机选择(0.08,1)中的一个比例缩放，然后随机裁剪出 (256,256)大小的图片.
            # transforms.RandomCrop(size=256, padding=64, padding_mode='reflect'),# 在上下左右进行64的padding，使用镜像填充，然后随机裁剪出(256,256)大小的图片。
            transforms.RandomHorizontalFlip(p1),                                # 随机水平翻转,概率为p1
            transforms.RandomVerticalFlip(p2),                                  # 随机垂直旋转,概率为p2
            transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1),  # 调整亮度、对比度、饱和度、色相
            transforms.RandomGrayscale(p3),                                     # 随机转换为灰度图，概率为p3
            transforms.ToTensor()                                               # 转化为Tensor，归一化至[0,1],为神经网络提供标准输入
        ])                                                                      # 使用定义好的transform就可以按照循序处理各个transforms的要求

        NM = transforms.Normalize(mean=[0.43527067, 0.4452293, 0.41307566],  # (WHU512)对像素值进行归一化处理
                                  std=[0.21683636, 0.20339176, 0.21733028])
        # NM = transforms.Normalize(mean=[0.3715939, 0.37807724, 0.38623255],  # (China500)对像素值进行归一化处理
        #                           std=[0.22545317, 0.20468025, 0.1955668])

        image = transform(image)
        image = NM(image)
        label = transform(label)
        edge = transform(edge)

        if self.transform is not None:
            image = self.transform(image)  # 归一化、标准差
        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label, edge    # 对原图和标签都transform

    def __len__(self):                               # 返回数据集中样本数量。
        return len(self.imgs)

class MyDataset_test(data.Dataset):     # 定义自己的数据集
    '''
    transform: 对src做归一化和标准差处理, 数据最后转换成tensor
    target_transform: 不做处理, label为0/1/2/3(long型)..., 数据最后转换成tensor
    '''

    def __init__(self, root1, root2, root3, transform =None, target_transform=None):   # 类的初始化执行函数，读入原图、标签，传入两种转换的目录。
        imgs = make_dataset1(root1, root2, root3)
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform      # 不做处理，label

    def __getitem__(self, index):      # 在给定索引上加载并返回数据集中的图片。基于该索引，它识别图像文件位置、转为张量，调用适用的转换功能等。
        image_path, label_path, edge_path = self.imgs[index]           # 将imgs获取的root1和root2地址传入
        image = PIL.Image.open(image_path)
        label = PIL.Image.open(label_path)
        edge = PIL.Image.open(edge_path)

        # p1 = np.random.choice([0, 1])  # 在0，1二者中随机取一个，作为一个概率值
        # p2 = np.random.choice([0, 1])
        # p3 = np.random.choice([0, 0.5])
        transform = transforms.Compose([                                        # transforms.Compose将transforms组合在一起，形成一个transform的序列，一起打包执行。
            transforms.Resize((256, 256)),                                    # 按照比例把图像最小的一个边长放缩到256，另一边按照相同比例放缩。会损失太多信息，所以不使用。
            # transforms.RandomResizedCrop(size=256, scale=(0.08, 1)),          # 随机选择(0.08,1)中的一个比例缩放，然后随机裁剪出 (256,256)大小的图片.
            # transforms.RandomCrop(size=256, padding=64, padding_mode='reflect'),# 在上下左右进行64的padding，使用镜像填充，然后随机裁剪出(256,256)大小的图片。
            # transforms.RandomHorizontalFlip(p1),                                # 随机水平翻转,概率为p1
            # transforms.RandomVerticalFlip(p2),                                  # 随机垂直旋转,概率为p2
            # transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1),  # 调整亮度、对比度、饱和度、色相
            # transforms.RandomGrayscale(p3),                                     # 随机转换为灰度图，概率为p3
            transforms.ToTensor()                                               # 转化为Tensor，归一化至[0,1],为神经网络提供标准输入
        ])                                                                      # 使用定义好的transform就可以按照循序处理各个transforms的要求


        # NM = transforms.Normalize(mean=[0.43527067, 0.4452293, 0.41307566],  # (WHU512)对像素值进行归一化处理
        #                           std=[0.21683636, 0.20339176, 0.21733028])
        NM = transforms.Normalize(mean=[0.3715939, 0.37807724, 0.38623255],  # (China500)对像素值进行归一化处理
                                  std=[0.22545317, 0.20468025, 0.1955668])

        image = transform(image)
        image = NM(image)
        label = transform(label)
        edge = transform(edge)

        if self.transform is not None:
            image = self.transform(image)  # 归一化、标准差
        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label, edge    # 对原图和标签都transform

    def __len__(self):                               # 返回数据集中样本数量。
        return len(self.imgs)


# if __name__ == '__main__':           # 打印出图片或者标签的数量
#     num = len(os.listdir(r'C:\Users\Administrator\Desktop\U_Net\Cheng_Road_Dataset\Train\image'))
#     print("数据个数：", num)