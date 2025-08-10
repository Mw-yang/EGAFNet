from torchvision.utils import save_image
import cv2
from Dataloader3whu import *
import numpy as np
from matplotlib import pyplot as plt
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from EGAFNet import *
import os
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)             # cuda或者CPU跑程序
# net = UNet(3, 1)   # 将网络放到设备上
net = EGAF()
# net = res2net50_v1b_26w_4s(pretrained=False)
# net = CPSCNet(3, 1)

net.to(device)

criterion = nn.BCELoss()
# criterion = dice_bce_loss()
# criterion = SoftDiceLoss()

name = 'EGAFNet5e04'
pth = 'Net_100.pth'

print("本次测试权重: ", pth)
print("本次测试损失函数：", criterion)
weight_path = f'params/{name}/{pth}'
test_path = f'test_images/{name}'
eval_path = f'threshold/{name}'
testout_edge_path = f'testoutedge/{name}'

if not os.path.isdir(test_path):
    os.makedirs(test_path)
    print("创建文件夹成功：", test_path)
if not os.path.isdir(eval_path):
    os.makedirs(eval_path)
    print("创建文件夹成功：", eval_path)
if not os.path.isdir(testout_edge_path):
    os.makedirs(testout_edge_path)
    print("创建文件夹成功：", testout_edge_path)

if os.path.exists(weight_path):
    net.load_state_dict(torch.load(weight_path))      # 当在GPU上训练，在CPU测试时，需要加上map_location='cpu'
    print('Successfully load weight！！！')
else:
    print('Not successful load weight！！！')             # 权重是否加载



path1 = r'E:\Dataset\WHU512\test\image1'    # 测试原图路径，r转义
path2 = r'E:\Dataset\WHU512\test\label1'   # 测试标签路径
path3 = r'E:\Dataset\WHU512\test\edge'
# path1 = r'E:\code\Dataset\China500\test\image'    # 测试原图路径，r转义
# path2 = r'E:\code\Dataset\China500\test\label'   # 测试标签路径
# path3 = r'E:\code\Dataset\China500\test\edge'
BuildDataset = MyDataset_test(path1, path2, path3)                                        # 将原图和标签传入MyDataset
test_loader = DataLoader(BuildDataset, batch_size=1)


def test():
    totalLoss = 0
    with torch.no_grad():
        step = 0
        for data in test_loader:
            step += 1
            val_images, val_labels, val_edges = data
            val_images, val_labels, val_edges = val_images.to(device), val_labels.to(device), val_edges.to(device)
            outputs, outedges, out3, out4, out5 = net(val_images)
            loss1 = criterion(outputs, val_labels)
            loss2 = criterion(outedges, val_edges)
            loss3 = criterion(out3, val_labels)
            loss4 = criterion(out4, val_labels)
            loss5 = criterion(out5, val_labels)
            loss = 0.4 * loss1 + 0.3 * loss2 + 0.1 * loss3 + 0.1 * loss4 + 0.1 * loss5
            totalLoss += loss.item()
            print("%d,test_loss:%0.3f" % (step, loss.item()))
            img_y = torch.squeeze(outputs).cpu().numpy()
            img_edge = torch.squeeze(outedges).cpu().numpy()

            yy = Image.fromarray((img_y * 255.0).astype(np.uint8))      # np.uint8()来转换类型
            eedge = Image.fromarray((img_edge * 255.0).astype(np.uint8))
            yy.save(f'{test_path}/%d.png' % step)
            eedge.save(f'{testout_edge_path}/%d.png' % step)


if __name__ == '__main__':
    net.eval()
    test()