import math
import os
import time
import torch.optim
from torch.autograd import Variable
from torch.utils.data import DataLoader   # 数据加载器
from datetime import datetime
import scipy.ndimage as ndimage
from Dataloader import *                  # 导入DataLoader

from EGAFNet import *

from torchvision.utils import save_image  # 导入save_image包

# 以下三行是需要更改的参数
num_epochs = 100
name = 'EGAFNet5e04'
learning_rate = 5e-04


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    # 是否用cuda
print(device)
# net = UNet(3, 1)         # 将网络放到设备上
net = EGAF()
# net = res2net50_v1b_26w_4s(pretrained=False)
# net = CPSCNet(3, 1)
net.to(device)

criterion = nn.BCELoss()                      # 损失函数BCE
# criterion = dice_bce_loss()                 # 损失函数BCE
# criterion = SoftDiceLoss()
# criterion = MulticlassDiceLoss()

print("本次训练轮数: ", num_epochs)
print("本次训练初始学习率：", learning_rate)
print("本次训练损失函数：", criterion)
print("训练开始！！！")


out_edge_path = f'outedge/{name}'
params_path = f'params/{name}'
save_path = f'train_images/{name}'                   # 训练过程图片保存路径
if not os.path.isdir(params_path):  # os.mkdir()创建路径中的最后一级目录，而如果之前的目录不存在并且也需要创建的话，就会报错。
    os.makedirs(params_path)        # os.makedirs()创建多层目录，如果中间目录都不存在的话，会自动创建。
    print("创建文件夹成功：", params_path)
if not os.path.isdir(save_path):
    os.makedirs(save_path)
    print("创建文件夹成功：", save_path)
if not os.path.isdir(out_edge_path):
    os.makedirs(out_edge_path)
    print("创建文件夹成功：", out_edge_path)

weight_path = f'{params_path}/Net_100.pth'
if os.path.exists(weight_path):   # 权重是否在，在的话就加载
    net.load_state_dict(torch.load(weight_path))
    print('Successfully load weight！！！')
else:                             # 不在就不加载
    print('Not successful load weight！！！')



train_path1 = r'E:\Dataset\WHU512\train\image'    # 训练原图路径，r转义
train_path2 = r'E:\Dataset\WHU512\train\label'    # 训练标签路径，r转义
train_path3 = r'E:\Dataset\WHU512\train\edge'
test_path1 = r'E:\Dataset\WHU512\test\image1'    # 测试原图路径，r转义
test_path2 = r'E:\Dataset\WHU512\test\label1'    # 测试原图路径，r转义
test_path3 = r'E:\Dataset\WHU512\test\edge'
# train_path1 = r'E:\Dataset\China500\train\image'    # 训练原图路径，r转义
# train_path2 = r'E:\Dataset\China500\train\label'    # 训练标签路径，r转义
# train_path3 = r'E:\Dataset\China500\train\edge'
# test_path1 = r'E:\Dataset\China500\test\image'    # 测试原图路径，r转义
# test_path2 = r'E:\Dataset\China500\test\label'    # 测试原图路径，r转义
# test_path3 = r'E:\Dataset\China500\test\edge'


train_loader = DataLoader(MyDataset(train_path1, train_path2, train_path3), batch_size=4, shuffle=True)
train_loader2 = DataLoader(MyDataset(test_path1, test_path2, test_path3), batch_size=4, shuffle=True)


optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)    # 优化器Adam，放进网络参数，学习率
lr_scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=10,eta_min=0,last_epoch=-1)

if __name__ == '__main__':
    best_train_loss = float('inf')   # best_loss统计，初始化为正无穷
    best_test_loss = float('inf')
    global_step = 0

    start = time.time()  # 记时开始

    print("start training....")
    prev_time = datetime.now()

    for epoch in range(num_epochs):         # train(epoch)，遍历数据集100次
        train_running_loss = 0
        train_correct = 0
        train_total = 0
        for i, (image, label, edge) in enumerate(train_loader):
            image, label, edge = image.to(device), label.to(device), edge.to(device)  # 将原图和标签都放进设备上
            out_image, out_edge, out3, out4, out5 = net(image)                    # 前向传播，将原图输入网络，得到输出

            loss1 = criterion(out_image, label)
            loss2 = criterion(out_edge, edge)        # 计算损失
            loss3 = criterion(out3, label)
            loss4 = criterion(out4, label)
            loss5 = criterion(out5, label)
            loss = 0.4 * loss1 + 0.3 * loss2 + 0.1 * loss3 + 0.1 * loss4 + 0.1 * loss5

            optimizer.zero_grad()                   # 置零参数的梯度
            loss.backward()                         # 反向传播
            optimizer.step()                        # 优化,更新权重和偏置值，即w和b

            global_step += 1


            train_running_loss += loss.item()       # 记录运行的损失值（loss）
            with torch.no_grad():
                pred_img = torch.argmax(out_image, dim=1)
                train_correct += (pred_img == label).sum().item()
                train_total += label.size(0)

            if i % 5 == 0:  # 每处理5个batch_size输出一次权重
                print(f'epoch:{epoch + 1}-{i}-train_loss===>>{loss.item()}---loss1===>>{loss1.item()}--Edge_loss===>>{loss2.item()}')

            if i % 50 == 0:        # 每处理50个batch_size保存一次权重，文件覆盖
                torch.save(net.state_dict(), f'{params_path}/Net50batchsize.pth')

            if loss < best_train_loss:         # 保存loss值最小的网络参数，文件覆盖
                best_train_loss = loss
                print(f"当前最佳训练Loss=================>{loss.item()}")
                torch.save(net.state_dict(), f'{params_path}/best_train_loss.pth')

            _label = label[0]  # 看训练过程中的效果--标签--输出
            _out_image = out_image[0]
            _edge = edge[0]
            _out_edge = out_edge[0]


            img1 = torch.stack([_label, _out_image], dim=0)  # 将标签和输出结果进行拼接
            save_image(img1, f'{save_path}/{i}.png')  # 将拼接结果image保存，路径为save_path，格式为png
            img2 = torch.stack([_edge, _out_edge], dim=0)  # 将标签和输出结果进行拼接
            save_image(img2, f'{save_path}/{i}.jpg')
        print("第%d个epoch的学习率：%f" % (epoch+1, optimizer.param_groups[0]['lr']))
        lr_scheduler.step()

        test_correct = 0
        test_total = 0
        test_running_loss = 0

        with torch.no_grad():
            for i, (image, label, edge) in enumerate(train_loader2):
                image, label, edge = image.to(device), label.to(device), edge.to(device)
                ppred, pred_edge, pred3, pred4, pred5 = net(image)
                loss1 = criterion(ppred, label)
                loss2 = criterion(pred_edge, edge)  # 计算损失
                loss3 = criterion(pred3, label)
                loss4 = criterion(pred4, label)
                loss5 = criterion(pred5, label)
                loss = 0.4 * loss1 + 0.3 * loss2 + 0.1 * loss3 + 0.1 * loss4 + 0.1 * loss5
                ppred = torch.argmax(ppred, dim=1)
                test_correct += (ppred == label).sum().item()
                test_total += label.size(0)
                test_running_loss += loss.item()

                if loss < best_test_loss:   # 保存loss值最小的网络参数，文件覆盖
                    best_test_loss = loss
                    print(f"当前最佳测试Loss=================>{loss.item()}")
                    torch.save(net.state_dict(), f'{params_path}/best_test_loss.pth')

        if epoch % 10 == 9:     # 每隔10轮次保存一个权重文件，文件不覆盖
            torch.save(net.state_dict(), f'{params_path}/Net_%d.pth' % (epoch + 1))

        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)
        prev_time = cur_time
        print('第 %d epoch平均train loss = %f, 平均test loss = %f, %s'
              % (epoch+1, train_running_loss / len(train_loader), test_running_loss / len(train_loader2), time_str))

    end = time.time()  # 计时结束
    print('训练结束！！！')
    print('本次训练时间: ', end - start, 's, ', (end - start) / 60, 'min, ', (end - start) / 3600, 'h')
    print('本次训练时间:  {:.5f} s, {:.5f} min, {:.5f} h'.format(end - start, (end - start) / 60, (end - start) / 3600))
    print('权重已保存在:', params_path)
    print('Over')