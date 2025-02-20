import torch
from torch import nn
from torch.nn import functional as F
from functools import partial

class Gate(nn.Module):
    def __init__(self, in_channels):
        super(Gate, self).__init__()
        self.psi = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        mid = g + x
        mid = self.relu(mid)
        mid = self.psi(mid)
        up = g * mid
        down = x * up
        out = up + down

        return out

class ConvBnRelu(nn.Module):
    def __init__(self, in_planes, out_planes, ksize, stride, pad, dilation=1,
                 groups=1, has_bn=True, norm_layer=nn.BatchNorm2d,
                 has_relu=True, inplace=True, has_bias=False):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=ksize,
                              stride=stride, padding=pad,
                              dilation=dilation, groups=groups, bias=has_bias)
        self.has_bn = has_bn
        if self.has_bn:
            self.bn = nn.BatchNorm2d(out_planes)
        self.has_relu = has_relu
        if self.has_relu:
            self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        x = self.conv(x)
        if self.has_bn:
            x = self.bn(x)
        if self.has_relu:
            x = self.relu(x)

        return x


class SAFM(nn.Module):
    def __init__(self, in_channels):
        super(SAFM, self).__init__()
        self.conv3x3 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, dilation=1, kernel_size=3,
                                 padding=1)
        self.AG1 = Gate(in_channels)
        self.AG2 = Gate(in_channels)
        self.AG3 = Gate(in_channels)
        self.bn = nn.ModuleList([nn.BatchNorm2d(in_channels), nn.BatchNorm2d(in_channels), nn.BatchNorm2d(in_channels), nn.BatchNorm2d(in_channels)])
        self.conv1x1 = nn.ModuleList(
            [nn.Conv2d(in_channels=2 * in_channels, out_channels=in_channels, dilation=1, kernel_size=1, padding=0),
             nn.Conv2d(in_channels=2 * in_channels, out_channels=in_channels, dilation=1, kernel_size=1, padding=0)])
        self.conv3x3_1 = nn.ModuleList(
            [nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 2, dilation=1, kernel_size=3, padding=1),
             nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 2, dilation=1, kernel_size=3, padding=1)])
        self.conv3x3_2 = nn.ModuleList(
            [nn.Conv2d(in_channels=in_channels // 2, out_channels=2, dilation=1, kernel_size=3, padding=1),
             nn.Conv2d(in_channels=in_channels // 2, out_channels=2, dilation=1, kernel_size=3, padding=1)])
        self.conv_last = ConvBnRelu(in_planes=in_channels, out_planes=in_channels, ksize=1, stride=1, pad=0, dilation=1)
        self.norm = nn.Sigmoid()
        self.conv1 = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, padding=0)
        self.dconv1 = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, padding=0)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        branches_1 = self.conv3x3(x)
        branches_1 = self.bn[0](branches_1)

        branches_2 = F.conv2d(x, self.conv3x3.weight, padding=2, dilation=2)  # share weight
        branches_2 = self.bn[1](branches_2)

        branches_3 = F.conv2d(x, self.conv3x3.weight, padding=4, dilation=4)  # share weight
        branches_3 = self.bn[2](branches_3)

        branches_4 = F.conv2d(x, self.conv3x3.weight, padding=8, dilation=8)  # share weight
        branches_4 = self.bn[3](branches_4)

        fusion_1_2 = self.AG1(branches_1, branches_2)

        fusion_2_3 = self.AG2(fusion_1_2, branches_3)

        fusion_3_4 = self.AG3(fusion_2_3, branches_4)

        alpha = self.norm(self.gamma)
        ax = self.relu(alpha * fusion_3_4 + (1 - alpha) * x)
        ax = self.conv_last(ax)

        return ax


class Double_conv_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Double_conv_block, self).__init__()
        self.conv = nn.Sequential(

            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class Deep_conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Deep_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=64),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class Deepsupervision(nn.Module):
    def __init__(self, in_channels, out_channels, factor):
        super(Deepsupervision, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=64),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=64),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, 1, kernel_size=1, stride=1, padding=0),
            nn.Upsample(scale_factor=factor)
        )


    def forward(self, x):
        return self.conv(x)


class MFE(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MFE, self).__init__()
        self.Branch1 = nn.Sequential(nn.Conv2d(in_channels, in_channels // 8, kernel_size=1, stride=1, padding=0),
                                     nn.BatchNorm2d(in_channels // 8),
                                     nn.ReLU(True))
        self.Branch2 = nn.Sequential(nn.Conv2d(in_channels, in_channels // 8, kernel_size=3, stride=1, padding=1),
                                     nn.BatchNorm2d(in_channels // 8),
                                     nn.ReLU(True))
        self.Branch3 = nn.Sequential(nn.Conv2d(in_channels, in_channels // 4, (3, 1), 1, (1, 0)),
                                     nn.ReLU(True),
                                     nn.Conv2d(in_channels // 4, in_channels // 4, (1, 3), 1, (0, 1)),
                                     nn.BatchNorm2d(in_channels // 4),
                                     nn.ReLU(True))
        self.Branch4 = nn.Sequential(nn.Conv2d(in_channels, in_channels // 2, (5, 1), 1, (2, 0)),
                                     nn.ReLU(True),
                                     nn.Conv2d(in_channels // 2, in_channels // 2, (1, 5), 1, (0, 2)),
                                     nn.BatchNorm2d(in_channels // 2),
                                     nn.ReLU(True))
        self.conv11 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1)))

    def forward(self, x):
        x1 = self.Branch1(x)
        x2 = self.Branch2(x)
        x3 = self.Branch3(x)
        x4 = self.Branch4(x)
        xc = torch.cat((x1, x2, x3, x4), dim=1)
        out = xc + x
        out = self.conv11(out)

        return out

# -------------------------------------- #
# Stochastic Depth dropout 方法，随机丢弃输出层
# -------------------------------------- #

def drop_path(x, drop_prob: float = 0., training: bool = False):  # drop_prob代表丢弃概率
    # （1）测试时不做 drop path 方法
    if drop_prob == 0. or training is False:
        return x
    # （2）训练时使用
    keep_prob = 1 - drop_prob  # 网络每个特征层的输出结果的保留概率
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        out = drop_path(x, self.drop_prob, self.training)
        return out

# -------------------------------------- #
# SE注意力机制
# -------------------------------------- #

class SequeezeExcite(nn.Module):
    def __init__(self,
                 input_c,  # 输入到MBConv模块的特征图的通道数
                 expand_c,  # 输入到SE模块的特征图的通道数
                 se_ratio=0.25,  # 第一个全连接下降的通道数的倍率
                 ):
        super(SequeezeExcite, self).__init__()

        # 第一个全连接下降的通道数
        sequeeze_c = int(input_c * se_ratio)
        # 1*1卷积代替全连接下降通道数
        self.conv_reduce = nn.Conv2d(expand_c, sequeeze_c, kernel_size=1, stride=1)
        self.act1 = nn.SiLU()
        # 1*1卷积上升通道数
        self.conv_expand = nn.Conv2d(sequeeze_c, expand_c, kernel_size=1, stride=1)
        self.act2 = nn.Sigmoid()

    # 前向传播
    def forward(self, x):
        # 全局平均池化[b,c,h,w]==>[b,c,1,1]
        scale = x.mean((2, 3), keepdim=True)
        scale = self.conv_reduce(scale)
        scale = self.act1(scale)
        scale = self.conv_expand(scale)
        scale = self.act2(scale)
        # 对输入特征图x的每个通道加权
        return scale * x

# -------------------------------------- #
# 卷积 + BN + 激活
# -------------------------------------- #

class ConvBnAct(nn.Module):
    def __init__(self,
                 in_planes,  # 输入特征图通道数
                 out_planes,  # 输出特征图通道数
                 kernel_size=3,  # 卷积核大小
                 stride=1,  # 滑动窗口步长
                 groups=1,  # 卷积时通道数分组的个数
                 norm_layer=None,  # 标准化方法
                 activation_layer=None,  # 激活函数
                 ):
        super(ConvBnAct, self).__init__()

        # 计算不同卷积核需要的0填充个数
        padding = (kernel_size - 1) // 2
        # 若不指定标准化和激活函数，就用默认的
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.SiLU

        # 卷积
        self.conv = nn.Conv2d(in_channels=in_planes,
                              out_channels=out_planes,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              groups=groups,
                              bias=False,
                              )
        # BN
        self.bn = norm_layer(out_planes)
        # silu
        self.act = activation_layer(inplace=True)

    # 前向传播
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


# -------------------------------------- #
# MBConv卷积块
# -------------------------------------- #

class MBConv(nn.Module):
    def __init__(self,
                 input_c,
                 output_c,
                 kernel_size,  # DW卷积的卷积核size
                 expand_ratio,  # 第一个1*1卷积上升通道数的倍率
                 stride,  # DW卷积的步长
                 se_ratio,  # SE模块的第一个全连接层下降通道数的倍率
                 drop_rate,  # 随机丢弃输出层的概率
                 norm_layer,
                 ):
        super(MBConv, self).__init__()

        # 下采样模块不存在残差边；基本模块stride=1时且输入==输出特征图通道数，才用到残差边
        self.has_shortcut = (stride == 1 and input_c == output_c)
        # 激活函数
        activation_layer = nn.SiLU
        # 第一个1*1卷积上升的输出通道数
        expanded_c = input_c * expand_ratio

        # 1*1升维卷积
        self.expand_conv = ConvBnAct(in_planes=input_c,  # 输入通道数
                                     out_planes=expanded_c,  # 上升的通道数
                                     kernel_size=1,
                                     stride=1,
                                     groups=1,
                                     norm_layer=norm_layer,
                                     activation_layer=activation_layer,
                                     )
        # DW卷积
        self.dwconv = ConvBnAct(in_planes=expanded_c,
                                out_planes=expanded_c,  # DW卷积输入和输出通道数相同
                                kernel_size=kernel_size,
                                stride=stride,
                                groups=expanded_c,  # 对特征图的每个通道做卷积
                                norm_layer=norm_layer,
                                activation_layer=activation_layer,
                                )
        # SE注意力
        # 如果se_ratio>0就使用SE模块，否则线性连接、
        if se_ratio > 0:
            self.se = SequeezeExcite(input_c=input_c,  # MBConv模块的输入通道数
                                     expand_c=expanded_c,  # SE模块的输出通道数
                                     se_ratio=se_ratio,  # 第一个全连接的通道数下降倍率
                                     )
        else:
            self.se = nn.Identity()

        # 1*1逐点卷积降维
        self.project_conv = ConvBnAct(in_planes=expanded_c,
                                      out_planes=output_c,
                                      kernel_size=1,
                                      stride=1,
                                      groups=1,
                                      norm_layer=norm_layer,
                                      activation_layer=nn.Identity,  # 不使用激活函数，恒等映射
                                      )
        # droppath方法
        self.drop_rate = drop_rate
        # 只在基本模块使用droppath丢弃输出层
        if self.has_shortcut and drop_rate > 0:
            self.dropout = DropPath(drop_prob=drop_rate)

    # 前向传播
    def forward(self, x):
        out = self.expand_conv(x)  # 升维
        out = self.dwconv(out)  # DW
        out = self.se(out)  # 通道注意力
        out = self.project_conv(out)  # 降维

        # 残差边
        if self.has_shortcut:
            if self.drop_rate > 0:
                out = self.dropout(out)  # drop_path方法
            out += x  # 输入和输出短接
        return out

# -------------------------------------- #
# FusedMBConv卷积块
# -------------------------------------- #

class FusedMBConv(nn.Module):
    def __init__(self,
                 input_c,
                 output_c,
                 kernel_size,  # DW卷积核的size
                 expand_ratio,  # 第一个1*1卷积上升的通道数
                 stride,  # DW卷积的步长
                 se_ratio,  # SE模块第一个全连接下降通道数的倍率
                 drop_rate,  # drop—path丢弃输出层的概率
                 norm_layer,
                 ):
        super(FusedMBConv, self).__init__()

        # 残差边只用于基本模块
        self.has_shortcut = (stride == 1 and input_c == output_c)
        # 随机丢弃输出层的概率
        self.drop_rate = drop_rate
        # 第一个卷积是否上升通道数
        self.has_expansion = (expand_ratio != 1)
        # 激活函数
        activation_layer = nn.SiLU

        # 卷积上升的通道数
        expanded_c = input_c * expand_ratio

        # 只有expand_ratio>1时才使用升维卷积
        if self.has_expansion:
            self.expand_conv = ConvBnAct(in_planes=input_c,
                                         out_planes=expanded_c,  # 升维的输出通道数
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         norm_layer=norm_layer,
                                         activation_layer=activation_layer,
                                         )
            # 1*1降维卷积
            self.project_conv = ConvBnAct(in_planes=expanded_c,
                                          out_planes=output_c,
                                          kernel_size=1,
                                          stride=1,
                                          norm_layer=norm_layer,
                                          activation_layer=nn.Identity,
                                          )
        # 如果expand_ratio=1，即第一个卷积不上升通道
        else:
            self.project_conv = ConvBnAct(in_planes=input_c,
                                          out_planes=output_c,
                                          kernel_size=kernel_size,
                                          stride=stride,
                                          norm_layer=norm_layer,
                                          activation_layer=activation_layer,
                                          )

        # 只有在基本模块中才使用shortcut，只有存在shortcut时才能用drop_path
        self.drop_rate = drop_rate
        if self.has_shortcut and drop_rate > 0:
            self.dropout = DropPath(drop_rate)

    # 前向传播
    def forward(self, x):
        # 第一个卷积块上升通道数倍率>1
        if self.has_expansion:
            out = self.expand_conv(x)
            out = self.project_conv(out)
        # 不上升通道数
        else:
            out = self.project_conv(x)

        # 基本模块中使用残差边
        if self.has_shortcut:
            if self.drop_rate > 0:
                out = self.dropout(out)
            out += x
        return out

class Mix(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Mix, self).__init__()
        self.Upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)  # 上采样
        self.Conv_1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.Upsample(x)
        x2 = self.Conv_1x1(x1)

        return x2


# -------------------------------------- #
# 主干网络
# -------------------------------------- #

class EGAFNet(nn.Module):
    def __init__(self,
                 model_cnf: list,  # 配置文件
                 num_classes=1000,  # 分类数
                 num_features=1280,  # 输出层的输入通道数
                 drop_path_rate=0.2,  # 用于卷积块中的drop_path层
                 drop_rate=0.2):  # 输出层的dropout层
        super(EGAFNet, self).__init__()

        # 为BN层传递默认参数
        norm_layer = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.1)

        # 第一个卷积层的输出通道数
        stem_filter_num = model_cnf[0][4]  # 24

        # 第一个卷积层[b,3,h,w]==>[b,24,h//2,w//2]
        self.stem = ConvBnAct(in_planes=3,
                              out_planes=stem_filter_num,
                              kernel_size=3,
                              stride=2,
                              norm_layer=norm_layer,
                              )
        # 统计一共堆叠了多少个卷积块
        total_blocks = sum([i[0] for i in model_cnf])
        block_id = 0
        # blocks = []  # 保存所有的卷积块
        block1 = []
        block2 = []
        block3 = []
        block4 = []
        k = 0
        # 遍历每个stage的参数
        for cnf in model_cnf:
            # 当前stage重复次数
            repeats = cnf[0]
            k += 1
            # 使用何种卷积块，0标记则用FusedMBConv
            op = FusedMBConv if cnf[-2] == 0 else MBConv
            # 堆叠每个stage
            for i in range(repeats):
                if k == 1:
                    block1.append(op(
                        input_c=cnf[4] if i == 0 else cnf[5],  # 只有第一个下采样卷积块的输入通道数需要调整，其余都一样
                        output_c=cnf[5],  # 输出通道数保持一致
                        kernel_size=cnf[1],  # 卷积核size
                        expand_ratio=cnf[3],  # 第一个卷积升维倍率
                        stride=cnf[2] if i == 0 else 1,  # 第一个卷积块可能是下采样stride=2，剩下的都是基本模块
                        se_ratio=cnf[-1],  # SE模块第一个全连接降维倍率
                        drop_rate=drop_path_rate * block_id / total_blocks,  # drop_path丢弃率满足线性关系
                        norm_layer=norm_layer,
                    ))
                    # 没堆叠完一个就计数
                    block_id += 1
                if k == 2:
                    block2.append(op(
                        input_c=cnf[4] if i == 0 else cnf[5],  # 只有第一个下采样卷积块的输入通道数需要调整，其余都一样
                        output_c=cnf[5],  # 输出通道数保持一致
                        kernel_size=cnf[1],  # 卷积核size
                        expand_ratio=cnf[3],  # 第一个卷积升维倍率
                        stride=cnf[2] if i == 0 else 1,  # 第一个卷积块可能是下采样stride=2，剩下的都是基本模块
                        se_ratio=cnf[-1],  # SE模块第一个全连接降维倍率
                        drop_rate=drop_path_rate * block_id / total_blocks,  # drop_path丢弃率满足线性关系
                        norm_layer=norm_layer,
                    ))
                    # 没堆叠完一个就计数
                    block_id += 1
                if k == 3:
                    block3.append(op(
                        input_c=cnf[4] if i == 0 else cnf[5],  # 只有第一个下采样卷积块的输入通道数需要调整，其余都一样
                        output_c=cnf[5],  # 输出通道数保持一致
                        kernel_size=cnf[1],  # 卷积核size
                        expand_ratio=cnf[3],  # 第一个卷积升维倍率
                        stride=cnf[2] if i == 0 else 1,  # 第一个卷积块可能是下采样stride=2，剩下的都是基本模块
                        se_ratio=cnf[-1],  # SE模块第一个全连接降维倍率
                        drop_rate=drop_path_rate * block_id / total_blocks,  # drop_path丢弃率满足线性关系
                        norm_layer=norm_layer,
                    ))
                    # 没堆叠完一个就计数
                    block_id += 1
                if k == 4:
                    block4.append(op(
                        input_c=cnf[4] if i == 0 else cnf[5],  # 只有第一个下采样卷积块的输入通道数需要调整，其余都一样
                        output_c=cnf[5],  # 输出通道数保持一致
                        kernel_size=cnf[1],  # 卷积核size
                        expand_ratio=cnf[3],  # 第一个卷积升维倍率
                        stride=cnf[2] if i == 0 else 1,  # 第一个卷积块可能是下采样stride=2，剩下的都是基本模块
                        se_ratio=cnf[-1],  # SE模块第一个全连接降维倍率
                        drop_rate=drop_path_rate * block_id / total_blocks,  # drop_path丢弃率满足线性关系
                        norm_layer=norm_layer,
                    ))
                    # 没堆叠完一个就计数
                    block_id += 1
            # 以非关键字参数形式返回堆叠后的stage
        # self.blocks = nn.Sequential(*blocks)
        self.blocks1 = nn.Sequential(*block1)
        self.blocks2 = nn.Sequential(*block2)
        self.blocks3 = nn.Sequential(*block3)
        self.blocks4 = nn.Sequential(*block4)

        # 输出层的输入通道数 256
        head_input_c = model_cnf[-1][-3]

        # 输出层
        self.head = nn.Sequential(
            # 1*1卷积 [b,256,h,w]==>[b,1024,h,w]
            ConvBnAct(head_input_c, num_features, kernel_size=1, stride=1, norm_layer=norm_layer),
            # 全剧平均池化[b,1024,h,w]==>[b,1024,1,1]
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),  # [b,1024]
            nn.Dropout(p=drop_rate, inplace=True),
            nn.Linear(num_features, num_classes)  # [b,1000]
        )

        # ----------------------------------------- #
        # 权重初始化
        # ----------------------------------------- #

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            # BN层初始化
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            # 全连接层
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

        self.Conv_1_down = nn.Sequential(self.blocks1)
        self.Conv_2_down = nn.Sequential(self.blocks2)
        self.Conv_3_down = nn.Sequential(self.blocks3)
        self.Conv_4_down = nn.Sequential(self.blocks4)
        self.Mixup1 = Mix(512, 256)
        self.Mixup2 = Mix(256, 128)
        self.Mixup3 = Mix(128, 64)
        self.Mixup4 = Mix(64, 64)
        self.Upsample = nn.Upsample(scale_factor=2)
        self.Upsample1 = nn.Upsample(scale_factor=4)
        self.Upsample2 = nn.Upsample(scale_factor=8)
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.SAFM = SAFM(512)
        self.Conv_4_up = nn.Sequential(MFE(in_channels=512, out_channels=256))
        self.Conv_3_up = nn.Sequential(MFE(in_channels=256, out_channels=128))
        self.Conv_2_up = nn.Sequential(MFE(in_channels=128, out_channels=64))
        self.Conv_1_up = nn.Sequential(MFE(in_channels=128, out_channels=64))
        self.skip1 = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=(1, 1), stride=(1, 1))
        self.skip2 = nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(1, 1), stride=(1, 1))
        self.skip3 = nn.Conv2d(in_channels=64, out_channels=16, kernel_size=(1, 1), stride=(1, 1))
        self.skip4 = nn.Conv2d(in_channels=64, out_channels=16, kernel_size=(1, 1), stride=(1, 1))

        self.Conv_1x1 = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=1, stride=1, padding=0)
        self.Th = torch.nn.Sigmoid()

        self.e1 = Double_conv_block(64, 128)
        self.e2 = Double_conv_block(256, 128)
        self.e3 = Double_conv_block(256, 128)
        self.e12 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0)
        self.e23 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=0)
        self.edge = nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, stride=1, padding=0)

        self.gm4 = Deep_conv(384, 256)
        self.gm3 = Deep_conv(256, 128)
        self.gm2 = Deep_conv(192, 64)
        self.gm1 = Deep_conv(192, 64)
        self.d4 = Deepsupervision(256, 64, 8)
        self.d3 = Deepsupervision(128, 64, 4)
        self.d2 = Deepsupervision(64, 64, 2)
        self.d1 = Deepsupervision(64, 64, 1)
        self.ReLU = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.stem(x)
        x2 = self.Conv_1_down(x1)
        x3 = self.Conv_2_down(x2)
        x4 = self.Conv_3_down(x3)
        x5 = self.Conv_4_down(x4)
        x5 = self.SAFM(x5)

        # Edge
        e1 = self.Upsample(x2)
        e1 = self.e1(e1)
        e12 = self.Upsample1(x3)
        e12 = self.e12(e12)
        f1 = torch.cat((e1, e12), dim=1)
        e2 = self.e2(f1)
        e22 = self.Upsample2(x4)
        e22 = self.e23(e22)
        f2 = torch.cat((e2, e22), dim=1)
        f2 = self.e3(f2)
        out2 = self.edge(f2)
        out2 = self.Th(out2)

        y4 = self.Mixup1(x5)
        y4 = torch.cat((x4, y4), dim=1)
        y4 = self.Conv_4_up(y4)

        out5 = self.d4(y4)
        out5 = self.Th(out5)

        # 边界引导4
        g4 = self.Maxpool(self.Maxpool(self.Maxpool(f2)))
        g4 = torch.cat((g4, y4), dim=1)
        g4 = self.gm4(g4)
        g4 = g4 + y4
        g4 = self.ReLU(g4)

        y3 = self.Mixup2(g4)
        y3 = torch.cat((x3, y3), dim=1)
        y3 = self.Conv_3_up(y3)

        out4 = self.d3(y3)
        out4 = self.Th(out4)

        # 边界引导3
        g3 = self.Maxpool(self.Maxpool(f2))
        g3 = torch.cat((g3, y3), dim=1)
        g3 = self.gm3(g3)
        g3 = g3 + y3
        g3 = self.ReLU(g3)

        y2 = self.Mixup3(g3)
        y2 = torch.cat((x2, y2), dim=1)
        y2 = self.Conv_2_up(y2)

        out3 = self.d2(y2)
        out3 = self.Th(out3)

        # 边界引导2
        g2 = self.Maxpool(f2)
        g2 = torch.cat((g2, y2), dim=1)
        g2 = self.gm2(g2)
        g2 = g2 + y2
        g2 = self.ReLU(g2)

        y1 = self.Mixup4(g2)
        x1 = self.Upsample(x1)
        y1 = torch.cat((x1, y1), dim=1)
        y1 = self.Conv_1_up(y1)

        y = self.Conv_1x1(y1)
        out = self.Th(y)

        return out, out2, out3, out4, out5


def EGAF(num_classes=1):
    # 配置文件
    # repeat, kernel, stride, expansion, in_c, out_c, operate, squeeze_rate
    model_config = [[4, 3, 1, 1, 64, 64, 0, 0],
                    [4, 3, 2, 4, 64, 128, 0, 0],
                    [6, 3, 2, 4, 128, 256, 0, 0],
                    [9, 3, 2, 4, 256, 512, 1, 0.25],
                    ]
    model = EGAFNet(model_cnf=model_config,
                           num_classes=num_classes)
    return model

if __name__ == '__main__':
    model = EGAF()
    inputs = torch.rand(1, 3, 256, 256)
    outputs = model(inputs)
    print([output.shape for output in outputs])