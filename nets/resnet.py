import math

import torch.nn as nn
import torch
# from torchvision.models.utils import load_state_dict_from_url
from torchsummary import summary
from torch.hub import load_state_dict_from_url


from torch import nn
from torch.utils.data import DataLoader as DataLoader
from torchinfo import summary

import torch.nn as nn
import math
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
import torch


from tqdm import tqdm


# SE注意力模块
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# EAC注意力模块
class ECA_Layer(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super(ECA_Layer, self).__init__()
        k = int(abs((math.log(channels, 2) + b) / gamma))
        kernel_size = k if k % 2 else k + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size // 2), bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)

        # 变为 [1, 1, channels] 方便进行1d卷积
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        y = self.sigmoid(y) * x

        return y

# CBAM注意力模块
class CBAM(nn.Module):
    def __init__(self,in_channel,reduction=16,kernel_size=7):
        super(CBAM, self).__init__()
        #通道注意力机制
        self.max_pool=nn.AdaptiveMaxPool2d(output_size=1)
        self.avg_pool=nn.AdaptiveAvgPool2d(output_size=1)
        self.mlp=nn.Sequential(
            nn.Linear(in_features=in_channel,out_features=in_channel//reduction,bias=False),
            nn.ReLU(),
            nn.Linear(in_features=in_channel//reduction,out_features=in_channel,bias=False)
        )
        self.sigmoid=nn.Sigmoid()
        #空间注意力机制
        self.conv=nn.Conv2d(in_channels=2,out_channels=1,kernel_size=kernel_size ,stride=1,padding=kernel_size//2,bias=False)

    def forward(self,x):
        #通道注意力机制
        maxout=self.max_pool(x)
        maxout=self.mlp(maxout.view(maxout.size(0),-1))
        avgout=self.avg_pool(x)
        avgout=self.mlp(avgout.view(avgout.size(0),-1))
        channel_out=self.sigmoid(maxout+avgout)
        channel_out=channel_out.view(x.size(0),x.size(1),1,1)
        channel_out=channel_out*x
        #空间注意力机制
        max_out,_=torch.max(channel_out,dim=1,keepdim=True)
        mean_out=torch.mean(channel_out,dim=1,keepdim=True)
        out=torch.cat((max_out,mean_out),dim=1)
        out=self.sigmoid(self.conv(out))
        out=out*channel_out
        return out


# 通道注意力机制
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

# 空间注意力机制
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class BasicBlock(nn.Module):  # resnet18和resnet34的主干网络搭建
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """
    注意：原论文中，在虚线残差结构的主分支上，第一个1x1卷积层的步距是2，第二个3x3卷积层步距是1。
    但在pytorch官方实现过程中是第一个1x1卷积层的步距是1，第二个3x3卷积层步距是2，
    这么做的好处是能够在top1上提升大概0.5%的准确率。
    可参考Resnet v1.5 https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch
    """
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None,
                 groups=1, width_per_group=64):
        super(Bottleneck, self).__init__()

        width = int(out_channel * (width_per_group / 64.)) * groups

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width,
                               kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.bn1 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channel * self.expansion,
                               kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.bn3 = nn.BatchNorm2d(out_channel * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, blocks_num, num_classes=1000, include_top=True, groups=1, width_per_group=64):
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64

        self.groups = groups
        self.width_per_group = width_per_group

        # SE attention block
        # self._SE1 = SEBlock(self.feature_extractor.layer2[0].conv1.in_channels)
        # self._SE1 = SEBlock(self._make_layer(block, 128, blocks_num[1]))
        # self._SE2 = SEBlock(self._make_layer(block, 256, blocks_num[2]))
        # self._SE3 = SEBlock(self._make_layer(block, 512, blocks_num[3]))
        # self.last_SE = SEBlock(self.feature_extractor.fc.in_features)

        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)

        # 网络的第一层加入注意力机制
        self.ca = ChannelAttention(self.in_channel)
        self.sa = SpatialAttention()

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        # self._SE1,
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        # self._SE2,
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        # self._SE3,
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)

        # 网络的卷积层的最后一层加入注意力机制
        self.ca1 = ChannelAttention(self.in_channel)
        self.sa1 = SpatialAttention()

        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion)
            )

        layers = []
        layers.append(block(self.in_channel, channel, downsample=downsample, stride=stride, groups=self.groups,
                            width_per_group=self.width_per_group))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel, channel, groups=self.groups, width_per_group=self.width_per_group))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # 添加注意力机制
        x = self.ca(x) * x
        x = self.sa(x) * x

        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # 添加注意力机制
        x = self.ca1(x) * x
        x = self.sa1(x) * x

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x


def resnet34(pretrained, num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet34-333f7ec4.pth
    model = ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)
    if pretrained:
        state_dict = load_state_dict_from_url("https://download.pytorch.org/models/resnet34-333f7ec4.pth",
                                              model_dir="./model_data")
        model.load_state_dict(state_dict)
    return model


def resnet50(pretrained=False, num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet50-19c8e357.pth
    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)
    if pretrained:
        state_dict = load_state_dict_from_url("https://download.pytorch.org/models/resnet50-19c8e357.pth",
                                              model_dir="./model_data")
        # 加入注意力机制后，使用预训练权值报错，
        # load_state_dict方法还有一个重要的参数是strict，
        # 该参数默认是True，表示预训练模型的层和自己定义的网络结构层严格对应相等（比如层名和维度）。
        model.load_state_dict(state_dict, strict=False)

    return model


def resnet101(pretrained, num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet101-5d3b4d8f.pth
    model = ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)
    if pretrained:
        state_dict = load_state_dict_from_url("https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
                                              model_dir="./model_data")
        model.load_state_dict(state_dict)
    return model


def resnext50_32x4d(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth
    groups = 32
    width_per_group = 4
    return ResNet(Bottleneck, [3, 4, 6, 3],
                  num_classes=num_classes,
                  include_top=include_top,
                  groups=groups,
                  width_per_group=width_per_group)


def resnext101_32x8d(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth
    groups = 32
    width_per_group = 8
    return ResNet(Bottleneck, [3, 4, 23, 3],
                  num_classes=num_classes,
                  include_top=include_top,
                  groups=groups,
                  width_per_group=width_per_group)


if __name__ == '__main__':
    net = resnet101(False)
    # del net.avgpool
    summary(net, (3, 105, 105))
    x = torch.rand(1, 3, 105, 105)
    out = net(x)
    print(out.shape)