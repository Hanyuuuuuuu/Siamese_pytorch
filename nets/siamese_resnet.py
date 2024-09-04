import torch
import torch.nn as nn

from nets.resnet import resnet34, resnet50, resnet101, resnext50_32x4d, resnext101_32x8d
# from nets.resnet import ResNet

class Siamese(nn.Module):
    def __init__(self, input_shape, pretrained=False, num_classes = 1000):
        super(Siamese, self).__init__()
        self.resnet = resnet50(pretrained, include_top=True)
        del self.resnet.avgpool
        del self.resnet.fc

        # flat_shape = 512 * get_img_output_length(input_shape[1], input_shape[0])
        flat_shape = 2048 * 4 * 4
        self.fully_connect1 = torch.nn.Linear(flat_shape, 512)
        self.fully_connect2 = torch.nn.Linear(512, 1)

    def forward(self, x):
        x1, x2 = x
        # ------------------------------------------#
        #   我们将两个输入传入到主干特征提取网络
        # ------------------------------------------#
        x1 = self.resnet.conv1(x1)
        x1 = self.resnet.bn1(x1)
        x1 = self.resnet.relu(x1)
        x1 = self.resnet.maxpool(x1)

        x1 = self.resnet.layer1(x1)
        x1 = self.resnet.layer2(x1)
        x1 = self.resnet.layer3(x1)
        x1 = self.resnet.layer4(x1)

        x2 = self.resnet.conv1(x2)
        x2 = self.resnet.bn1(x2)
        x2 = self.resnet.relu(x2)
        x2 = self.resnet.maxpool(x2)

        x2 = self.resnet.layer1(x2)
        x2 = self.resnet.layer2(x2)
        x2 = self.resnet.layer3(x2)
        x2 = self.resnet.layer4(x2)
        # -------------------------#
        #   相减取绝对值
        # -------------------------#
        x1 = torch.flatten(x1, 1)
        x2 = torch.flatten(x2, 1)
        x = torch.abs(x1 - x2)
        # -------------------------#
        #   进行两次全连接
        # -------------------------#
        x = self.fully_connect1(x)
        x = self.fully_connect2(x)
        return x