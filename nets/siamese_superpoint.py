import torch
import torch.nn as nn
import torchvision.transforms as transforms

from nets.superpoint import SuperPointFrontend, SuperPointNet


def get_img_output_length(width, height):
    def get_output_length(input_length):
        # input_length += 6
        filter_sizes = [2, 2, 2, 2, 2]
        padding = [0, 0, 0, 0, 0]
        stride = 2
        for i in range(5):
            input_length = (input_length + 2 * padding[i] - filter_sizes[i]) // stride + 1
        return input_length

    return get_output_length(width) * get_output_length(height)


class Siamese(nn.Module):
    def __init__(self, input_shape, pretrained=False):
        super(Siamese, self).__init__()
        self.superpoint = SuperPointNet()
        # del self.superpoint.pool
        # del self.superpoint.relu
        c1, c2, c3, c4, c5, d1 = 64, 64, 128, 128, 256, 256
        # Shared Encoder.
        # 共享编码器。
        self.conv1a = torch.nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
        self.conv1b = torch.nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
        self.conv2a = torch.nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.conv2b = torch.nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
        self.conv3a = torch.nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.conv3b = torch.nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
        self.conv4a = torch.nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        self.conv4b = torch.nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)

        # Detector Head.
        # 探测头。
        self.convPa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convPb = torch.nn.Conv2d(c5, 65, kernel_size=1, stride=1, padding=0)
        # Descriptor Head.
        # 描述符头。
        self.convDa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convDb = torch.nn.Conv2d(c5, d1, kernel_size=1, stride=1, padding=0)


        # flat_shape = 512 * get_img_output_length(input_shape[1], input_shape[0])
        # self.fully_connect1 = torch.nn.Linear(flat_shape, 512)
        self.fully_connect1 = torch.nn.Linear(21632, 512) # nn.Linear（a,b）函数的意思是，FC层的输入神经元为a,输出神经元为b.
        self.fully_connect2 = torch.nn.Linear(512, 1)

    def forward(self, x):
        x1, x2 = x

        # 假设x1是一个3通道的图像Tensor，可以使用以下代码将其转换为灰度图像
        transform = transforms.Grayscale(num_output_channels=1)
        x1 = transform(x1)
        x2 = transform(x2)

        # ------------------------------------------#
        #   我们将两个输入传入到主干特征提取网络
        # ------------------------------------------#
        x1 = self.superpoint.relu(self.conv1a(x1))
        x1 = self.superpoint.relu(self.conv1b(x1))
        x1 = self.superpoint.pool(x1)
        x1 = self.superpoint.relu(self.conv2a(x1))
        x1= self.superpoint.relu(self.conv2b(x1))
        x1 = self.superpoint.pool(x1)
        x1 = self.superpoint.relu(self.conv3a(x1))
        x1 = self.superpoint.relu(self.conv3b(x1))
        x1 = self.superpoint.pool(x1)
        x1 = self.superpoint.relu(self.conv4a(x1))
        x1 = self.superpoint.relu(self.conv4b(x1))

        x2 = self.superpoint.relu(self.conv1a(x2))
        x2 = self.superpoint.relu(self.conv1b(x2))
        x2 = self.superpoint.pool(x2)
        x2 = self.superpoint.relu(self.conv2a(x2))
        x2 = self.superpoint.relu(self.conv2b(x2))
        x2 = self.superpoint.pool(x2)
        x2 = self.superpoint.relu(self.conv3a(x2))
        x2 = self.superpoint.relu(self.conv3b(x2))
        x2 = self.superpoint.pool(x2)
        x2 = self.superpoint.relu(self.conv4a(x2))
        x2 = self.superpoint.relu(self.conv4b(x2))

        # -------------------------#
        #   相减取绝对值
        # -------------------------#
        '''这个公式属于计算两个向量之间的L1范数距离。
        L1范数距离也称为曼哈顿距离，它衡量了两个向量之间对应元素差的绝对值的和。
        在这个公式中，x1和x2分别是两个向量，
        torch.flatten用于将输入的多维张量扁平化为一维张量，
        然后计算它们的绝对值差，最后得到它们的L1范数距离
       '''
        x1 = torch.flatten(x1, 1)
        x2 = torch.flatten(x2, 1)
        x = torch.abs(x1 - x2)
        # -------------------------#
        #   进行两次全连接
        # -------------------------#
        x = self.fully_connect1(x)  # 全连接层的公式： Y = W * X + B
        x = self.fully_connect2(x)
        return x