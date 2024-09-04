import torch
import torch.nn as nn
# from torchvision.models.utils import load_state_dict_from_url
from torch.hub import load_state_dict_from_url


# 定义自注意力层
class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        query = self.query_conv(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        key = self.key_conv(x).view(batch_size, -1, height * width)
        energy = torch.bmm(query, key)
        attention = torch.softmax(energy, dim=-1)
        value = self.value_conv(x).view(batch_size, -1, height * width)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)
        out = self.gamma * out + x
        return out

# 定义VggNet网络模型
class VGG(nn.Module):
    # init()：进行初始化，申明模型中各层的定义
    # features：make_features(cfg: list)生成提取特征的网络结构
    # num_classes：需要分类的类别个数
    # init_weights：是否对网络进行权重初始化
    def __init__(self, features, num_classes=1000, initialize_weights=False):
        # super：引入父类的初始化方法给子类进行初始化
        super(VGG, self).__init__()
        # 生成提取特征的网络结构
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        # 生成分类的网络结构
        # Sequential：自定义顺序连接成模型，生成网络结构
        self.classifier = nn.Sequential(
            # Dropout：随机地将输入中50%的神经元激活设为0，即去掉了一些神经节点，防止过拟合
            nn.Linear(512 * 7 * 7, 4096),
            # ReLU(inplace=True)：将tensor直接修改，不找变量做中间的传递，节省运算内存，不用多存储额外的变量
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        # 如果为真，则对网络参数进行初始化
        if initialize_weights:
            self._initialize_weights()

    # forward()：定义前向传播过程,描述了各层之间的连接关系
    def forward(self, x):
        # 将数据输入至提取特征的网络结构，N x 3 x 224 x 224
        x = self.features(x)
        x = self.avgpool(x)
        # N x 512 x 7 x 7
        # 图像经过提取特征网络结构之后，得到一个7*7*512的特征矩阵，进行展平
        # Flatten()：将张量（多维数组）平坦化处理，神经网络中第0维表示的是batch_size，所以Flatten()默认从第二维开始平坦化
        x = torch.flatten(x, start_dim=1)
        # 将数据输入分类网络结构，N x 512*7*7
        x = self.classifier(x)
        return x

    # 网络结构参数初始化
    def _initialize_weights(self):
        # 遍历网络中的每一层
        # 继承nn.Module类中的一个方法:self.modules(), 他会返回该网络中的所有modules
        for m in self.modules():
            # isinstance(object, type)：如果指定对象是指定类型，则isinstance()函数返回True
            # 如果是卷积层
            if isinstance(m, nn.Conv2d):
                # uniform_(tensor, a=0, b=1)：服从~U(a,b)均匀分布，进行初始化
                # nn.init.xavier_uniform_(m.weight)
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                # 如果偏置不是0，将偏置置成0，对偏置进行初始化
                if m.bias is not None:
                    # constant_(tensor, val)：初始化整个矩阵为常数val
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                # 如果是全连接层
            elif isinstance(m, nn.Linear):
                # 正态分布初始化
                # nn.init.xavier_uniform_(m.weight)
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

# 105, 105, 3   -> 105, 105, 64 -> 52, 52, 64
# 52, 52, 64    -> 52, 52, 128  -> 26, 26, 128
# 26, 26, 128   -> 26, 26, 256  -> 13, 13, 256
# 13, 13, 256   -> 13, 13, 512  -> 6, 6, 512
# 6, 6, 512     -> 6, 6, 512    -> 3, 3, 512
# 生成提取特征的网络结构
# 参数是网络配置变量，传入对应配置的列表（list类型）
def make_layers(cfg, batch_norm=False, in_channels = 3):
    # 定义空列表，存放创建的每一层结构
    layers = []
    # for循环遍历配置列表，得到由卷积层和池化层组成的一个列表
    for v in cfg:
        # 如果列表的值是M字符，说明该层是最大池化层
        if v == 'M':
            # 创建一个最大池化层，在VGG中所有的最大池化层的kernel_size=2，stride=2
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        # 否则是卷积层
        else:
            # in_channels：输入的特征矩阵的深度，v：输出的特征矩阵的深度，深度也就是卷积核的个数
            # 在Vgg中，所有的卷积层的padding=1，stride=1
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    # 将列表通过非关键字参数的形式返回，*layers可以接收任意数量的参数
    return nn.Sequential(*layers)

# 定义cfgs字典文件，每一个key代表一个模型的配置文件，如：VGG11代表A配置，也就是一个11层的网络
# 数字代表卷积层中卷积核的个数，'M'代表池化层的结构
# 通过函数make_features(cfg: list)生成提取特征网络结构
cfgs = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    # 'vgg16': [64, SelfAttention(64), 64, 'M', 128, SelfAttention(128), 128, 'M', 256, 256, 256, 'M', 512, 512, 512,
    # 'M', 512, 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def VGG16(pretrained, in_channels, **kwargs):
    # 实例化VGG网络
    # 这个字典变量包含了分类的个数以及是否初始化权重的布尔变量
    model = VGG(make_layers(cfgs["vgg16"], batch_norm=False, in_channels=in_channels), **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url("https://download.pytorch.org/models/vgg16-397923af.pth",
                                           model_dir="./model_data")
        model.load_state_dict(state_dict)
    return model

def VGG19(pretrained, in_channels, **kwargs):
    # 实例化VGG网络
    # 这个字典变量包含了分类的个数以及是否初始化权重的布尔变量
    model = VGG(make_layers(cfgs["vgg19"], batch_norm = False, in_channels = in_channels), **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url("https://download.pytorch.org/models/", model_dir="./model_data")
        model.load_state_dict(state_dict)
    return model
