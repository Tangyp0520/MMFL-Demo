import torch
import torch.nn as nn
import torchvision.models as models


# class ModelNetResNet18(nn.Module):
#     """
#     ModelNetResNet18
#     针对ModelNet40数据集转换成的154*154三通道png图像
#     输出256长度向量
#     """
#     def __init__(self):
#         super(ModelNetResNet18, self).__init__()
#         # 加载预训练的 ResNet18
#         # resnet18 = models.resnet18(weights='ResNet18_Weights.DEFAULT')
#         resnet18 = models.resnet18()
#         # 修改第一层卷积层以适应 154*154 输入
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
#         # 加载 ResNet18 的剩余层
#         self.layer1 = resnet18.layer1
#         self.layer2 = resnet18.layer2
#         self.layer3 = resnet18.layer3
#         self.layer4 = resnet18.layer4
#         # 添加全连接层以输出 256 长度向量
#         self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = nn.Linear(512, 256)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#         x = self.avg_pool(x)
#         x = torch.flatten(x, 1)
#         x = self.fc(x)
#         return x

class Resnet18ForModelNet(nn.Module):
    def __init__(self):
        super(Resnet18ForModelNet, self).__init__()
        # 加载预训练的 ResNet18 模型，并去除最后的全连接层
        self.resnet = models.resnet18(weights='ResNet18_Weights.DEFAULT')
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        # 添加一个自适应平均池化层，将特征图转换为 256 长度的特征向量
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 256)

    def forward(self, x):
        x = self.resnet(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x