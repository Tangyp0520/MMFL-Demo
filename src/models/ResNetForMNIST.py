import torch
import torch.nn as nn
import torchvision.models as models


class ResNetForMNIST(nn.Module):

    def __init__(self, color):
        super(ResNetForMNIST, self).__init__()
        if color:
            in_channels = 3
        else:
            in_channels = 1
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(16, 16, 2)
        self.layer2 = self._make_layer(16, 32, 2, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(32, 64)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride=1):
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
                  nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)]
        for _ in range(1, num_blocks):
            layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
