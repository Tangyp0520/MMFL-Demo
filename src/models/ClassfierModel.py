import torch
import torch.nn as nn


class ClassifierModel(nn.Module):
    """
    分类器模型 用于多模态训练图像分类
    接受1*256向量 输出1*10向量
    """
    def __init__(self):
        super(ClassifierModel, self).__init__()
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)
