import torch
import torch.nn as nn


# class ClassifierModel(nn.Module):
#     """
#     分类器模型 用于多模态训练图像分类
#     接受1*256向量 输出1*10向量
#     """
#     def __init__(self):
#         super(ClassifierModel, self).__init__()
#         self.fc1 = nn.Linear(256*4, 128)
#         self.fc2 = nn.Linear(128, 64)
#         self.fc3 = nn.Linear(64, 10)
#         # self.relu = nn.ReLU()
#         self.relu = nn.PReLU()
#         self.softmax = nn.Softmax(dim=1)
#
#     def forward(self, x):
#         print(x)
#         x = self.relu(self.fc1(x))
#         print(x)
#         x = self.relu(self.fc2(x))
#         x = self.fc3(x)
#         return self.softmax(x)

class ClassifierModel(nn.Module):
    def __init__(self):
        super(ClassifierModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.flatten = nn.Flatten()

        # num_features = 64 * ((256*4 - 4) // 2 - 4) // 2 kernel_size=5
        # num_features = 16256 2层卷积
        num_features = 16352
        self.fc1 = nn.Linear(num_features, 128)
        self.fc2 = nn.Linear(128, 10)
        # self.relu = nn.ReLU()
        self.relu = nn.PReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        print('--------')
        print(x)
        x = x.unsqueeze(1)
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        print(x)
        # x = self.relu(self.conv2(x))
        # x = self.pool2(x)
        x = self.flatten(x)
        print(x)
        x = self.relu(self.fc1(x))
        print(x)
        return self.softmax(self.fc2(x))
