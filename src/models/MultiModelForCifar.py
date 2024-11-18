import torch
import torch.nn as nn

from src.models.CNNForCifar import *
from src.models.ClassfierModel import *


class MultiModelForCifar(nn.Module):
    def __init__(self):
        super(MultiModelForCifar, self).__init__()
        self.classifier = ClassifierModel()
        self.color_model = CNNForCifar(True, 64)
        self.gray_model = CNNForCifar(False, 64)
        self.relu = nn.ReLU()

    def forward(self, color=None, gray=None):
        if color is not None:
            # print(color)
            color = self.color_model(color)
            # print(color)
            color = self.relu(color)
            # print(color)
            color = self.classifier(color)
            # print(color)
        if gray is not None:
            # print(gray)
            gray = self.gray_model(gray)
            # print(gray)
            gray = self.relu(gray)
            # print(gray)
            gray = self.classifier(gray)
            # print(gray)
        if color is not None and gray is not None:
            return (color+gray) / 2
        else:
            return color if color is not None else gray
