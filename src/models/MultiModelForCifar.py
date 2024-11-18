import torch
import torch.nn as nn

from src.models.CNNForCifar import *
from src.models.ClassfierModel import *


class MultiModelForCifar(nn.Module):
    def __init__(self, device):
        super(MultiModelForCifar, self).__init__()
        self.device = device

        self.classifier = ClassifierModel()
        self.color_model = CNNForCifar(True, 64)
        self.gray_model = CNNForCifar(False, 64)
        self.relu = nn.ReLU()

    def forward(self, color=None, gray=None):
        if color is not None:
            color = self.color_model(color)
        else:
            color = torch.zeros(*gray.shape).to(self.device)
        if gray is not None:
            gray = self.gray_model(gray)
        else:
            gray = torch.zeros(*color.shape).to(self.device)

        x = torch.cat((color, gray), dim=1)
        x = self.relu(x)
        x = self.classifier(x)

        return x
