import numpy as np
import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torchvision.models.resnet import BasicBlock, ResNet

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


class ResNet50(nn.Module):

    def __init__(self, parameters):

        super().__init__()
        self.pretrained = parameters['pretrained']
        self.num_classes = parameters['num_classes']
        self.num_channels = parameters['num_channels']
        self.pooling_output_dim = parameters['pooling_output_dim']
        self.output_features = parameters['pooling_output_dim'] ** 2
        self.debug = False

        self.resnet = torchvision.models.resnet50(
            pretrained=self.pretrained)
        self.resnet.avgpool = nn.AdaptiveMaxPool2d((
            self.pooling_output_dim, self.pooling_output_dim))

        self.resnet.fc = nn.Linear(
            in_features=self.output_features * 2048,
            out_features=self.num_classes,
            bias=True)

    def forward(self, x):

        # x[:, 0, :, :] = (x[:, 0, :, :] - mean[0]) / std[0]
        # x[:, 1, :, :] = (x[:, 1, :, :] - mean[1]) / std[1]
        # x[:, 2, :, :] = (x[:, 2, :, :] - mean[2]) / std[2]

        if self.debug:
            print('input: {}'.format(x.size()))

        out = self.resnet(x)

        return out
