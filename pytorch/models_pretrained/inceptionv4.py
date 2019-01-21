import numpy as np
import pretrainedmodels
import torch
import torchvision
from torch import nn
from torch.nn import functional as F

mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]


class InceptionV4(nn.Module):

    def __init__(self, parameters):

        super().__init__()
        self.pretrained = parameters['pretrained']
        self.num_classes = parameters['num_classes']
        self.num_channels = parameters['num_channels']
        self.pooling_output_dim = parameters['pooling_output_dim']
        self.output_features = 1536 * (parameters['pooling_output_dim'] ** 2)
        self.debug = False

        if self.pretrained:
            self.inception = pretrainedmodels.__dict__['inceptionv4'](
                num_classes=1000, pretrained='imagenet')
        else:
            self.inception = pretrainedmodels.__dict__['inceptionv4'](
                num_classes=1000, pretrained=None)

        self.avg_pool = nn.AdaptiveMaxPool2d((
            self.pooling_output_dim, self.pooling_output_dim))
        self.last_linear = nn.Linear(
            self.output_features, self.num_classes)

    def forward(self, x):

        if self.debug:
            print('input: {}'.format(x.size()))

        # x[:, 0, :, :] = (x[:, 0, :, :] - mean[0]) / std[0]
        # x[:, 1, :, :] = (x[:, 1, :, :] - mean[1]) / std[1]
        # x[:, 2, :, :] = (x[:, 2, :, :] - mean[2]) / std[2]

        x = self.inception.features(x)
        if self.debug:
            print('features: {}'.format(features.size()))

        out = self.avg_pool(x)
        out = out.view(out.size(0), -1)
        out = self.last_linear(out)

        if self.debug:
            print('output', out.size())

        return out
