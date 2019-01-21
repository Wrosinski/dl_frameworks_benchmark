import numpy as np
import torch
import torchvision
from torch import nn
from torch.nn import functional as F

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

mean_ = np.mean(mean)
std_ = np.mean(std)


class DenseNet161(nn.Module):

    def __init__(self, parameters):

        super().__init__()
        self.pretrained = parameters['pretrained']
        self.num_classes = parameters['num_classes']
        self.num_channels = parameters['num_channels']
        self.pooling_output_dim = parameters['pooling_output_dim']
        self.output_features = 2208 * (parameters['pooling_output_dim'] ** 2)
        self.debug = False

        self.densenet = torchvision.models.densenet161(
            pretrained=self.pretrained)

        self.avgpool = nn.AdaptiveMaxPool2d((
            self.pooling_output_dim, self.pooling_output_dim))
        self.classifier = nn.Linear(
            in_features=self.output_features,
            out_features=self.num_classes,
            bias=True)

    def forward(self, x):

        if self.debug:
            print('input: {}'.format(x.size()))

        # x[:, 0, :, :] = (x[:, 0, :, :] - mean[0]) / std[0]
        # x[:, 1, :, :] = (x[:, 1, :, :] - mean[1]) / std[1]
        # x[:, 2, :, :] = (x[:, 2, :, :] - mean[2]) / std[2]

        features = self.densenet.features(x)
        if self.debug:
            print('features: {}'.format(features.size()))

        out = F.relu(features, inplace=True)
        out = self.avgpool(out).view(out.size(0), -1)
        out = self.classifier(out)

        return out
