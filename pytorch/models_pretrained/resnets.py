import numpy as np
import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torchvision.models.resnet import BasicBlock, ResNet

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


class ResNet(nn.Module):

    def __init__(self, parameters):

        super().__init__()
        self.pretrained = parameters['pretrained']
        self.model_name = parameters['model_name']
        self.num_classes = parameters['num_classes']
        self.num_channels = parameters['num_channels']
        self.pooling_output_dim = parameters['pooling_output_dim']
        self.output_features = parameters['pooling_output_dim'] ** 2
        self.debug = False

        assert self.model_name in [
            'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'], "\
        ResNet type must be one of 'resnet18', 'resnet34', \
        'resnet50', 'resnet101', 'resnet152'"

        if self.model_name == 'resnet18':
            self.resnet = torchvision.models.resnet18(
                pretrained=self.pretrained)
        if self.model_name == 'resnet34':
            self.resnet = torchvision.models.resnet34(
                pretrained=self.pretrained)
        if self.model_name == 'resnet50':
            self.resnet = torchvision.models.resnet50(
                pretrained=self.pretrained)
        if self.model_name == 'resnet101':
            self.resnet = torchvision.models.resnet101(
                pretrained=self.pretrained)
        if self.model_name == 'resnet152':
            self.resnet = torchvision.models.resnet152(
                pretrained=self.pretrained)

        self.resnet.avgpool = nn.AdaptiveMaxPool2d((
            self.pooling_output_dim, self.pooling_output_dim))

        if self.model_name in ['resnet18', 'resnet34']:
            self.resnet.last_linear = nn.Linear(
                in_features=self.output_features * 512,
                out_features=self.num_classes,
                bias=True)
        else:
            self.resnet.last_linear = nn.Linear(
                in_features=self.output_features * 2048,
                out_features=self.num_classes,
                bias=True)

    def forward(self, x):

        x[:, 0, :, :] = (x[:, 0, :, :] - mean[0]) / std[0]
        x[:, 1, :, :] = (x[:, 1, :, :] - mean[1]) / std[1]
        x[:, 2, :, :] = (x[:, 2, :, :] - mean[2]) / std[2]

        if self.debug:
            print('input: {}'.format(x.size()))

        out = self.resnet(x)

        return out
