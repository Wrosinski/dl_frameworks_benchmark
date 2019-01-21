import numpy as np
import pretrainedmodels
import torch
import torchvision
from torch import nn
from torch.nn import functional as F

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


class SENet(nn.Module):

    def __init__(self, parameters):

        super().__init__()
        self.pretrained = parameters['pretrained']
        self.model_name = parameters['model_name']
        self.num_classes = parameters['num_classes']
        self.num_channels = parameters['num_channels']
        self.pooling_output_dim = parameters['pooling_output_dim']
        self.output_features = parameters['pooling_output_dim'] ** 2
        self.debug = False
        self.dropout = True
        self.dropout_p = 0.2

        assert self.model_name in [
            'se_resnet50', 'se_resnet101', 'se_resnet152', 'senet154',
            'se_resnext50_32x4d', 'se_resnext101_32x4d'], "\
        ResNet type must be one of 'se_resnet50', 'se_resnet101'\
        'se_resnet152', 'senet154', 'se_resnext50_32x4d',\
        'se_resnext101_32x4d'."

        self.senet = pretrainedmodels.__dict__[
            self.model_name](num_classes=1000, pretrained='imagenet')

        self.senet.avg_pool = nn.AdaptiveMaxPool2d((
            self.pooling_output_dim, self.pooling_output_dim))
        self.senet.last_linear = nn.Linear(
            in_features=self.output_features * 2048,
            out_features=self.num_classes,
            bias=True)

    def forward(self, x):

        x[:, 0, :, :] = (x[:, 0, :, :] - mean[0]) / std[0]
        x[:, 1, :, :] = (x[:, 1, :, :] - mean[1]) / std[1]
        x[:, 2, :, :] = (x[:, 2, :, :] - mean[2]) / std[2]

        if self.debug:
            print('input: {}'.format(x.size()))

        out = self.senet(x)
        if self.debug:
            print('out: {}'.format(out.size()))

        return out
