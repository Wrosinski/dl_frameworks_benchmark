import pretrainedmodels
import torch
import torchvision
from torch import nn
from torch.nn import functional as F

mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]


class NASNet(nn.Module):

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

        if self.pretrained:
            self.nasnet = pretrainedmodels.__dict__['nasnetalarge'](
                num_classes=1000, pretrained='imagenet')
        else:
            self.nasnet = pretrainedmodels.__dict__['nasnetalarge'](
                num_classes=1000, pretrained=None)

        self.nasnet.avg_pool = nn.AdaptiveMaxPool2d((
            self.pooling_output_dim, self.pooling_output_dim))
        self.nasnet.last_linear = nn.Linear(
            in_features=self.output_features * 4032,
            out_features=self.num_classes,
            bias=True)

    def forward(self, x):

        # x[:, 0, :, :] = (x[:, 0, :, :] - mean[0]) / std[0]
        # x[:, 1, :, :] = (x[:, 1, :, :] - mean[1]) / std[1]
        # x[:, 2, :, :] = (x[:, 2, :, :] - mean[2]) / std[2]

        if self.debug:
            print('input: {}'.format(x.size()))

        out = self.nasnet(x)

        if self.debug:
            print('out', out.size())

        return out
