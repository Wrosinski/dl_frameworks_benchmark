import pretrainedmodels
import torch
import torchvision
from torch import nn
from torch.nn import functional as F

mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]


class InceptionResNetV2(nn.Module):

    def __init__(self, parameters):

        super().__init__()
        self.pretrained = parameters['pretrained']
        self.num_classes = parameters['num_classes']
        self.num_channels = parameters['num_channels']
        self.pooling_output_dim = parameters['pooling_output_dim']
        self.output_features = 1536 * (parameters['pooling_output_dim'] ** 2)
        self.debug = False

        if self.pretrained:
            self.inception = pretrainedmodels.__dict__['inceptionresnetv2'](
                num_classes=1000, pretrained='imagenet')
        else:
            self.inception = pretrainedmodels.__dict__['inceptionresnetv2'](
                num_classes=1000, pretrained=None)

        self.inception.avgpool_1a = nn.AdaptiveMaxPool2d((
            self.pooling_output_dim, self.pooling_output_dim))
        self.inception.last_linear = nn.Linear(
            in_features=self.output_features,
            out_features=self.num_classes,
            bias=True)

    def forward(self, x):

        if self.debug:
            print('input: {}'.format(x.size()))

        # x[:, 0, :, :] = (x[:, 0, :, :] - mean[0]) / std[0]
        # x[:, 1, :, :] = (x[:, 1, :, :] - mean[1]) / std[1]
        # x[:, 2, :, :] = (x[:, 2, :, :] - mean[2]) / std[2]

        out = self.inception(x)
        if self.debug:
            print('output: {}'.format(out.size()))

        return out
