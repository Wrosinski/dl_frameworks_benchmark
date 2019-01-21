import pretrainedmodels
import torch
import torchvision
from torch import nn
from torch.nn import functional as F

mean = [124 / 255, 117 / 255, 104 / 255]
std = [1 / (.0167 * 255)] * 3


class DPN(nn.Module):

    def __init__(self, parameters):

        super().__init__()
        self.pretrained = parameters['pretrained']
        self.model_name = parameters['model_name']
        self.num_classes = parameters['num_classes']
        self.num_channels = parameters['num_channels']
        self.pooling_output_dim = parameters['pooling_output_dim']
        self.output_features = 128
        self.debug = False
        self.dropout = True
        self.dropout_p = 0.2

        assert self.model_name in ['dpn92', 'dpn68b', 'dpn107']

        if self.pretrained:
            self.dpn = pretrainedmodels.__dict__[self.model_name](
                num_classes=1000, pretrained='imagenet+5k')
        else:
            self.dpn = pretrainedmodels.__dict__[self.model_name](
                num_classes=1000, pretrained=None)

        self.dpn.last_linear = nn.Conv2d(
            2688, self.num_classes, kernel_size=(1, 1), stride=(1, 1))

        self.dpn_features = self.dpn.features

        self.avg_pool = nn.AdaptiveMaxPool2d((
            self.pooling_output_dim, self.pooling_output_dim))
        self.last_linear = nn.Conv2d(
            2688, self.num_classes, kernel_size=1, bias=True)

    def logits(self, features):

        x = self.avg_pool(features)
        out = self.last_linear(x)
        # out = out.view(out.size(0), -1)

        return out

    def forward(self, x):

        x[:, 0, :, :] = (x[:, 0, :, :] - mean[0]) / std[0]
        x[:, 1, :, :] = (x[:, 1, :, :] - mean[1]) / std[1]
        x[:, 2, :, :] = (x[:, 2, :, :] - mean[2]) / std[2]

        if self.debug:
            print('input: {}'.format(x.size()))

        x = self.dpn_features(x)
        x = self.logits(x)

        return x

# def forward(self, x):
#
#     x[:, 0, :, :] = (x[:, 0, :, :] - mean[0]) / std[0]
#     x[:, 1, :, :] = (x[:, 1, :, :] - mean[1]) / std[1]
#     x[:, 2, :, :] = (x[:, 2, :, :] - mean[2]) / std[2]
#
#     if self.debug:
#         print('input: {}'.format(x.size()))
#
#     out = self.dpn(x)
#     if self.debug:
#         print('features: {}'.format(out.size()))
#     # out = self.avg_pool(out)
#     # if self.debug:
#     #     print('pool: {}'.format(out.size()))
#     # out = self.last_linear(out)
#
#     if self.debug:
#         print('out', out.size())
#
#     return out
