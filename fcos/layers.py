import numpy as np
import torch
from torch import nn


class Conv2dBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=3, stride=1,
                 padding=1, bias=True, bias_value=None, bias_pi=None,
                 activation='LeakyReLU', activation_pars={'negative_slope': 0.01},
                 batch_norm=True, initializer=None):
        super(Conv2dBN, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(kernel, kernel),
            stride=(stride, stride),
            padding=(padding, padding),
            bias=bias,
        )
        if bias_pi:
            bias_value = -1. * np.log((1. - bias_pi) / bias_pi)
        if bias_value:
            self.conv.bias.data.fill_(bias_value)
        if initializer:
            initializer(self.conv.weight)

        self.act = None
        if activation:
            self.act = getattr(torch.nn, activation)(**activation_pars)
        self.bn = None
        if batch_norm:
            self.bn = torch.nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        if self.act:
            x = self.act(x)
        return x
