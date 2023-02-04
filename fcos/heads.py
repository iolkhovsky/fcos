from torch import nn
import torch

from fcos.layers import Conv2dBN


class RegressionHead(nn.Module):
    def __init__(self, layers=4, in_channels=256, intermediate_channels=256, scale=1.):
        super().__init__()

        def _initializer(weights):
            nn.init.kaiming_normal_(weights, nonlinearity='relu')
        
        self.neck = [
            Conv2dBN(
                in_channels=in_channels,
                out_channels=intermediate_channels,
                batch_norm=True,
                bias=False,
                activation='ReLU',
                activation_pars={},
                initializer=_initializer,
            )
        ]
        self.neck.extend([
            Conv2dBN(
                in_channels=intermediate_channels,
                out_channels=intermediate_channels,
                batch_norm=True,
                bias=False,
                activation='ReLU',
                activation_pars={},
                initializer=_initializer,
            )
            for _ in range(1, layers)
        ])
        self.neck = nn.ModuleList(self.neck)
        self.regression_head = Conv2dBN(
                in_channels=intermediate_channels,
                out_channels=4,
                activation=None,
                batch_norm=False,
                bias=True,
        )
        self._scale = torch.nn.Parameter(torch.tensor(scale, dtype=torch.float))

    def forward(self, x):
        for layer in self.neck:
            x = layer(x)
        return self._scale * torch.exp(self.regression_head(x))


class ClassificationHead(nn.Module):
    def __init__(self, classes, layers=4, in_channels=256, intermediate_channels=256):
        super().__init__()

        def _initializer(weights):
            nn.init.kaiming_normal_(weights, nonlinearity='relu')
        
        self.neck = [
            Conv2dBN(
                in_channels=in_channels,
                out_channels=intermediate_channels,
                batch_norm=True,
                bias=False,
                activation='ReLU',
                activation_pars={},
                initializer=_initializer,
            )
        ]
        self.neck.extend([
            Conv2dBN(
                in_channels=intermediate_channels,
                out_channels=intermediate_channels,
                batch_norm=True,
                bias=False,
                activation='ReLU',
                activation_pars={},
                initializer=_initializer,
            )
            for _ in range(1, layers)
        ])
        self.neck = nn.ModuleList(self.neck)
        self.logits_head = Conv2dBN(
            in_channels=intermediate_channels,
            out_channels=classes,
            activation=None,
            bias_pi=0.01,
            batch_norm=False,
        )
        self.centerness_head = Conv2dBN(
            in_channels=intermediate_channels,
            out_channels=1,
            activation=None,
            batch_norm=False,
        )        

    def forward(self, x):
        for layer in self.neck:
            x = layer(x)
        return self.logits_head(x), self.centerness_head(x)
