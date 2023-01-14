from torch import nn

from fcos.layers import Conv2dBN


class RegressionHead(nn.Module):
    def __init__(self, layers=4, channels=256):
        super().__init__()
        self.layers = [
            Conv2dBN(
                in_channels=channels,
                out_channels=channels
            )
            for _ in range(layers)
        ]
        self.layers.append(
            Conv2dBN(
                in_channels=channels,
                out_channels=4,
                activation='ReLU',
                activation_pars={},
            )
        )
        self.layers = nn.ModuleList(self.layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ClassificationHead(nn.Module):
    def __init__(self, classes, layers=4, channels=256):
        super().__init__()
        self.neck = [
            Conv2dBN(
                in_channels=channels,
                out_channels=channels,
            )
            for _ in range(layers)
        ]
        self.logits_head = Conv2dBN(
            in_channels=channels,
            out_channels=classes,
            activation='Sigmoid',
            activation_pars={},
            bias_pi=0.01,
        )
        self.centerness_head = Conv2dBN(
            in_channels=channels,
            out_channels=1,
            activation='Sigmoid',
            activation_pars={},
            bias_pi=0.01,
        )        

    def forward(self, x):
        for layer in self.neck:
            x = layer(x)
        return self.logits_head(x), self.centerness_head(x)
