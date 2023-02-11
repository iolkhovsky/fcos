import torchvision
import torch.nn as nn

from fcos.layers import Conv2dBN


class FeatureExtractor(nn.Module):
    def __init__(self, backbone, out_channels=256):
        super(FeatureExtractor, self).__init__()
        self._backbone = backbone

        self.layer1 = nn.Sequential(*list(self._backbone.children())[:5])
        self.layer2 = nn.Sequential(*list(self._backbone.children())[5])
        self.layer3 = nn.Sequential(*list(self._backbone.children())[6])
        self.layer4 = nn.Sequential(*list(self._backbone.children())[7])

        self._fpn = torchvision.ops.FeaturePyramidNetwork(
            in_channels_list=[512, 1024, 2048],
            out_channels=out_channels,
        )

        self._conv1 = Conv2dBN(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel=3,
            stride=2,
            padding=1,
            batch_norm=False,
            activation='ReLU',
            activation_pars={},
        )
        self._conv2 = Conv2dBN(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel=3,
            stride=2,
            padding=1,
            batch_norm=False,
            activation='ReLU',
            activation_pars={},
        )

    def forward(self, x):
        base_features = self.layer1(x)
        p3_features = self.layer2(base_features)
        p4_features = self.layer3(p3_features)
        p5_features = self.layer4(p4_features)

        feature_maps = self._fpn({
            'P3': p3_features,
            'P4': p4_features,
            'P5': p5_features,
        })
        feature_maps['P6'] = self._conv1(feature_maps['P5'])
        feature_maps['P7'] = self._conv1(feature_maps['P6'])
        return feature_maps

    @property
    def feature_maps(self):
        return ['P3', 'P4', 'P5', 'P6', 'P7']
