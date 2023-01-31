from torch import nn

from fcos.feature_extractor import FeatureExtractor
from fcos.heads import RegressionHead, ClassificationHead


class FcosCore(nn.Module):
    def __init__(self, backbone, classes, fext_channels=256):
        super().__init__()
        self._fext = FeatureExtractor(backbone, out_channels=fext_channels)
        self._regression_heads = nn.ModuleDict({
            name: RegressionHead(in_channels=fext_channels)
            for name in self._fext.feature_maps
        })
        self._clf_heads = nn.ModuleDict({
            name: ClassificationHead(classes, in_channels=fext_channels)
            for name in self._fext.feature_maps
        })

    def forward(self, x):
        return {
            k: (self._clf_heads[k](fmap), self._regression_heads[k](fmap))
            for k, fmap in self._fext(x).items()
        }
