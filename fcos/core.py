from torch import nn

from fcos.feature_extractor import FeatureExtractor
from fcos.heads import RegressionHead, ClassificationHead


class FcosCore(nn.Module):
    def __init__(self, backbone, classes):
        super().__init__()
        self._fext = FeatureExtractor(backbone)
        self._regression_heads = {name: RegressionHead(channels=256) for name in self._fext.feature_maps}
        self._clf_heads = {name: ClassificationHead(classes, channels=256) for name in self._fext.feature_maps}

    def forward(self, x):
        out = {}
        for k, fmap in self._fext(x).items():
            out[k] = (
                self._clf_heads[k](fmap),
                self._regression_heads[k](fmap),
            )
        return out
