import torch
from torch import nn

from fcos.feature_extractor import FeatureExtractor
from fcos.heads import RegressionHead, ClassificationHead


class FcosCore(nn.Module):
    def __init__(self, backbone, classes, fext_channels=256):
        super().__init__()
        self._fext = FeatureExtractor(backbone, out_channels=fext_channels)
        init_scales = {f'P{idx}': 2 ** idx for idx in range(3, 7 + 1)}
        self._regression_heads = nn.ModuleDict({
            name: RegressionHead(in_channels=fext_channels, scale=init_scales[name])
            for name in self._fext.feature_maps
        })
        self._clf_heads = nn.ModuleDict({
            name: ClassificationHead(classes, in_channels=fext_channels)
            for name in self._fext.feature_maps
        })

    @staticmethod
    def bchw_to_bnc(tensor):
        b, c, h, w = tensor.shape
        return torch.permute(
            torch.reshape(tensor, [b, c, h * w]),
            [0, 2, 1]
        )

    def _concatenate_outputs(self, fmap_predictions):
        return {
            'classes': FcosCore.bchw_to_bnc(
                torch.cat(
                    [
                        fmap_predictions[fmap_name][0][0]
                        for fmap_name in self._fext.feature_maps
                    ],
                    axis=1,
                )
            ),
            'centerness': FcosCore.bchw_to_bnc(
                torch.cat(
                    [
                        fmap_predictions[fmap_name][0][1]
                        for fmap_name in self._fext.feature_maps
                    ],
                    axis=1,
                )
            ),
            'boxes': FcosCore.bchw_to_bnc(
                torch.cat(
                    [
                        fmap_predictions[fmap_name][1]
                        for fmap_name in self._fext.feature_maps
                    ],
                    axis=1,
                )
            ),
        }

    def forward(self, x):
        return self._concatenate_outputs({
            k: (self._clf_heads[k](fmap), self._regression_heads[k](fmap))
            for k, fmap in self._fext(x).items()
        })
