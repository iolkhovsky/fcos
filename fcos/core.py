import torch
from torch import nn

from fcos.feature_extractor import FeatureExtractor
from fcos.heads import RegressionHead, ClassificationHead


class FcosCore(nn.Module):
    def __init__(self, backbone, classes, fext_channels=256):
        super(FcosCore, self).__init__()
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

    def forward(self, x):
        feature_maps = self._fext(x)

        classes, centerness, boxes = [], [], []
        for fmap_name in self._fext.feature_maps:
            feature_map = feature_maps[fmap_name]
            class_tensor, centerness_tensor = self._clf_heads[fmap_name](feature_map)
            b, _, h, w = class_tensor.shape
            classes.append(
                torch.permute(
                    torch.reshape(class_tensor, [b, -1, h * w]),
                    [0, 2, 1]
                )
            )
            centerness.append(
                torch.permute(
                    torch.reshape(centerness_tensor, [b, -1, h * w]),
                    [0, 2, 1]
                )
            )
            boxes_tensor = self._regression_heads[fmap_name](feature_map)
            boxes.append(
                torch.permute(
                    torch.reshape(boxes_tensor, [b, -1, h * w]),
                    [0, 2, 1]
                )
            )

        classes = torch.cat(classes, axis=1)
        centerness = torch.cat(centerness, axis=1)
        boxes = torch.cat(boxes, axis=1)

        return {
            'classes': classes,
            'centerness': centerness,
            'boxes': boxes,
        }
