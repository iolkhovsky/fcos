import torch

from fcos.backbone import build_backbone
from fcos.core import *


def test_fcos_core():
    classes = 4
    model = FcosCore(
        backbone=build_backbone('resnet50'),
        classes=classes,
    )

    batch_size = 4
    input_tensor = torch.rand((4, 3, 128, 256), dtype=torch.float32)
    out = model(input_tensor)

    target_shapes = {
        'P3': [16, 32],
        'P4': [8, 16],
        'P5': [4, 8],
        'P6': [2, 4],
        'P7': [1, 2],
    }

    for level_label, ((cls, cntr), regr) in out.items():
        cls_shape = list(cls.shape)
        cntr_shape = list(cntr.shape)
        regr_shape = list(regr.shape)
        assert cls_shape == [batch_size, classes,] + target_shapes[level_label]
        assert cntr_shape == [batch_size, 1,] + target_shapes[level_label]
        assert regr_shape == [batch_size, 4,] + target_shapes[level_label]
