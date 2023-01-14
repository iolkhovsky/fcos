import torch

from fcos.backbone import *


def test_build_backbone():
    BATCH_SIZE = 2
    INPUT_CHANNELS = 3
    IMAGENET_CLASSES = 1000
    for backbone_type in BACKBONES.keys():
        model = build_backbone(backbone_type)
        for test_shape in [
            [64, 64],
            [256, 320],
        ]:
            test_input = torch.rand(
                [BATCH_SIZE, INPUT_CHANNELS] + test_shape,
                dtype=torch.float
            )
            out = model(test_input)
            out_shape = list(out.size())
            assert out_shape == [BATCH_SIZE, IMAGENET_CLASSES]
