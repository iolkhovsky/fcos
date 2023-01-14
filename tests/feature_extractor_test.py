import torch

from fcos.backbone import build_backbone
from fcos.feature_extractor import *


def test_feature_extractor():
    backbone = 'resnet50'
    input_shape = [2, 3, 256, 256]
    out_channels = 64
    h, w = input_shape[2:]
    targets_shapes = {
        'P3': [h // 8, w // 8],
        'P4': [h // 16, w // 16],
        'P5': [h // 32, w // 32],
        'P6': [h // 64, w // 64],
        'P7': [h // 128, w // 128],
    }

    model = FeatureExtractor(build_backbone(backbone), out_channels=out_channels)
    test_input = torch.rand(input_shape, dtype=torch.float)
    out = model(test_input)
  
    assert targets_shapes.keys() == out.keys()
    for k, v in out.items(): 
        pred_shape = list(v.shape)
        assert pred_shape[-2:] == targets_shapes[k]
        assert pred_shape[:2] == [input_shape[0], out_channels]
