import torch

from fcos.backbone import build_backbone
from fcos.core import *
from fcos.model import *
from dataset.labels_codec import LabelsCodec


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


def test_fcos():
    batch_size = 3
    target_resolution = (512, 512)
    level_scales = {f'P{idx}': 2 ** idx for idx in range(3, 7 + 1)}
    labels_codec = LabelsCodec()

    model = FCOS(
        backbone=build_backbone('resnet50'),
        labels_codec=labels_codec,
        res=target_resolution,
    )

    test_input = torch.randint(0, 255, size=(batch_size, 3, 256, 384), dtype=torch.uint8)
    outs = model(test_input)

    for level, predictions in outs.items():
        classes_shape = list(predictions['classes'].shape)
        centerness_shape = list(predictions['centerness'].shape)
        boxes_shape = list(predictions['boxes'].shape)

        w, h = target_resolution
        w = w // level_scales[level]
        h = h // level_scales[level]
        assert classes_shape == [batch_size, w * h, len(labels_codec)]
        assert centerness_shape == [batch_size, w * h, 1]
        assert boxes_shape == [batch_size, w * h, 4]
