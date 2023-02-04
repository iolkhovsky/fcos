import torch

from fcos.backbone import build_backbone
from fcos.core import *
from fcos.model import *
from dataset.voc_labels import VocLabelsCodec


def test_fcos_core():
    classes = 4
    model = FcosCore(
        backbone=build_backbone('resnet50'),
        classes=classes,
    )

    batch_size = 4
    input_tensor = torch.rand((4, 3, 128, 256), dtype=torch.float32)
    out = model(input_tensor)

    target_detections = sum([y * x for y, x in [
        [16, 32],
        [8, 16],
        [4, 8],
        [2, 4],
        [1, 2],
    ]])

    assert list(out['classes'].shape) == [batch_size, target_detections, classes,]
    assert list(out['centerness'].shape) == [batch_size, target_detections, 1,]
    assert list(out['boxes'].shape) == [batch_size, target_detections, 4,]


def test_fcos():
    target_resolution = (256, 256)
    labels_codec = VocLabelsCodec()

    model = FCOS(
        backbone=build_backbone('resnet50'),
        labels_codec=labels_codec,
        res=target_resolution,
    ).eval()

    batch_size = 3
    test_input = torch.randint(0, 255, size=(batch_size, 3, 256, 384), dtype=torch.uint8)
    outs = model(test_input)
    out_detections = len(model._postprocessor._centers)

    assert len(outs['classes']) == len(outs['scores']) == len(outs['boxes']) == batch_size

    for img_idx in range(batch_size):
        classes_shape = list(outs['classes'][img_idx].shape)
        scores_shape = list(outs['scores'][img_idx].shape)
        boxes_shape = list(outs['boxes'][img_idx].shape)

        assert classes_shape == [out_detections]
        assert scores_shape == [out_detections]
        assert boxes_shape == [out_detections, 4]
