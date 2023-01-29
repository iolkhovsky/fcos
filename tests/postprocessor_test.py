import torch

from fcos.postprocessor import *
from fcos.codec import FcosDetectionsCodec


def test_postprocessor():
    codec = FcosDetectionsCodec((256, 256), None)
    postprocessor = FcosPostprocessor(codec)

    scales = [(1, 1), (2, 2)]
    classes, batch_size = 4, len(scales)
    width, height = 3, 2
    cls_inputs = torch.rand([batch_size, classes, height, width])
    cntr_inputs = torch.rand([batch_size, 1, height, width])
    rgr_inputs = torch.rand([batch_size, 4, height, width])
    test_inputs = {
        f"P{i}": ((cls_inputs, cntr_inputs), rgr_inputs) for i in range(3, 7 + 1)
    }

    outs = postprocessor(test_inputs, scales)

    cls, cntr, regr = outs['classes'], outs['centerness'], outs['boxes']

    total_detections = len(test_inputs) * width * height

    assert list(cls.shape) == [batch_size, total_detections, classes]
    assert list(cntr.shape) == [batch_size, total_detections, 1]
    assert list(regr.shape) == [batch_size, total_detections, 4]
