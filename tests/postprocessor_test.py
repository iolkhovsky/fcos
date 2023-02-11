import pytest
import torch

from dataset import VocLabelsCodec
from fcos.postprocessor import FcosPostprocessor
from fcos.encoder import FcosDetectionsEncoder


def test_postprocessor():
    img_ysz, img_xsz = 256, 256
    postprocessor = FcosPostprocessor((img_ysz, img_xsz))

    scales = [(1, 1), (2, 2)]
    classes, batch_size = 4, len(scales)
    detections = len(postprocessor._centers)

    cls_inputs = torch.rand([batch_size, detections, classes])
    cntr_inputs = torch.rand([batch_size, detections, 1])
    rgr_inputs = torch.rand([batch_size, detections, 4])
    test_inputs = {
        'classes': cls_inputs,
        'centerness': cntr_inputs,
        'boxes': rgr_inputs,
    }

    outs = postprocessor(test_inputs, scales)

    labels, scores, boxes = outs['classes'], outs['scores'], outs['boxes']

    assert len(labels) == len(scores) == len(boxes) == batch_size
    assert all([list(x.shape) == [detections,] for x in labels])
    assert all([list(x.shape) == [detections,] for x in scores])
    assert all([list(x.shape) == [detections, 4,] for x in boxes])


def test_codec_centers():
    level_scales = [2**i for i in range(3, 4 + 1)]
    height, width = 256, 256

    centers = FcosPostprocessor.generate_centers((height, width), level_scales)
    target_positions = sum(
        [(height // scale) * (width // scale) for scale in level_scales]
    )
    assert list(centers.shape) == [target_positions, 2]

    map_offset = 0
    for scale in level_scales:
        fmap_ysz = height // scale
        fmap_xsz = width // scale
        for y in range(fmap_ysz):
            for x in range(fmap_xsz):
                plane_index = map_offset + y * fmap_xsz + x
                assert centers[plane_index] == \
                    pytest.approx(torch.tensor([(y + 0.5) * scale, (x + 0.5) * scale]))
        map_offset += fmap_ysz * fmap_xsz


def test_decode_predictions():
    img_res = (256, 256)
    encoder = FcosDetectionsEncoder(img_res, VocLabelsCodec())
    img_scales = [(1., 1.), (1., 1.), (1., 1.)]

    test_boxes = [
        torch.tensor([
            [50, 60, 110, 210],
            [30, 70, 40, 80],
            [100, 200, 110, 210],
        ]),
        torch.zeros([0, 4]),
        torch.tensor([
            [0, 0, 256, 256],
            [180, 200, 200, 230],
        ]),
    ]

    test_labels = [
        torch.tensor([0, 7, 10]),
        torch.zeros([0, 1]),
        torch.tensor([2, 3]),
    ]
    batch_size = test_boxes

    encoded_gt = encoder(test_boxes, test_labels)
    encoded_gt = {k: torch.tensor(v) for k, v in encoded_gt.items()}
    postprocessor = FcosPostprocessor(img_res)
    decoded = postprocessor(encoded_gt, scales=img_scales)

    pred_boxes = decoded['boxes']
    pred_scores = decoded['scores']
    pred_labels = decoded['classes']

    for img_idx in range(len(batch_size)):
        img_gt_boxes = test_boxes[img_idx]
        img_gt_labels = test_labels[img_idx]

        pred_boxes = decoded['boxes'][img_idx]
        pred_labels = decoded['classes'][img_idx]
        pred_scores = decoded['scores'][img_idx]

        for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
            if score > 0.25:               
                ok = False
                for gt_box, gt_label in zip(img_gt_boxes, img_gt_labels):
                    if torch.all(torch.eq(gt_box, box)):
                        ok = gt_label == label
                assert ok
