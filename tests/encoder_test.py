import torch
import numpy as np

from fcos import FcosDetectionsEncoder
import pytest

from fcos.encoder import box_centerness
from dataset import VocLabelsCodec


def test_box_centerness():
    test_boxes = [
        [1, 1, 1, 1],
        [1, 1, 2, 2],
        [0, 1, 2, 2], 
    ]

    for box in test_boxes:
        l, t, r, b = box
        target = np.sqrt(
            (min(l, r) * min(t, b)) / (max(l, r) * max(t, b))
        )
        assert target == pytest.approx(box_centerness(torch.tensor(box)))


def test_encode_box_at():
    img_res = (256, 256)
    encoder = FcosDetectionsEncoder(img_res, VocLabelsCodec())

    test_boxes = [
        ([0, 0, 15, 15], 'P3', (1, 2)),
        ([50, 40, 230, 250], 'P7', (0, 0)),
        ([100, 115, 109, 127], 'P4', (14, 13)),
    ]

    for box, level, (y, x) in test_boxes:
        _, (l, t, r, b) = encoder._encode_box_at(box, level, y, x)
        scale = encoder._level_scales[level]
        center_x = (x + 0.5) * scale
        center_y = (y + 0.5) * scale
        x1, y1, x2, y2 = box

        target_l = center_x - x1
        target_r = x2 - center_x
        target_t = center_y - y1
        target_b = y2 - center_y

        assert l == pytest.approx(target_l)
        assert t == pytest.approx(target_t)
        assert r == pytest.approx(target_r)
        assert b == pytest.approx(target_b)


def test_encoder():
    img_res = (256, 256)
    encoder = FcosDetectionsEncoder(img_res, VocLabelsCodec())

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

    encoded_gt = encoder(test_boxes, test_labels)

    classes_tensor = encoded_gt['classes']
    centerness_tensor = encoded_gt['centerness']
    boxes_tensor = encoded_gt['boxes']

    level_names = [f'P{idx}' for idx in range(3, 7 + 1)]
    level_scales = {f'P{idx}': 2 ** idx for idx in range(3, 7 + 1)}

    offset = 0
    ysz, xsz = img_res
    batch_size = len(test_boxes)
    for level in level_names:
        scale = level_scales[level]
        level_ysz = ysz // scale
        level_xsz = xsz // scale
        level_detections = level_xsz * level_ysz

        level_classes = classes_tensor[:, offset:offset+level_detections, :]
        level_boxes = boxes_tensor[:, offset:offset+level_detections, :]
        level_centerness = centerness_tensor[:, offset:offset+level_detections, :]
        offset += level_detections       

        for img_idx in range(batch_size):
            for y_pos in range(level_ysz):
                for x_pos in range(level_xsz):
                    plane_idx = y_pos * level_ysz + x_pos
                    scores = level_classes[img_idx, plane_idx, :]
                    positive_position = torch.any(scores)
                    if positive_position:
                        assert len(test_boxes[img_idx]) > 0

                        pos_center_x = (x_pos + 0.5) * encoder._level_scales[level]
                        pos_center_y = (y_pos + 0.5) * encoder._level_scales[level]

                        ltrb = level_boxes[img_idx, plane_idx, :]
                        cntr = level_centerness[img_idx, plane_idx, 0]

                        smallest_box = None
                        for (x1, y1, x2, y2) in test_boxes[img_idx]:
                            if (x1 < pos_center_x < x2) and (y1 < pos_center_y < y2):
                                regr_min, regr_max = encoder._regression_targets[level]
                                if regr_min <= max(ltrb) < regr_max:
                                    if smallest_box is None:
                                        smallest_box = (x1, y1, x2, y2)
                                    else:
                                        sm_x1, sm_y1, sm_x2, sm_y2 = smallest_box
                                        sm_area = (sm_x2 - sm_x1) * (sm_y2 - sm_y1)
                                        if ((x2 - x1) * (y2 - y1) < sm_area):
                                            smallest_box = (x1, y1, x2, y2)
                        
                        if smallest_box is not None:
                            target_cntr, target_ltrb = encoder._encode_box_at(smallest_box, level, y_pos, x_pos)
                            assert torch.all(torch.eq(target_ltrb, torch.tensor(ltrb)))
                            assert cntr == target_cntr
