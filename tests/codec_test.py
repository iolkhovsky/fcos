import torch
import numpy as np

from fcos import FcosDetectionsCodec
from fcos.encoder import box_centerness
from dataset import LabelsCodec


def test_codec_centers():
    scales = torch.tensor([(1, 2), (3, 4), (5, 6)])
    height, width = 2, 3

    centers = FcosDetectionsCodec._generate_centers(height, width, scales)
    assert list(centers.shape) == [len(scales), height, width, 2]

    for batch_idx in range(len(scales)):
        scale_y, scale_x = scales[batch_idx]
        for y_idx in range(height):
            for x_idx in range(width):
                assert scale_y * (y_idx + 0.5) == centers[batch_idx, y_idx, x_idx, 0]
                assert scale_x * (x_idx + 0.5) == centers[batch_idx, y_idx, x_idx, 1]


def test_codec_decode():
    codec = FcosDetectionsCodec((256, 256), None)

    scales = torch.tensor([(1., 2.), (3., 4.)])
    test_data = torch.tensor(
        [[1, 1, 1, 1],
        [1, 2, 3, 4],
        [4, 3, 2, 1],
        [2, 1, 5, 2]]
    )
    test_ltrb_2x2 = torch.reshape(
        test_data,
        shape=[1, 2, 2, 4]
    )
    test_ltrb = test_ltrb_2x2.repeat(2, 1, 1, 1)
    test_ltrb = torch.permute(test_ltrb, [0, 3, 1, 2])

    decoded = codec.decode(
        test_ltrb, scales
    )
    assert list(decoded.shape) == [len(scales), 4, 4]

    for img_idx, image_detections in enumerate(decoded):
        scale_y, scale_x = scales[img_idx]
        for det_idx, x1y1x2y2 in enumerate(image_detections):
            xmin, ymin, xmax, ymax = x1y1x2y2
            left, top, right, bottom = test_data[det_idx]

            y_center = det_idx // 2
            x_center = det_idx - y_center * 2
            x_center = (x_center + 0.5) * scale_x
            y_center = (y_center + 0.5) * scale_y

            assert xmin == x_center - left
            assert ymin == y_center - top
            assert xmax == x_center + right
            assert ymax == y_center + bottom


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
        assert abs(target - box_centerness(torch.tensor(box))) < 1e-4


def test_codec_encode():
    img_res = (256, 256)
    labels_codec = LabelsCodec()
    codec = FcosDetectionsCodec(img_res, labels_codec)

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

    encoded_gt = codec.encode(test_boxes, test_labels)

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

                        pos_center_x = (x_pos + 0.5) * codec._level_scales[level]
                        pos_center_y = (y_pos + 0.5) * codec._level_scales[level]

                        x1y1x2y2 = level_boxes[img_idx, plane_idx, :]
                        cntr, ltrb = codec._encode_box_at(x1y1x2y2, level, y_pos, x_pos)

                        smallest_box, smallest_label = None, None
                        for gt_box, gt_label in zip(test_boxes[img_idx], test_labels[img_idx]):
                            x1, y1, x2, y2 = gt_box
                            if (x1 < pos_center_x < x2) and (y1 < pos_center_y < y2):
                                regr_min, regr_max = codec._regression_targets[level]
                                if regr_min <= max(ltrb) < regr_max:
                                    if smallest_box is None:
                                        smallest_box = gt_box
                                        smallest_label = gt_label
                                    else:
                                        sm_x1, sm_y1, sm_x2, sm_y2 = smallest_box
                                        sm_area = (sm_x2 - sm_x1) * (sm_y2 - sm_y1)
                                        if ((x2 - x1) * (y2 - y1) < sm_area):
                                            smallest_box = gt_box
                                            smallest_label = gt_label
                        
                        assert scores[smallest_label] == 1
                        assert torch.all(torch.eq(x1y1x2y2, torch.tensor(smallest_box)))
                        assert cntr == level_centerness[img_idx, plane_idx, 0]
