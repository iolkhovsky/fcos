import torch
from torch import nn
import numpy as np


def box_centerness(ltrb_box):
    l, t, r, b = ltrb_box
    return torch.sqrt(
        (min(l, r) * min(t, b)) / (max(l, r) * max(t, b))
    )


class FcosDetectionsCodec(nn.Module):
    def __init__(self, res, labels):
        super().__init__()
        self._res = res
        self._labels = labels
        self._level_scales = {f'P{idx}': 2 ** idx for idx in range(3, 7 + 1)}

        max_in_size = max(self._res)
        self._regression_targets = {
            'P3': (0, max_in_size // 8),
            'P4': (max_in_size // 8, max_in_size // 4),
            'P5': (max_in_size // 4, max_in_size // 2),
            'P6': (max_in_size // 2, max_in_size),
            'P7': (max_in_size, np.inf), 
        }

    @staticmethod
    def _generate_centers(map_h, map_w, scales):
        batch_size = len(scales)
        y = torch.tensor(list(range(map_h))) + 0.5
        x = torch.tensor(list(range(map_w))) + 0.5
        grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
        centers = torch.cat(
            [grid_y[..., None], grid_x[..., None]],
            axis=-1
        )[None, ...]
        
        centers = centers.repeat(batch_size, 1, 1, 1)
        centers = centers * scales[:, None, None, :]
        return centers

    def forward(self, x):
        raise RuntimeError("Not implemented")

    def decode(self, ltrb_map, scales):
        b, _, h, w = ltrb_map.shape
        ltrb_map = torch.permute(ltrb_map, [0, 2, 3, 1])  # bchw -> bhwc

        centers = FcosDetectionsCodec._generate_centers(h, w, scales)
        centers_y, centers_x = centers[:, :, :, 0], centers[:, :, :, 1]
        centers_y = centers_y[..., None]
        centers_x = centers_x[..., None]

        xmin = centers_x - ltrb_map[:, :, :, 0, None]
        ymin = centers_y - ltrb_map[:, :, :, 1, None]
        xmax = centers_x + ltrb_map[:, :, :, 2, None]
        ymax = centers_y + ltrb_map[:, :, :, 3, None]

        boxes = torch.cat(
            [xmin, ymin, xmax, ymax],
            axis=-1
        )
        return torch.reshape(boxes, [b, h * w, -1])

    def _find_box_positions(self, box):
        positions = {}
        h, w = self._res
        xmin, ymin, xmax, ymax = box
        for fmap, scale in self._level_scales.items():
            x_size, y_size = scale, scale
            xmin_norm = xmin / x_size
            xmax_norm = xmax / x_size
            ymin_norm = ymin / y_size
            ymax_norm = ymax / y_size

            x_first, x_last = int(xmin_norm + 0.5), int(xmax_norm - 0.5)
            y_first, y_last = int(ymin_norm + 0.5), int(ymax_norm - 0.5)

            positions[fmap] = (x_first, y_first, x_last, y_last)
        return positions

    def _encode_box_at(self, box, map, y, x):
        scale = self._level_scales[map]
        center_x = (x + 0.5) * scale
        center_y = (y + 0.5) * scale
        xmin, ymin, xmax, ymax = box

        ltrb_box = torch.tensor(
            [
                center_x - xmin,
                center_y - ymin,
                xmax - center_x,
                ymax - center_y
            ]
        )

        return box_centerness(ltrb_box), ltrb_box

    def encode(self, boxes, labels):
        batch_size = len(boxes)
        classes = len(self._labels)
        h, w = self._res
        
        classes_tensor = {
            name: torch.zeros(size=(batch_size, h // k, w // k, classes)) for name, k in self._level_scales.items()
        }
        centerness_tensor = {
            name: torch.zeros(size=(batch_size, h // k, w // k, 1)) for name, k in self._level_scales.items()
        }
        boxes_tensor = {
            name: torch.zeros(size=(batch_size, h // k, w // k, 4)) for name, k in self._level_scales.items()
        }

        for img_idx, (img_boxes, img_labels) in enumerate(zip(boxes, labels)):
            if len(img_boxes) == 0:
                continue
            x1, y1, x2, y2 = img_boxes[:, 0], img_boxes[:, 1], img_boxes[:, 2], img_boxes[:, 3]
            areas = (x2 - x1) * (y2 - y1)
            _, indices = torch.sort(areas, descending=True)

            sorted_boxes = img_boxes[indices]
            sorted_labels = img_labels[indices]

            for box, label in zip(sorted_boxes, sorted_labels):
                target_positions = self._find_box_positions(box)

                for feature_map, pos_ranges in target_positions.items():
                    x_first, y_first, x_last, y_last = pos_ranges
                    classes_tensor[feature_map][img_idx, y_first:y_last+1, x_first:x_last+1, label] = 1
                    regression_min, regression_max = self._regression_targets[feature_map]

                    for y_pos in range(y_first, y_last+1):
                        for x_pos in range(x_first, x_last+1):
                            centerness, ltrb = self._encode_box_at(box, map=feature_map, y=y_pos, x=x_pos)
                            centerness_tensor[feature_map][img_idx, y_pos, x_pos, 0] = centerness
                            ltrb_tensor = torch.tensor(ltrb, dtype=torch.float32)
                            if regression_min <= max(ltrb_tensor) < regression_max:
                                boxes_tensor[feature_map][img_idx, y_pos, x_pos] = ltrb_tensor
                            else:
                                classes_tensor[feature_map][img_idx, y_pos, x_pos, label] = 0

        for feature_map in classes_tensor.keys():
            classes_tensor[feature_map] = torch.permute(classes_tensor[feature_map], [0, 3, 1, 2])
            centerness_tensor[feature_map] = torch.permute(centerness_tensor[feature_map], [0, 3, 1, 2])
            boxes_tensor[feature_map] = torch.permute(boxes_tensor[feature_map], [0, 3, 1, 2])

        return classes_tensor, centerness_tensor, boxes_tensor
