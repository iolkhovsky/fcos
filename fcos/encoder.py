import numpy as np

from dataset.loader import disbatch
from common.torch_utils import tensor2numpy


def box_centerness(ltrb_box):
    l, t, r, b = ltrb_box
    return np.sqrt(
        (min(l, r) * min(t, b)) / (max(l, r) * max(t, b))
    )


class FcosDetectionsEncoder:
    def __init__(self, res, labels):
        self._res = res
        self._labels = labels
        self._level_names = [f'P{idx}' for idx in range(3, 7 + 1)]
        self._level_scales = {f'P{idx}': 2 ** idx for idx in range(3, 7 + 1)}

        max_in_size = max(self._res)
        self._regression_targets = {
            'P3': (0, max_in_size // 8),
            'P4': (max_in_size // 8, max_in_size // 4),
            'P5': (max_in_size // 4, max_in_size // 2),
            'P6': (max_in_size // 2, max_in_size),
            'P7': (max_in_size, np.inf), 
        }

    def _find_box_positions(self, box):
        positions = {}
        xmin, ymin, xmax, ymax = box
        for fmap, scale in self._level_scales.items():
            cell_xsz, cell_ysz = scale, scale
            xmin_norm = xmin / cell_xsz
            xmax_norm = xmax / cell_xsz
            ymin_norm = ymin / cell_ysz
            ymax_norm = ymax / cell_ysz

            x_first, x_last = int(xmin_norm + 0.5), int(xmax_norm - 0.5)
            y_first, y_last = int(ymin_norm + 0.5), int(ymax_norm - 0.5)

            positions[fmap] = (x_first, y_first, x_last, y_last)
        return positions

    def _encode_box_at(self, box, map, y, x):
        scale = self._level_scales[map]
        center_x = (x + 0.5) * scale
        center_y = (y + 0.5) * scale
        xmin, ymin, xmax, ymax = box
        ltrb_box = np.asarray(
            [
                center_x - xmin,
                center_y - ymin,
                xmax - center_x,
                ymax - center_y
            ]
        )
        return box_centerness(ltrb_box), ltrb_box

    def __call__(self, boxes, labels, objects_amount):
        boxes = tensor2numpy(boxes)
        labels = tensor2numpy(labels)
        objects_amount = tensor2numpy(objects_amount)

        batch_size = len(objects_amount)
        classes = len(self._labels)
        h, w = self._res
        
        classes_tensor = {
            name: np.zeros(shape=(batch_size, h // k, w // k, classes), dtype=np.float32) for name, k in self._level_scales.items()
        }
        centerness_tensor = {
            name: np.zeros(shape=(batch_size, h // k, w // k, 1), dtype=np.float32) for name, k in self._level_scales.items()
        }
        boxes_tensor = {
            name: np.zeros(shape=(batch_size, h // k, w // k, 4), dtype=np.float32) for name, k in self._level_scales.items()
        }

        boxes, labels = disbatch(boxes, labels, objects_amount)
        for img_idx, (img_boxes, img_labels) in enumerate(zip(boxes, labels)):
            if len(boxes) == 0:
                continue

            x1, y1, x2, y2 = img_boxes[:, 0], img_boxes[:, 1], img_boxes[:, 2], img_boxes[:, 3]
            areas = (x2 - x1) * (y2 - y1)
            indices = np.argsort(areas, axis=-1)
            indices = np.flip(indices)

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
                            max_regression_value = max(ltrb)
                            
                            if max_regression_value < regression_min or max_regression_value > regression_max:
                                classes_tensor[feature_map][img_idx, y_pos, x_pos, label] = 0
                            else:
                                centerness_tensor[feature_map][img_idx, y_pos, x_pos, 0] = centerness
                                boxes_tensor[feature_map][img_idx, y_pos, x_pos] = ltrb
    
        encoded_targets = {
            'classes': [],
            'centerness': [],
            'boxes': [],
        }

        for feature_map in self._level_names:
            bhwc_tensor = classes_tensor[feature_map] 
            b, h, w, c = bhwc_tensor.shape
            encoded_targets['classes'].append(
                np.reshape(bhwc_tensor, [b, w * h, c])
            )

            bhwc_tensor = centerness_tensor[feature_map] 
            b, h, w, c = bhwc_tensor.shape
            encoded_targets['centerness'].append(
                np.reshape(bhwc_tensor, [b, w * h, c])
            )

            bhwc_tensor = boxes_tensor[feature_map] 
            b, h, w, c = bhwc_tensor.shape
            encoded_targets['boxes'].append(
                np.reshape(bhwc_tensor, [b, w * h, c])
            )

        return {
            k: np.concatenate(v, axis=1) for k, v in encoded_targets.items()
        }
