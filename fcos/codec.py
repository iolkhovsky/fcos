import torch


def box_centerness(ltrb_box):
    l, t, r, b = ltrb_box
    return torch.sqrt(
        (min(l, r) * min(t, b)) / (max(l, r) * max(t, b))
    )


class DetectionsCodec:
    BATCH_IDX = 0
    CENTERS_IDX = 1
    PLANE_IDX = 2
    LEFT_IDX = 3
    TOP_IDX = 4
    RIGHT_IDX = 5
    BOTTOM_IDX = 6
    LABELS_IDX = 7
    
    def __init__(self, img_res, fmap_res, classes):
        self._classes = classes
        self._img_xsz, self._img_ysz = img_res
        self._map_xsz, self._map_ysz = fmap_res
        self._stride_x = self._img_xsz / self._map_xsz
        self._stride_y = self._img_ysz / self._map_ysz

    def _sort_objects_by_area(self, image_objects):
        xsize = image_objects[:, 3] - image_objects[:, 1]
        ysize = image_objects[:, 4] - image_objects[:, 2]
        areas = torch.mul(xsize, ysize)
        _, indices = torch.sort(areas, descending=True)
        return image_objects[indices]

    def _get_map_positions(self, obj):
        xmin, ymin, xmax, ymax = obj[1:5]
        xmin_norm = xmin / self._stride_x
        xmax_norm = xmax / self._stride_x
        ymin_norm = ymin / self._stride_y
        ymax_norm = ymax / self._stride_y
        x_idx_first, x_idx_last = int(xmin_norm + 0.5), int(xmax_norm - 0.5)
        y_idx_first, y_idx_last = int(ymin_norm + 0.5), int(ymax_norm - 0.5)

        def validate(v, l, h):
            return min(h, max(l, v))
        
        x_idx_first = validate(x_idx_first, 0, self._map_xsz - 1)
        x_idx_last = validate(x_idx_last, 0, self._map_xsz - 1)
        y_idx_first = validate(y_idx_first, 0, self._map_ysz - 1)
        y_idx_last = validate(y_idx_last, 0, self._map_ysz - 1)
        return (
            (x_idx_first, x_idx_last),
            (y_idx_first, y_idx_last),
        )

    def _encode_signle_position(self, map_pos, obj):
        x_idx, y_idx = map_pos
        xmin, ymin, xmax, ymax = obj[1:5]
        
        batch_idx = obj[0]
        plane_idx = y_idx * self._map_xsz + x_idx
        l = (x_idx + 0.5) * self._stride_x - xmin
        t = (y_idx + 0.5) * self._stride_y - ymin
        r = xmax - (x_idx + 0.5) * self._stride_x
        b = ymax - (y_idx + 0.5) * self._stride_y
        centerness = box_centerness((l, t, r, b))
        class_idx = obj[-1]
        
        return [batch_idx, centerness, plane_idx, l, t, r, b, class_idx]
    
    def _encode_one_object(self, obj):
        x_interval, y_interval = self._get_map_positions(obj)
        res = []

        for map_x in range(x_interval[0], x_interval[1] + 1):
            for map_y in range(y_interval[0], y_interval[1] + 1):
                encoded_position = self._encode_signle_position((map_x, map_y), obj)
                res.append(encoded_position)

        return torch.Tensor(res)

    def _scatter_encoded(self, encoded, targets):
        for encoded_position in encoded:
            plane_index = encoded_position[DetectionsCodec.PLANE_IDX].int()
            class_idx = encoded_position[-1].int()
            targets[plane_index][:DetectionsCodec.LABELS_IDX] = encoded_position[:DetectionsCodec.LABELS_IDX]
            targets[plane_index][DetectionsCodec.LABELS_IDX + class_idx] = 1

    def encode(self, objects):
        """
        objects: [N x {batch_idx, xmin, ymin, xmax, ymax, class_idx}]
        return: [M x {batch_idx, centerness, plane_idx, l, t, r, b, labels[...]}]
        M >= N
        """
        total_batches = (torch.max(objects[:, DetectionsCodec.BATCH_IDX]) + 1).int()
        total_positions = self._map_xsz * self._map_ysz
        encoded_targets = []
        
        for batch_idx in range(total_batches):
            img_targets = torch.zeros(total_positions, self._classes + DetectionsCodec.LABELS_IDX)

            image_objects = objects[objects[:, DetectionsCodec.BATCH_IDX] == batch_idx]
            if len(image_objects) == 0:
                continue

            image_objects = self._sort_objects_by_area(image_objects)
            for obj in image_objects:
                encoded_positions = self._encode_one_object(obj)
                self._scatter_encoded(encoded_positions, img_targets)          
            
            positive_positions_mask = torch.sum(img_targets[:, DetectionsCodec.LABELS_IDX:], dim=1) > 0
            encoded_targets.append(img_targets[positive_positions_mask])
        
        return torch.vstack(encoded_targets)

    def _get_centers(self, indices):
        assert len(indices.shape) == 1, indices.shape
        n = indices.shape[0]
        centers = torch.zeros(n, 2).float()
        centers[:, 1] = indices[:] // self._map_xsz
        centers[:, 0] = indices[:] - centers[:, 1] * self._map_xsz
        centers[:, 0] = (centers[:, 0] + 0.5) * self._stride_x
        centers[:, 1] = (centers[:, 1] + 0.5) * self._stride_y
        return centers

    def decode(self, encoded):
        """
        objects: [N x x {batch_idx, centerness, plane_idx, l, t, r, b, class_scores[...]}]
        return: [M x {batch_idx, xmin, ymin, xmax, ymax, scores[...]}]
        M <= N
        """

        batch_indices = encoded[:, DetectionsCodec.BATCH_IDX]
        centerness = encoded[:, DetectionsCodec.CENTERS_IDX]
        plane_indices = encoded[:, DetectionsCodec.PLANE_IDX]

        centers = self._get_centers(plane_indices)
        xmin = centers[..., 0] - encoded[:, DetectionsCodec.LEFT_IDX]
        xmax = centers[..., 0] + encoded[:, DetectionsCodec.RIGHT_IDX]
        ymin = centers[..., 1] - encoded[..., DetectionsCodec.TOP_IDX]
        ymax = centers[..., 1] + encoded[..., DetectionsCodec.BOTTOM_IDX]
        
        scores = encoded[:, DetectionsCodec.LABELS_IDX:]
        for offset in range(self._classes):
            scores[:, offset] = torch.mul(
                 scores[:, offset], centerness
            )
        
        return torch.cat(
            [
                batch_indices[:, None],
                xmin[:, None],
                ymin[:, None],
                xmax[:, None],
                ymax[:, None],
                scores
            ],
            axis=1,
        )
