import torch
from torch import nn


class FcosPostprocessor(nn.Module):
    def __init__(self, img_res, scales=None):
        super(FcosPostprocessor, self).__init__()
        if scales is None:
            scales = [2 ** idx for idx in range(3, 7 + 1)]
        self._centers = FcosPostprocessor.generate_centers(
            img_resolution=img_res,
            scales=scales,
        )

    @staticmethod
    def generate_map_centers(ysz, xsz, scale):
        y = torch.tensor(list(range(ysz))) + 0.5
        x = torch.tensor(list(range(xsz))) + 0.5
        grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
        return scale * torch.reshape(
            torch.cat(
                [grid_y[..., None], grid_x[..., None]],
                axis=-1
            ),
            [-1, 2],
        )

    @staticmethod
    def generate_centers(img_resolution, scales):
        centers = []
        ysz, xsz = img_resolution
        for scale in scales:
            centers.append(
                FcosPostprocessor.generate_map_centers(
                    ysz // scale,
                    xsz // scale,
                    scale,
                )
            )
        return torch.cat(
            centers,
            axis=0,
        )

    def forward(self, core_predictions, scales=None):
        """
        core_predictions:
            'classes': [B, N, classes]
            'centerness': [B, N, 1]
            'boxes': [B, N, 4] - ltrb-encoded
        scales: [(sy_img0, sx_img0), ..., (sy_imgN, sx_imgN)]
        """
        classes = torch.sigmoid(core_predictions['classes'])
        centerness = torch.sigmoid(core_predictions['centerness'])
        boxes = core_predictions['boxes']

        batch_size = len(classes)
        if scales is None:
            scales = [(1., 1.) * batch_size]
        img_scales = torch.Tensor(scales)

        if self._centers.device != boxes.device:
            self._centers = self._centers.to(boxes.device)

        out_boxes = []
        out_ids = []
        out_scores = []
        for img_idx, (scale_y, scale_x) in zip(range(batch_size), img_scales):
            img_classes = classes[img_idx]
            img_centerness = centerness[img_idx]
            l, t, r, b = torch.unbind(boxes[img_idx], axis=-1)
            l, r = l * scale_x, r * scale_x
            t, b = t * scale_y, b * scale_y
            x1 = self._centers[..., 1] - l
            x2 = self._centers[..., 1] + r
            y1 = self._centers[..., 0] - t
            y2 = self._centers[..., 0] + b
            img_boxes = torch.cat(
                [
                    x1[..., None],
                    y1[..., None],
                    x2[..., None],
                    y2[..., None],
                ],
                axis=-1
            )
            img_class_score, img_class_idx = torch.max(img_classes, axis=-1)
            img_scores = img_class_score * torch.squeeze(img_centerness)
            out_boxes.append(img_boxes)
            out_ids.append(img_class_idx)
            out_scores.append(img_scores)

        return {
            "classes": out_ids,
            "scores": out_scores,
            "boxes": out_boxes,
        }
