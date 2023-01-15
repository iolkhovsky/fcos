import torch
from torch import nn


class FcosDetectionsCodec(nn.Module):
    def __init__(self):
        super().__init__()

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

    def encode(self):
        raise RuntimeError("Not implemented")
