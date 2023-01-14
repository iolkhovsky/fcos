import torch
from torch import nn
import torchvision
import numpy as np


class FcosPreprocessor(nn.Module):
    def __init__(self, target_resolution=(512, 512)):
        super().__init__()
        self._height, self._width = target_resolution
        self._normalize = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def _prepare_single_image(self, tensor_or_array):
        assert len(tensor_or_array.shape) == 3
        if isinstance(tensor_or_array, np.ndarray):
            tensor = torch.from_numpy(tensor_or_array)
        else:
            tensor = tensor_or_array
        if tensor.shape[0] != 3 and tensor.shape[-1] == 3:
            tensor = torch.permute(tensor, (2, 0, 1))
        _, h, w = tensor.shape
        scale = (1., 1.)
        if (h, w) != (self._height, self._width):
            scale = (h / self._height, w / self._width)
            tensor = torchvision.transforms.Resize(
                size=(self._height, self._width),
                interpolation=torchvision.transforms.InterpolationMode.BILINEAR
            )(tensor)
        return tensor.float(), scale

    def forward(self, img_or_batch):
        if isinstance(img_or_batch, list) or len(img_or_batch.shape) == 4:
            tensors_and_scales = [self._prepare_single_image(x) for x in img_or_batch]
            tensors, scales = zip(*tensors_and_scales)
            tensors = torch.stack(tensors)
        else:
            tensor, scale = self._prepare_single_image(img_or_batch)
            tensors, scales = torch.unsqueeze(tensor, 0), [scale]
        b, c, h, w = tensors.shape
        assert c == 3 and h == self._height and w == self._width and b > 0, \
            f"b, h, w, c = {b, h, w, c}"
        normalized_tensors = self._normalize(tensors)
        return normalized_tensors, scales
