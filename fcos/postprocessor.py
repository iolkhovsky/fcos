import torch
from torch import nn


class FcosPostprocessor(nn.Module):
    def __init__(self, codec):
        super().__init__()
        self._codec = codec
        self._level_names = [f'P{idx}' for idx in range(3, 7 + 1)]
        self._level_scales = {f'P{idx}': 2 ** idx for idx in range(3, 7 + 1)}

    @staticmethod
    def bchw_to_bnc(tensor):
        b, c, h, w = tensor.shape
        return torch.permute(
            torch.reshape(tensor, [b, c, h * w]),
            [0, 2, 1]
        )

    def forward(self, core_predictions, scales=None):
        """
        core_predictions: {'P3': p3_pred, ..., 'P7': p7_pred}
            pX_pred: ((cls, cntr), regr)
            cls: [b, classes, h, w]
            cntr: [b, 1, h, w]
            regr: [b, 4, h, w]
                encoded bbox: ltrb
        scales: [(sy_img0, sx_img0), ..., (sy_imgN, sx_imgN)]
        """
        batch_size = len(next(iter(core_predictions))[1])
        if scales is None:
            scales = [(1., 1.) * batch_size]

        img_scales = torch.Tensor(scales)
        outputs = {
            'classes': [],
            'centerness': [],
            'boxes': [],
        }

        for level in self._level_names:
            (cls, cntr), ltrb = core_predictions[level]
            map_scales = img_scales * self._level_scales[level]
            outputs['classes'].append(FcosPostprocessor.bchw_to_bnc(cls))
            outputs['centerness'].append(FcosPostprocessor.bchw_to_bnc(cntr))
            outputs['boxes'].append(self._codec.decode(ltrb, map_scales))
        return {
            k: torch.cat(v, axis=1) for k, v in outputs.items()
        }
