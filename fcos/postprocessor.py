import torch
from torch import nn

from fcos.codec import DetectionsCodec


class FcosPostprocessor(nn.Module):
    def __init__(self, img_res, feature_maps, labels_codec):
        super().__init__()
        self._labels = labels_codec
        self._codecs = {
            k: DetectionsCodec(img_res, v, len(self._labels)) for k, v in feature_maps.items()
        }

    def _process_predictions(self, raw_predictions):
        predictions = {}
        for level, ((cls_outs, cntr_outs), regr_outs) in raw_predictions.items():
            codec = self._codecs[level]
            
            b, _, h, w = cls_outs.shape
            cls_outs = torch.reshape(cls_outs, (b, -1, h * w))
            cntr_outs = torch.reshape(cntr_outs, (b, -1, h * w))
            regr_outs = torch.reshape(regr_outs, (b, -1, h * w))
            plane_indices = torch.cat(
                [torch.reshape(torch.range(0, h * w - 1), (1, 1, h * w))] * b,
                axis=0,
            )
            batch_indices = torch.cat(
                [torch.full((1, 1, h * w), i) for i in range(b)],
                axis=0,
            )
            
            flat_outputs = torch.cat(
                [
                    batch_indices, cntr_outs, plane_indices, regr_outs, cls_outs
                ],
                axis=1,
            )
            flat_outputs = torch.permute(flat_outputs, (0, 2, 1)) # b, c, positions -> b, positions, c
            flat_outputs = torch.reshape(flat_outputs, (-1, flat_outputs.shape[-1]))
            
            predictions[level] = codec.decode(flat_outputs)
        
        return predictions

    def forward(self, raw_predictions, scales=None):
        """
        objects: [N x x {batch_idx, centerness, plane_idx, l, t, r, b, class_scores[...]}]
        return: [M x {batch_idx, xmin, ymin, xmax, ymax, scores[...]}]
        M <= N
        """
        detections_per_map = self._process_predictions(raw_predictions)
        all_detections = torch.cat(
            [detections for _, detections in detections_per_map.items()],
            axis=0
        )
        
        batches_cnt = 1 + torch.max(all_detections[:, 0]).int()
        batches = []
        for batch_idx in range(batches_cnt):
            mask = all_detections[:, 0] == batch_idx
            batch_detections = all_detections[mask]
            if scales:
                scale_y, scale_x = scales[batch_idx]
                batch_detections[:, 1] = batch_detections[:, 1] * scale_x
                batch_detections[:, 2] = batch_detections[:, 2] * scale_y
                batch_detections[:, 3] = batch_detections[:, 3] * scale_x
                batch_detections[:, 4] = batch_detections[:, 4] * scale_y
            
            batch_detections = torch.unsqueeze(batch_detections[:, 1:], 0)
            batches.append(batch_detections)

        batches = torch.cat(
            batches,
            axis = 0
        )
        
        return batches
