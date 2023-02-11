import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, apply_sigmoid=True):
        super(FocalLoss, self).__init__()
        self._alpha = alpha
        self._gamma = gamma
        self._apply_sigmoid = apply_sigmoid

    def forward(self, pred, target):
        if self._apply_sigmoid:
            pred = F.sigmoid(pred)
        ce = F.binary_cross_entropy(pred, target, reduction='none')
        alpha = target * self._alpha + (1. - target) * (1. - self._alpha)
        pt = torch.where(target == 1,  pred, 1 - pred)
        return alpha * (1. - pt) ** self._gamma * ce


class CenternessLoss(nn.Module):
    def __init__(self, apply_sigmoid=True, threshold=1e-4):
        super(CenternessLoss, self).__init__()
        self._apply_sigmoid = apply_sigmoid
        self._criterion = nn.BCELoss(reduction='none')
        self._thresh = threshold
        self._clamper = lambda x: torch.clamp(x, self._thresh, 1. - self._thresh)

    def forward(self, pred, target):
        if self._apply_sigmoid:
            pred = F.sigmoid(pred)
        pred = self._clamper(pred)
        target = self._clamper(target)
        return self._criterion(pred, target)


class IoULoss(nn.Module):
    def __init__(self):
        super(IoULoss, self).__init__()

    def forward(self, pred, target):
        pred_l, pred_t, pred_r, pred_b = torch.unbind(pred, dim=-1)
        target_l, target_t, target_r, target_b = torch.unbind(target, dim=-1)
        
        intersection_l = torch.min(pred_l, target_l)
        intersection_t = torch.min(pred_t, target_t)
        intersection_r = torch.min(pred_r, target_r)
        intersection_b = torch.min(pred_b, target_b)
        
        intersection = torch.multiply(
            torch.add(intersection_r, intersection_l),
            torch.add(intersection_b, intersection_t),
        )
        predArea = torch.multiply(
            torch.add(pred_r, pred_l),
            torch.add(pred_b, pred_t),
        )
        targetArea = torch.multiply(
            torch.add(target_r, target_l),
            torch.add(target_b, target_t),
        )
        union = predArea + targetArea - intersection
        iou = intersection / (union + 1e-6)
        return -1. * torch.log(iou)


class FcosLoss(nn.Module):
    def __init__(self):
        super(FcosLoss, self).__init__()
        self._clf_loss = FocalLoss()
        self._cntr_loss = CenternessLoss()
        self._regr_loss = IoULoss()

    def forward(self, pred, target, aggregator='sum'):
        aggregator = getattr(torch, aggregator)
        
        _, _, classes = target['classes'].shape
        target_classes = torch.reshape(target['classes'], [-1, classes])
        target_boxes = torch.reshape(target['boxes'], [-1, 4])
        target_centerness = torch.reshape(target['centerness'], [-1, 1])

        pred_classes = torch.reshape(pred['classes'], [-1, classes])
        pred_boxes = torch.reshape(pred['boxes'], [-1, 4])
        pred_centerness = torch.reshape(pred['centerness'], [-1, 1])
        
        positive_samples = torch.sum(target_classes, axis=-1)  
        positive_mask = positive_samples > 0
        positive_samples_cnt = torch.sum(positive_mask)
        assert positive_samples_cnt > 0, f"Targets dont contain any objects"

        class_loss = self._clf_loss(
            pred_classes,
            target_classes,
        )
        class_loss = aggregator(class_loss) / positive_samples_cnt

        pred_boxes_positive = pred_boxes[positive_mask]
        target_boxes_positive = target_boxes[positive_mask]
        pred_centerness_positive = pred_centerness[positive_mask]
        target_centerness_positive = target_centerness[positive_mask]

        centerness_loss = self._cntr_loss(
            pred_centerness_positive,
            target_centerness_positive,
        )
        centerness_loss = aggregator(centerness_loss) / positive_samples_cnt

        regression_loss = self._regr_loss(
            pred_boxes_positive,
            target_boxes_positive,
        )
        regression_loss = aggregator(regression_loss) / positive_samples_cnt

        total_loss = class_loss + centerness_loss + regression_loss

        return {
            'total': total_loss,
            'classification': class_loss.detach().cpu().numpy(),
            'centerness': centerness_loss.detach().cpu().numpy(),
            'regression': regression_loss.detach().cpu().numpy(),
        }
