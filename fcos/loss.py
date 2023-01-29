import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, apply_sigmoid=False):
        super(FocalLoss, self).__init__()
        self._alpha = alpha
        self._gamma = gamma
        self._apply_sigmoid = apply_sigmoid

    def forward(self, pred, target):
        if self._apply_sigmoid:
            pred = pred.sigmoid()
        ce = F.binary_cross_entropy(pred, target, reduction='none')
        alpha = target * self._alpha + (1. - target) * (1. - self._alpha)
        pt = torch.where(target == 1,  pred, 1 - pred)
        return alpha * (1. - pt) ** self._gamma * ce


class CenternessLoss(nn.Module):
    def __init__(self):
        super(CenternessLoss, self).__init__()
        self._criterion = nn.BCELoss(reduction='none')

    def forward(self, pred, target):
        return self._criterion(pred, target)


class IoULoss(nn.Module):
    def __init__(self):
        super(IoULoss, self).__init__()

    def forward(self, pred, target):     
        pred_l, target_l = pred[..., 0], target[..., 0]
        pred_t, target_t = pred[..., 1], target[..., 1]
        pred_r, target_r = pred[..., 2], target[..., 2]
        pred_b, target_b = pred[..., 3], target[..., 3]
        
        intersection_l = torch.min(pred_l, target_l)
        intersection_t = torch.min(pred_t, target_t)
        intersection_r = torch.min(pred_r, target_r)
        intersection_b = torch.min(pred_b, target_b)
        
        intersection = (intersection_r + intersection_l) * (intersection_b + intersection_t)
        predArea = (pred_r + pred_l) * (pred_b + pred_t)
        targetArea = (target_r + target_l) * (target_b + target_t)
        union = predArea + targetArea - intersection
        iou = intersection / (union + 1e-6)
        
        return 1. - iou


class FcosLoss(nn.Module):
    def __init__(self):
        super(FcosLoss, self).__init__()
        self._clf_loss = FocalLoss()
        self._cntr_loss = CenternessLoss()
        self._regr_loss = IoULoss()

    def forward(self, pred, target, aggregator='sum'):     
        aggregator = getattr(torch, aggregator)
        
        target_classes = target['classes']
        target_boxes = target['boxes']
        target_centerness = target['centerness']

        pred_classes = pred['classes']
        pred_boxes = pred['boxes']
        pred_centerness = pred['centerness']
        
        positive_samples = torch.sum(target_classes, axis=-1)  
        positive_mask = positive_samples > 0
        positive_samples_cnt = torch.sum(positive_mask)
        print("Positive samples cnt:", positive_samples_cnt)
        if positive_samples_cnt == 0:
            return None
        
        class_loss = self._clf_loss(
            pred_classes,
            target_classes.to(pred_classes.device),
        )
        class_loss = aggregator(class_loss) / positive_samples_cnt

        pred_boxes_positive = torch.reshape(pred_boxes[positive_mask], [-1, 4])
        target_boxes_positive = torch.reshape(target_boxes[positive_mask], [-1, 4])
        pred_centerness_positive = torch.reshape(pred_centerness[positive_mask], [-1, 1])
        target_centerness_positive = torch.reshape(target_centerness[positive_mask], [-1, 1])

        centerness_loss = self._cntr_loss(
            pred_centerness_positive,
            target_centerness_positive.to(pred_centerness_positive.device),
        )
        centerness_loss = aggregator(centerness_loss) / positive_samples_cnt

        regression_loss = self._regr_loss(
            pred_boxes_positive,
            target_boxes_positive.to(pred_boxes_positive.device),
        )
        regression_loss = aggregator(regression_loss) / positive_samples_cnt

        return class_loss, centerness_loss, regression_loss
