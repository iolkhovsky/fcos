import numpy as np
import pytest
import torch
from torchvision.ops.focal_loss import sigmoid_focal_loss

from fcos import FocalLoss, CenternessLoss, IoULoss


def test_focal_loss():
    alpha, gamma = 0.25, 2
    logits = torch.from_numpy(np.asarray([[0.05, 0.3, -2], [0.5, -0.01, 0.03]], dtype=np.float32))
    scores = torch.sigmoid(logits)
    targets = torch.from_numpy(np.asarray([[0, 1, 0], [0, 0, 1]], dtype=np.float32))

    fl_wo_sigmoid = FocalLoss(alpha=alpha, gamma=gamma, apply_sigmoid=False)
    loss_tensor = fl_wo_sigmoid(pred=scores, target=targets)

    target_tensor = sigmoid_focal_loss(
        inputs=logits,
        targets=targets,
    )

    assert torch.allclose(loss_tensor, target_tensor, rtol=1e-05, atol=1e-08, equal_nan=False)

    fl_with_sigmoid = FocalLoss(alpha=alpha, gamma=gamma, apply_sigmoid=True)
    loss_tensor = fl_with_sigmoid(pred=logits, target=targets)

    assert torch.allclose(loss_tensor, target_tensor, rtol=1e-05, atol=1e-08, equal_nan=False)


def test_focal_loss_legacy():
    alpha, gamma = 0.25, 2
    scores = torch.sigmoid(
        torch.from_numpy(np.asarray([[0.05, 0.3, -2], [0.5, -0.01, 0.03]], dtype=np.float32))
    )
    targets = torch.from_numpy(np.asarray([[0, 1, 0], [0, 0, 1]], dtype=np.float32))

    target_loss = 0.
    for scores_vector, targets_vector in zip(scores, targets):
        for p, t in zip(scores_vector, targets_vector):
            pt = p if t else 1. - p
            alpha_t = alpha if t else 1. - alpha
            target_loss += -1. * alpha_t * ((1. - pt) ** gamma) * torch.log(pt)
    target_loss = target_loss.item()

    fl_wo_sigmoid = FocalLoss(alpha=alpha, gamma=gamma, apply_sigmoid=False)
    loss_tensor = fl_wo_sigmoid(pred=scores, target=targets)

    assert torch.sum(loss_tensor) == pytest.approx(target_loss, 1e-5)


def test_centerness_loss():
    predicted = torch.from_numpy(np.asarray([[-0.05, -0.5, 0.999], [0.3, -0.91, 0.17]], dtype=np.float32))
    targets = torch.from_numpy(np.asarray([[0.1, 1., 0.], [0.7, 0.6, 0.33]], dtype=np.float32))
    threshold = 1e-3

    loss = CenternessLoss(threshold=threshold)(pred=predicted, target=targets)

    target_loss = 0.
    for pred_vector, targets_vector in zip(predicted, targets):
        for p, t in zip(pred_vector, targets_vector):
            p = torch.sigmoid(p)
            p = torch.clamp(p, min=threshold, max=1. - threshold)
            t = torch.clamp(t, min=threshold, max=1. - threshold)
            item_loss = -(t * torch.log(p) + (1 - t) * torch.log(1 - p))
            target_loss += item_loss
    target_loss = target_loss.item()

    assert torch.sum(loss) == pytest.approx(target_loss, 1e-5)


def test_iou_loss():
    predicted_ltrb = [
        [1., 2., 3., 4.],
        [0., 0., 1., 1.],
        [10., 10., 5., 5.],
    ]
    predicted_ltrb = torch.from_numpy(np.asarray(predicted_ltrb, dtype=np.float32))

    target_ltrb = [
        [2., 3., 5., 7.],
        [1., 1., 2., 2.],
        [10., 10., 5., 5.],
    ]
    target_ltrb = torch.from_numpy(np.asarray(target_ltrb, dtype=np.float32))

    criterion = IoULoss()
    loss = criterion(
        pred=predicted_ltrb,
        target=target_ltrb,
    )

    def iou(box_1, box_2):
        l1, t1, r1, b1 = box_1
        l2, t2, r2, b2 = box_2

        inter_l = min(l1, l2)
        inter_r = min(r1, r2)
        inter_t = min(t1, t2)
        inter_b = min(b1, b2)

        inter_xsz = inter_l + inter_r
        inter_ysz = inter_t + inter_b

        inter = inter_xsz * inter_ysz
        area_1 = (l1 + r1) * (t1 + b1)
        area_2 = (l2 + r2) * (t2 + b2)
        union = area_1 + area_2 - inter
        return inter / union

    target_loss = 0.
    for pred_box, targets_box in zip(predicted_ltrb, target_ltrb):
        target_loss += -torch.log(iou(pred_box, targets_box))

    assert torch.sum(loss) == pytest.approx(target_loss, 1e-5)
