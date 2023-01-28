import numpy as np
import pytest
import torch
from torchvision.ops.focal_loss import sigmoid_focal_loss

from fcos import FocalLoss, CenternessLoss


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
    predicted = torch.from_numpy(np.asarray([[0.05, 0.5, 0.999], [0.3, 0.91, 0.17]], dtype=np.float32))
    targets = torch.from_numpy(np.asarray([[0.1, 1., 0.], [0.7, 0.6, 0.33]], dtype=np.float32))

    loss = CenternessLoss()(pred=predicted, target=targets)

    target_loss = 0.
    for pred_vector, targets_vector in zip(predicted, targets):
        for p, t in zip(pred_vector, targets_vector):
            item_loss = -(t * torch.log(p) + (1 - t) * torch.log(1 - p))
            target_loss += item_loss
    target_loss = target_loss.item()

    assert torch.sum(loss) == pytest.approx(target_loss, 1e-5)


def test_iou_loss():
    pass


def test_fcos_loss():
    pass
