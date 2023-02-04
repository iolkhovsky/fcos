import torch

from fcos.heads import *


def test_regression_head():
    batch_size, channels = 2, 32
    res = [32, 64]
    head = RegressionHead(
        layers=3,
        in_channels=channels,
        intermediate_channels=16,
    )
    in_tensor = torch.rand([batch_size, channels,] + res, dtype=torch.float)
    out = head(in_tensor)
    assert list(out.shape) == [batch_size, 4,] + res
    assert torch.all(out >= 0.)


def test_classification_head():
    batch_size, channels, classes = 2, 32, 7
    res = [32, 64]
    head = ClassificationHead(
        classes=classes,
        layers=3,
        in_channels=channels,
        intermediate_channels=16,
    )
    in_tensor = torch.rand([batch_size, channels,] + res, dtype=torch.float)
    scores, centerness = head(in_tensor)
    assert list(scores.shape) == [batch_size, classes,] + res
    assert list(centerness.shape) == [batch_size, 1,] + res
