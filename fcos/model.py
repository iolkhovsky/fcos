import os
import torch
from torch import nn

from fcos.preprocessor import FcosPreprocessor
from fcos.core import FcosCore
from fcos.loss import FcosLoss
from fcos.postprocessor import FcosPostprocessor


class FCOS(nn.Module):
    def __init__(self, backbone, labels_codec, res=(512, 512)):
        super(FCOS, self).__init__()
        self._labels = labels_codec
        self._preprocessor = FcosPreprocessor(res)
        self._core = FcosCore(backbone, len(self._labels))
        self._postprocessor = FcosPostprocessor(img_res=res)
        self._loss = FcosLoss()

    def forward(self, x, target=None):
        preprocessed_inputs, scales = self._preprocessor(x)
        core_outputs = self._core(preprocessed_inputs)
        if self.training:
            return self._loss(pred=core_outputs, target=target)
        else:
            return self._postprocessor(core_outputs, scales)

    def save(self, path):
        torch.save(self.state_dict(), path)

    @staticmethod
    def load(path, backbone, labels_codec, res=(512, 512)):
        model = FCOS(backbone, labels_codec, res)
        model.load_state_dict(torch.load(path))
        model.eval()
        return model
