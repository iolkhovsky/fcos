from torch import nn

from fcos.preprocessor import FcosPreprocessor
from fcos.core import FcosCore
from fcos.postprocessor import FcosPostprocessor


class FCOS(nn.Module):
    def __init__(self, backbone, labels_codec, res=(512, 512)):
        super().__init__()
        self._labels = labels_codec
        self._preprocessor = FcosPreprocessor(res)
        self._core = FcosCore(backbone, len(self._labels))
        feature_maps = {
            'P3': (64, 64),
            'P4': (32, 32),
            'P5': (16, 16),
            'P6': (8, 8),
            'P7': (4, 4),
        }
        self._postprocessor = FcosPostprocessor(res, feature_maps, self._labels)

    def forward(self, x):
        preprocessed_inputs, scales = self._preprocessor(x)
        core_outputs = self._core(preprocessed_inputs)
        outputs = self._postprocessor(core_outputs, scales)
        return outputs
