import torch

from fcos.preprocessor import *


def test_preprocessor():
    in_res = [64, 256]
    target_res = [128, 128]
    preprocessor = FcosPreprocessor(target_res)
    test_array = np.random.randint(0, 255, size=in_res + [3,], dtype=np.uint8)
    out, scales = preprocessor(test_array)

    assert isinstance(out, torch.Tensor)
    assert list(out.shape) == [1, 3,] + target_res
    assert out.dtype == torch.float
    assert scales == [(in_res[0] / target_res[0], in_res[1] / target_res[1])]
