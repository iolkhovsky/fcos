from torch import nn
import torchvision
from typing import Dict
from collections import OrderedDict

from fcos.layers import Conv2dBN


class IntermediateLayerGetter(nn.ModuleDict):
    """
    Module wrapper that returns intermediate layers from a model
    It has a strong assumption that the modules have been registered
    into the model in the same order as they are used.
    This means that one should **not** reuse the same nn.Module
    twice in the forward if you want this to work.
    Additionally, it is only able to query submodules that are directly
    assigned to the model. So if `model` is passed, `model.feature1` can
    be returned, but not `model.feature1.layer2`.
    Args:
        model (nn.Module): model on which we will extract the features
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).
    Examples::
        >>> m = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)
        >>> # extract layer1 and layer3, giving as names `feat1` and feat2`
        >>> new_m = torchvision.models._utils.IntermediateLayerGetter(m,
        >>>     {'layer1': 'feat1', 'layer3': 'feat2'})
        >>> out = new_m(torch.rand(1, 3, 224, 224))
        >>> print([(k, v.shape) for k, v in out.items()])
        >>>     [('feat1', torch.Size([1, 64, 56, 56])),
        >>>      ('feat2', torch.Size([1, 256, 14, 14]))]
    """

    _version = 2
    __annotations__ = {
        "return_layers": Dict[str, str],
    }

    def __init__(self, model: nn.Module, return_layers: Dict[str, str]) -> None:
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            print([name for name, _ in model.named_children()])
            raise ValueError("return_layers are not present in model")
        orig_return_layers = return_layers
        return_layers = {str(k): str(v) for k, v in return_layers.items()}
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super().__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x):
        out = OrderedDict()
        for name, module in self.items():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out


class FeatureExtractor(nn.Module):
    def __init__(self, backbone, out_channels=256):
        super().__init__()
        self._backbone = backbone
        self._getter = IntermediateLayerGetter(
            model=self._backbone,
            return_layers = {
                'layer2': 'P3',
                'layer3': 'P4',
                'layer4': 'P5',
            }
        )
        self._fpn = torchvision.ops.FeaturePyramidNetwork(
            in_channels_list=[512, 1024, 2048],
            out_channels=out_channels,
        )
        self._conv1 = Conv2dBN(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel=3,
            stride=2,
            padding=1,
            batch_norm=False,
            activation='ReLU',
            activation_pars={},
        )
        self._conv2 = Conv2dBN(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel=3,
            stride=2,
            padding=1,
            batch_norm=False,
            activation='ReLU',
            activation_pars={},
        )

    def forward(self, x):
        feature_maps = self._fpn(self._getter(x))
        feature_maps['P6'] = self._conv1(feature_maps['P5'])
        feature_maps['P7'] = self._conv1(feature_maps['P6'])
        return feature_maps

    @property
    def feature_maps(self):
        return ['P3', 'P4', 'P5', 'P6', 'P7']
