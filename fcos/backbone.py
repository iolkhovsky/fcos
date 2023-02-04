import torchvision
from torchvision.models import *


BACKBONES = {
    'resnet50': 'ResNet50_Weights'
}


def build_backbone(model_type):
    assert model_type in BACKBONES, f"{model_type} is not valid backbone type"
    model = getattr(torchvision.models, model_type)
    weights = getattr(torchvision.models, BACKBONES[model_type])
    return model(weights=weights.DEFAULT)
