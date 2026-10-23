"""ResNet-50 (He et al., 2016) with an ImageNet-pretrained backbone and a 20-way VOC head."""
import torch
import torch.nn as nn
from torchvision import models

NUM_CLASSES = 20


def build_model(num_classes: int = NUM_CLASSES, pretrained: bool = True) -> nn.Module:
    weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
    m = models.resnet50(weights=weights)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


def target_layers(m: nn.Module) -> dict:
    """Output stages used as Grad-CAM target layers (strides 8, 16, 32)."""
    return {"layer2": m.layer2, "layer3": m.layer3, "layer4": m.layer4}


def load_checkpoint(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    m = build_model(pretrained=False)
    m.load_state_dict(ck["state_dict"])
    return m.to(device).eval(), ck
