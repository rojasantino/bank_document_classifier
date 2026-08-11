"""
transfer_models.py
-------------------
Transfer-learning wrappers around torchvision backbones:
  - MobileNetV2      (lightweight, fast inference)
  - ResNet50          (stronger baseline)
  - EfficientNetB0     (final candidate)

Each function returns a ready-to-train nn.Module with its final
classifier layer replaced for NUM_CLASSES, plus a helper to fetch the
correct Grad-CAM target layer for that architecture.

NOTE: pretrained=True downloads ImageNet weights from PyTorch's servers
the first time you run this. If you're offline, pass pretrained=False.
"""

import torch
import torch.nn as nn
from torchvision import models


def build_mobilenet_v2(num_classes, pretrained=True):
    weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.mobilenet_v2(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def build_resnet50(num_classes, pretrained=True):
    weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = models.resnet50(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def build_efficientnet_b0(num_classes, pretrained=True):
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def get_gradcam_target_layer(model_name, model):
    """Returns the last convolutional layer to hook for Grad-CAM, per architecture."""
    if model_name == "mobilenet_v2":
        return model.features[-1]
    if model_name == "resnet50":
        return model.layer4[-1]
    if model_name == "efficientnet_b0":
        return model.features[-1]
    if model_name == "custom_cnn":
        return model.get_gradcam_target_layer()
    raise ValueError(f"Unknown model_name: {model_name}")


BUILDERS = {
    "mobilenet_v2": build_mobilenet_v2,
    "resnet50": build_resnet50,
    "efficientnet_b0": build_efficientnet_b0,
}


def build_model(model_name, num_classes, pretrained=True):
    """Single entry point used by train.py / evaluate.py / api."""
    if model_name == "custom_cnn":
        from src.models.custom_cnn import CustomCNN
        return CustomCNN(num_classes=num_classes)
    if model_name not in BUILDERS:
        raise ValueError(f"Unknown model_name: {model_name}. Choose from {list(BUILDERS) + ['custom_cnn']}")
    return BUILDERS[model_name](num_classes=num_classes, pretrained=pretrained)


if __name__ == "__main__":
    for name in ["mobilenet_v2", "resnet50", "efficientnet_b0"]:
        m = BUILDERS[name](num_classes=8, pretrained=False)
        out = m(torch.randn(2, 3, 224, 224))
        n_params = sum(p.numel() for p in m.parameters())
        print(f"{name}: output={tuple(out.shape)}  params={n_params:,}")
