"""Exact-weight ResNet-50 models with lambda-conditioned classifiers."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn


WEIGHT_VERSION = "IMAGENET1K_V2"


def build_transform_and_spec(weight_version: str = WEIGHT_VERSION):
    if weight_version != WEIGHT_VERSION:
        raise ValueError(f"Only the immutable {WEIGHT_VERSION} configuration is supported")
    try:
        from torchvision.models import ResNet50_Weights
    except ImportError as exc:
        raise ImportError("torchvision is required for ImageNet100C models") from exc
    weights = ResNet50_Weights.IMAGENET1K_V2
    transform = weights.transforms()
    spec = {
        "weights_enum": "torchvision.models.ResNet50_Weights.IMAGENET1K_V2",
        "resize_size": 232,
        "crop_size": 224,
        "interpolation": "bilinear",
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "rgb": True,
    }
    return transform, spec


class LambdaConditionedClassifier(nn.Module):
    """FiLM-conditioned 100-class head h(z, lambda)."""

    uses_lambda = True

    def __init__(self, feature_dim: int = 2048, num_classes: int = 100, hidden_dim: int = 512):
        super().__init__()
        self.input = nn.Linear(feature_dim, hidden_dim)
        self.gamma = nn.Linear(1, hidden_dim)
        self.beta = nn.Linear(1, hidden_dim)
        self.output = nn.Linear(hidden_dim, num_classes)
        nn.init.ones_(self.gamma.bias)
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)

    def forward(self, features: torch.Tensor, lambda_value: torch.Tensor) -> torch.Tensor:
        if lambda_value.ndim == 0:
            lambda_value = lambda_value.reshape(1, 1).expand(features.shape[0], 1)
        elif lambda_value.ndim == 1:
            lambda_value = lambda_value.reshape(-1, 1)
        if lambda_value.shape[0] == 1 and features.shape[0] != 1:
            lambda_value = lambda_value.expand(features.shape[0], 1)
        hidden = torch.relu(self.input(features))
        hidden = self.gamma(lambda_value) * hidden + self.beta(lambda_value)
        return self.output(torch.relu(hidden))


class ImageNet100CModel(nn.Module):
    uses_lambda = True

    def __init__(self, backbone: nn.Module, mode: str, num_classes: int = 100, hidden_dim: int = 512):
        super().__init__()
        self.backbone = backbone
        self.mode = mode
        self.classifier = LambdaConditionedClassifier(2048, num_classes, hidden_dim)

    def forward(self, images: torch.Tensor, lambda_value: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)
        return self.classifier(features, lambda_value)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.mode == "frozen_feature_pilot":
            self.backbone.eval()
        elif self.mode == "finetune_last_stage":
            # Keep frozen BatchNorm statistics fixed; only layer4 and the head
            # participate in representation adaptation.
            self.backbone.eval()
            self.backbone.layer4.train(mode)
        return self


def build_model(mode: str, num_classes: int = 100, weight_version: str = WEIGHT_VERSION):
    if weight_version != WEIGHT_VERSION:
        raise ValueError(f"Expected {WEIGHT_VERSION}, got {weight_version}")
    try:
        from torchvision.models import ResNet50_Weights, resnet50
    except ImportError as exc:
        raise ImportError("torchvision is required for ImageNet100C models") from exc
    backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    backbone.fc = nn.Identity()
    for parameter in backbone.parameters():
        parameter.requires_grad = False
    if mode == "frozen_feature_pilot":
        backbone.eval()
    elif mode == "finetune_last_stage":
        for parameter in backbone.layer4.parameters():
            parameter.requires_grad = True
    else:
        raise ValueError("mode must be frozen_feature_pilot or finetune_last_stage")
    return ImageNet100CModel(backbone, mode, num_classes=num_classes)


def trainable_parameters(model: nn.Module):
    return [parameter for parameter in model.parameters() if parameter.requires_grad]

