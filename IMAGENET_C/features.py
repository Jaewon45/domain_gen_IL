#!/usr/bin/env python3
"""Frozen backbone feature extraction utilities for ImageNet-C."""

from typing import Tuple

import torch
import torch.nn as nn

from datasets import default_imagenet_transforms, maybe_import_torchvision


def load_frozen_resnet50(device: torch.device) -> Tuple[nn.Module, object, int]:
    _, models_module = maybe_import_torchvision()
    transform, weights = default_imagenet_transforms(models_module)
    model = models_module.resnet50(weights=weights)
    model.fc = nn.Identity()
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.to(device)
    return model, transform, 2048


def extract_dataset_features(dataset, feature_extractor: nn.Module, device: torch.device, batch_size: int):
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    features = []
    labels = []
    feature_extractor.eval()
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            batch_features = feature_extractor(inputs)
            features.append(batch_features.detach().cpu())
            labels.append(targets.detach().cpu())
    return torch.cat(features, dim=0), torch.cat(labels, dim=0)