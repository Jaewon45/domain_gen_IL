#!/usr/bin/env python3
"""Simple model components for the ImageNet-C extension."""

import torch
import torch.nn as nn


class LinearClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class LambdaConditionedLinearClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.classifier = nn.Linear(input_dim + 1, num_classes)

    def forward(self, x: torch.Tensor, lambda_value: torch.Tensor) -> torch.Tensor:
        if lambda_value.ndim == 0:
            lambda_value = lambda_value.view(1, 1).expand(x.shape[0], 1)
        elif lambda_value.ndim == 1:
            lambda_value = lambda_value.view(-1, 1)
        return self.classifier(torch.cat([x, lambda_value], dim=1))
