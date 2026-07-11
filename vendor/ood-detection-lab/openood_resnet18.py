"""CIFAR ResNet-18 used by the OpenOOD CIFAR-10 benchmark.

The architecture is reproduced from OpenOOD's ``ResNet18_32x32`` under its
MIT license.  Keeping the small definition local makes final verification
offline and pins the exact classifier architecture independently of a
torchvision release.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False,
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False,
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        output = F.relu(self.bn1(self.conv1(value)))
        output = self.bn2(self.conv2(output))
        output = output + self.shortcut(value)
        return F.relu(output)


class ResNet18_32x32(nn.Module):
    """OpenOOD's four-stage ResNet-18 for 32x32 CIFAR images."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, blocks=2, stride=1)
        self.layer2 = self._make_layer(128, blocks=2, stride=2)
        self.layer3 = self._make_layer(256, blocks=2, stride=2)
        self.layer4 = self._make_layer(512, blocks=2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)
        self.feature_size = 512

    def _make_layer(self, planes: int, blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for block_stride in strides:
            layers.append(BasicBlock(self.in_planes, planes, block_stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def representations(
        self, value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the fixed layer-2 pooled and penultimate representations."""
        output = F.relu(self.bn1(self.conv1(value)))
        output = self.layer1(output)
        output = self.layer2(output)
        early = F.adaptive_avg_pool2d(output, 1).flatten(1)
        output = self.layer3(output)
        output = self.layer4(output)
        feature = self.avgpool(output).flatten(1)
        return early, feature

    def forward(
        self, value: torch.Tensor, *, return_feature: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        _early, feature = self.representations(value)
        logits = self.fc(feature)
        if return_feature:
            return logits, feature
        return logits
