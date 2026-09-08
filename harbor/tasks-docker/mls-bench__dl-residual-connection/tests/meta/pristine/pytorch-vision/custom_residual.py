"""CV Residual Connection Benchmark.

Train CIFAR ResNets with custom residual blocks to evaluate
skip/residual connection designs.

FIXED: ResNet backbone, data pipeline, training loop.
EDITABLE: CustomBlock class (residual block design).

Usage:
    python custom_residual.py --arch resnet20 --dataset cifar10 --seed 42
"""

import argparse
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


# ============================================================================
# Residual Block
# ============================================================================

# -- EDITABLE REGION START (lines 31-61) ------------------------------------
class CustomBlock(nn.Module):
    """Custom residual block for CIFAR ResNets.

    Args:
        in_planes: input channels
        planes: output channels
        stride: spatial stride (1 or 2)

    Must set class attribute `expansion = 1` (or 4 for bottleneck).
    The shortcut dimension must match planes * expansion.
    """
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)
# -- EDITABLE REGION END (lines 31-61) --------------------------------------


# ============================================================================
# ResNet Architecture (FIXED)
# ============================================================================

class ResNet(nn.Module):
    """CIFAR-adapted ResNet using CustomBlock.

    Uses 3x3 initial conv (no 7x7), no max pooling, global avg pool at end.
    Standard depths: ResNet-20 ([3,3,3]), ResNet-56 ([9,9,9]), ResNet-110 ([18,18,18]).
    """

    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()
        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        return self.fc(out)


def build_model(arch, num_classes):
    """Build model by architecture name."""
    if arch == 'resnet20':
        return ResNet(CustomBlock, [3, 3, 3], num_classes)
    elif arch == 'resnet56':
        return ResNet(CustomBlock, [9, 9, 9], num_classes)
    elif arch == 'resnet110':
        return ResNet(CustomBlock, [18, 18, 18], num_classes)
    else:
        raise ValueError(f"Unknown architecture: {arch}")


# ============================================================================
# Weight Initialization (FIXED)
# ============================================================================

def initialize_weights(model):
    """Standard Kaiming initialization for all layers."""
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


# ============================================================================
# Data Loading (FIXED)
# ============================================================================

def get_dataloaders(dataset, data_root, batch_size=128, num_workers=4):
    """Create CIFAR train/test dataloaders with standard augmentation."""
    if dataset == 'cifar10':
        mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        num_classes = 10
        Dataset = torchvision.datasets.CIFAR10
    elif dataset == 'cifar100':
        mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
        num_classes = 100
        Dataset = torchvision.datasets.CIFAR100
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_set = Dataset(root=data_root, train=True, download=False, transform=train_transform)
    test_set = Dataset(root=data_root, train=False, download=False, transform=test_transform)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, test_loader, num_classes


# ============================================================================
# Training Loop (FIXED)
# ============================================================================

def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch. Returns (avg_loss, accuracy%)."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()
        total += inputs.size(0)
    return total_loss / total, 100.0 * correct / total


def evaluate(model, loader, criterion, device):
    """Evaluate on test set. Returns (avg_loss, accuracy%)."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += inputs.size(0)
    return total_loss / total, 100.0 * correct / total


def main():
    parser = argparse.ArgumentParser(description="CV Residual Connection Benchmark")
    parser.add_argument('--arch', type=str, required=True,
                        choices=['resnet20', 'resnet56', 'resnet110'])
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['cifar10', 'cifar100'])
    parser.add_argument('--data-root', type=str, default='/data/cifar')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=str, default='.')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data
    train_loader, test_loader, num_classes = get_dataloaders(
        args.dataset, args.data_root, args.batch_size,
    )

    # Model
    model = build_model(args.arch, num_classes)

    # Initialize
    initialize_weights(model)
    model = model.to(device)

    # Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), lr=args.lr,
        momentum=args.momentum, weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Train
    best_acc = 0.0
    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device,
        )
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"TRAIN_METRICS: epoch={epoch+1} train_loss={train_loss:.4f} "
                f"train_acc={train_acc:.2f} test_loss={test_loss:.4f} "
                f"test_acc={test_acc:.2f} lr={optimizer.param_groups[0]['lr']:.6f}",
                flush=True,
            )

        if test_acc > best_acc:
            best_acc = test_acc

    print(f"TEST_METRICS: test_acc={best_acc:.2f}", flush=True)


if __name__ == '__main__':
    main()
