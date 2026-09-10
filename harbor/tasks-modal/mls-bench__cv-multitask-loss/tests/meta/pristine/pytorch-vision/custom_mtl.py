"""CV Multi-Task Loss Benchmark.

Train vision models (ResNet, VGG) on CIFAR-100 with TWO classification heads
(fine: 100 classes, coarse: 20 superclasses) to evaluate multi-task loss
combination strategies.

FIXED: Model architectures, data pipeline, training loop.
EDITABLE: MultiTaskLoss class.

Usage:
    python custom_mtl.py --arch resnet20 --seed 42
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
# CIFAR-100 Coarse Label Mapping (FIXED)
# ============================================================================

# Maps each of the 100 fine classes to one of 20 coarse superclasses.
# Source: CIFAR-100 dataset specification (Krizhevsky, 2009).
CIFAR100_COARSE_MAP = [
    4, 1, 14, 8, 0, 6, 7, 7, 18, 3,
    3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
    6, 11, 5, 10, 7, 6, 13, 15, 3, 15,
    0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
    5, 19, 8, 8, 15, 13, 14, 17, 18, 10,
    16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
    10, 3, 2, 12, 12, 16, 12, 1, 9, 19,
    2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
    16, 19, 2, 4, 6, 19, 5, 5, 8, 19,
    18, 1, 2, 15, 6, 0, 17, 8, 14, 13,
]


# ============================================================================
# Model Architectures with Two Heads (FIXED)
# ============================================================================

class BasicBlock(nn.Module):
    """Basic residual block for CIFAR ResNets."""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
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


class ResNet(nn.Module):
    """CIFAR-adapted ResNet with two classification heads.

    Uses 3x3 initial conv (no 7x7), no max pooling, global avg pool at end.
    Standard depths: ResNet-20 ([3,3,3]), ResNet-56 ([9,9,9]).
    Two heads: fc_fine (100 classes) and fc_coarse (20 superclasses).
    """

    def __init__(self, block, num_blocks):
        super().__init__()
        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.fc_fine = nn.Linear(64 * block.expansion, 100)
        self.fc_coarse = nn.Linear(64 * block.expansion, 20)

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
        return self.fc_fine(out), self.fc_coarse(out)


class VGG(nn.Module):
    """VGG-16 with BatchNorm, adapted for CIFAR, with two classification heads.

    Uses adaptive avg pool instead of large FC layers, suitable for 32x32 input.
    Two heads: fine (100 classes) and coarse (20 superclasses).
    """

    VGG16_CFG = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M',
                 512, 512, 512, 'M', 512, 512, 512, 'M']

    def __init__(self):
        super().__init__()
        self.features = self._make_layers(self.VGG16_CFG)
        self.classifier_fine = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(512, 100),
        )
        self.classifier_coarse = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(512, 20),
        )

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers.append(nn.MaxPool2d(2, 2))
            else:
                layers += [
                    nn.Conv2d(in_channels, v, 3, padding=1),
                    nn.BatchNorm2d(v),
                    nn.ReLU(inplace=True),
                ]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        return self.classifier_fine(x), self.classifier_coarse(x)


def build_model(arch):
    """Build model by architecture name (always CIFAR-100 two-head)."""
    if arch == 'resnet20':
        return ResNet(BasicBlock, [3, 3, 3])
    elif arch == 'resnet56':
        return ResNet(BasicBlock, [9, 9, 9])
    elif arch == 'vgg16bn':
        return VGG()
    else:
        raise ValueError(f"Unknown architecture: {arch}")


# ============================================================================
# Weight Initialization (FIXED)
# ============================================================================

def initialize_weights(model):
    """Kaiming initialization for all layers."""
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
# EDITABLE
# ============================================================================

# -- EDITABLE REGION START (lines 195-216) ------------------------------------
class MultiTaskLoss(nn.Module):
    """Multi-task loss combination for fine + coarse classification.

    Args:
        num_tasks: int (always 2)
    """

    def __init__(self, num_tasks=2):
        super().__init__()

    def forward(self, fine_loss, coarse_loss, epoch, total_epochs):
        """Combine fine and coarse classification losses.

        Args:
            fine_loss: scalar tensor, CE loss for 100-class fine prediction
            coarse_loss: scalar tensor, CE loss for 20-class coarse prediction
            epoch: int, current epoch (0-indexed)
            total_epochs: int, total number of training epochs
        Returns:
            combined scalar loss
        """
        return fine_loss + coarse_loss
# -- EDITABLE REGION END (lines 195-216) --------------------------------------


# ============================================================================
# Data Loading (FIXED)
# ============================================================================

class CIFAR100MultiTask(torch.utils.data.Dataset):
    """Wraps CIFAR-100 to return (image, fine_label, coarse_label) tuples."""

    def __init__(self, root, train=True, transform=None, download=False):
        self.dataset = torchvision.datasets.CIFAR100(
            root=root, train=train, transform=transform, download=download,
        )
        self.coarse_map = torch.tensor(CIFAR100_COARSE_MAP, dtype=torch.long)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, fine_label = self.dataset[idx]
        coarse_label = self.coarse_map[fine_label].item()
        return image, fine_label, coarse_label


def get_dataloaders(data_root, batch_size=128, num_workers=4):
    """Create CIFAR-100 multi-task train/test dataloaders."""
    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)

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

    train_set = CIFAR100MultiTask(
        root=data_root, train=True, transform=train_transform, download=False,
    )
    test_set = CIFAR100MultiTask(
        root=data_root, train=False, transform=test_transform, download=False,
    )

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, test_loader


# ============================================================================
# Training Loop (FIXED)
# ============================================================================

def train_epoch(model, loader, mtl_loss, optimizer, device, epoch, total_epochs):
    """Train for one epoch with multi-task loss. Returns (avg_loss, fine_accuracy%)."""
    model.train()
    mtl_loss.train()
    total_loss, correct, total = 0.0, 0, 0
    for inputs, fine_targets, coarse_targets in loader:
        inputs = inputs.to(device)
        fine_targets = fine_targets.to(device)
        coarse_targets = coarse_targets.to(device)

        optimizer.zero_grad()
        fine_logits, coarse_logits = model(inputs)
        fine_loss = F.cross_entropy(fine_logits, fine_targets)
        coarse_loss = F.cross_entropy(coarse_logits, coarse_targets)
        loss = mtl_loss(fine_loss, coarse_loss, epoch, total_epochs)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        _, predicted = fine_logits.max(1)
        correct += predicted.eq(fine_targets).sum().item()
        total += inputs.size(0)
    return total_loss / total, 100.0 * correct / total


def evaluate(model, loader, device):
    """Evaluate on test set. Returns fine-class accuracy%."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, fine_targets, coarse_targets in loader:
            inputs = inputs.to(device)
            fine_targets = fine_targets.to(device)
            fine_logits, _ = model(inputs)
            _, predicted = fine_logits.max(1)
            correct += predicted.eq(fine_targets).sum().item()
            total += inputs.size(0)
    return 100.0 * correct / total


# ============================================================================
# Main (FIXED)
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="CV Multi-Task Loss Benchmark")
    parser.add_argument('--arch', type=str, required=True,
                        choices=['resnet20', 'resnet56', 'vgg16bn'])
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
    train_loader, test_loader = get_dataloaders(
        args.data_root, args.batch_size,
    )

    # Model
    model = build_model(args.arch)
    initialize_weights(model)
    model = model.to(device)

    # Multi-task loss
    mtl_loss = MultiTaskLoss(num_tasks=2).to(device)

    # Optimizer — include mtl_loss parameters (e.g. learnable weights)
    all_params = list(model.parameters()) + list(mtl_loss.parameters())
    optimizer = optim.SGD(
        all_params, lr=args.lr,
        momentum=args.momentum, weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Train
    best_acc = 0.0
    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(
            model, train_loader, mtl_loss, optimizer, device, epoch, args.epochs,
        )
        test_acc = evaluate(model, test_loader, device)
        scheduler.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"TRAIN_METRICS: epoch={epoch+1} train_loss={train_loss:.4f} "
                f"train_acc={train_acc:.2f} test_acc={test_acc:.2f} "
                f"lr={optimizer.param_groups[0]['lr']:.6f}",
                flush=True,
            )

        if test_acc > best_acc:
            best_acc = test_acc

    print(f"TEST_METRICS: test_acc={best_acc:.2f}", flush=True)


if __name__ == '__main__':
    main()
