"""CV Sample Reweighting Benchmark.

Train vision models (ResNet-32, VGG-16-BN) on long-tail imbalanced CIFAR
to evaluate sample reweighting strategies for class-imbalanced classification.

FIXED: Model architectures, imbalanced dataset creation, data pipeline, training loop.
EDITABLE: compute_class_weights() function.

Usage:
    python custom_weighting.py --arch resnet32 --dataset cifar10 --imbalance-ratio 100 --seed 42
"""

import argparse
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset


# ============================================================================
# FIXED
# ============================================================================

# ── Model Architectures ──

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
    """CIFAR-adapted ResNet (He et al., 2016).

    Uses 3x3 initial conv (no 7x7), no max pooling, global avg pool at end.
    ResNet-32: [5,5,5] blocks.
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


class VGG(nn.Module):
    """VGG-16 with BatchNorm, adapted for CIFAR (Simonyan & Zisserman, 2015).

    Uses adaptive avg pool instead of large FC layers, suitable for 32x32 input.
    """

    VGG16_CFG = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M',
                 512, 512, 512, 'M', 512, 512, 512, 'M']

    def __init__(self, num_classes=100):
        super().__init__()
        self.features = self._make_layers(self.VGG16_CFG)
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
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
        return self.classifier(x)


def build_model(arch, num_classes):
    """Build model by architecture name."""
    if arch == 'resnet32':
        return ResNet(BasicBlock, [5, 5, 5], num_classes)
    elif arch == 'vgg16bn':
        return VGG(num_classes)
    else:
        raise ValueError(f"Unknown architecture: {arch}")


# ── Weight Initialization (standard Kaiming) ──

def initialize_weights(model):
    """Standard Kaiming initialization."""
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
# -- EDITABLE REGION START (lines 164-195) ------------------------------------
def compute_class_weights(class_counts, num_classes, config):
    """Compute per-class loss weights for imbalanced classification.

    Called after creating the imbalanced dataset, before training begins.
    The returned weights are used as: nn.CrossEntropyLoss(weight=weights).

    Args:
        class_counts: torch.Tensor of shape [num_classes] — number of training
            samples per class (sorted by class index, class 0 has the most samples).
        num_classes: int — number of classes (10 for CIFAR-10, 100 for CIFAR-100).
        config: dict with keys:
            - imbalance_ratio: float (e.g. 100.0 or 50.0)
            - dataset: str ('cifar10' or 'cifar100')
            - arch: str ('resnet32' or 'vgg16bn')
            - total_samples: int (total training samples after imbalancing)

    Returns:
        torch.Tensor of shape [num_classes] — per-class weights for CrossEntropyLoss.
            Higher weight = more emphasis on that class during training.

    Design considerations:
        - The dataset follows exponential imbalance: class i has
          n_max * (1/imbalance_ratio)^(i/(C-1)) samples.
        - Class 0 (most frequent) may have 5000 samples while class C-1
          (rarest) may have only 50 samples (for ratio=100).
        - Simple uniform weights (no reweighting) tend to bias toward
          frequent classes.
        - Inverse frequency weighting can overfit to rare classes.
        - The optimal strategy balances between these extremes.
    """
    # Default: uniform weights (no reweighting)
    return torch.ones(num_classes)
# -- EDITABLE REGION END (lines 164-195) --------------------------------------

# ============================================================================
# FIXED
# ============================================================================

# ── Imbalanced Dataset Creation ──

def create_imbalanced_cifar(dataset, imbalance_ratio, num_classes, seed=42):
    """Create a long-tail imbalanced version of a CIFAR dataset.

    Uses exponential decay: class i gets n_i = n_max * (1/imbalance_ratio)^(i/(C-1))
    samples, where n_max is the original per-class count.

    Args:
        dataset: torchvision CIFAR dataset (full balanced training set).
        imbalance_ratio: float — ratio between most and least frequent class.
        num_classes: int.

    Returns:
        imbalanced_dataset: Subset with imbalanced class distribution.
        class_counts: torch.Tensor [num_classes] — samples per class.
    """
    targets = np.array(dataset.targets)
    # Original per-class count (CIFAR-10: 5000, CIFAR-100: 500)
    n_max = np.sum(targets == 0)

    # Compute per-class sample counts via exponential decay
    class_counts_np = np.zeros(num_classes, dtype=np.int64)
    for c in range(num_classes):
        mu = (1.0 / imbalance_ratio) ** (c / (num_classes - 1))
        class_counts_np[c] = max(int(n_max * mu), 1)

    # Select subset indices
    selected_indices = []
    rng = np.random.RandomState(seed)
    for c in range(num_classes):
        class_indices = np.where(targets == c)[0]
        rng.shuffle(class_indices)
        selected_indices.extend(class_indices[:class_counts_np[c]])

    imbalanced_dataset = Subset(dataset, selected_indices)
    class_counts = torch.tensor(class_counts_np, dtype=torch.float32)
    return imbalanced_dataset, class_counts


# ── Data Loading ──

def get_dataloaders(dataset_name, data_root, imbalance_ratio, batch_size=128, num_workers=4, seed=42):
    """Create imbalanced CIFAR train and balanced test dataloaders."""
    if dataset_name == 'cifar10':
        mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        num_classes = 10
        Dataset = torchvision.datasets.CIFAR10
    elif dataset_name == 'cifar100':
        mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
        num_classes = 100
        Dataset = torchvision.datasets.CIFAR100
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

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

    full_train_set = Dataset(root=data_root, train=True, download=False, transform=train_transform)
    test_set = Dataset(root=data_root, train=False, download=False, transform=test_transform)

    # Create imbalanced training set
    imbalanced_train, class_counts = create_imbalanced_cifar(
        full_train_set, imbalance_ratio, num_classes, seed,
    )

    train_loader = DataLoader(
        imbalanced_train, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, test_loader, num_classes, class_counts


# ── Training Loop ──

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
    """Evaluate on balanced test set. Returns (avg_loss, accuracy%)."""
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
    parser = argparse.ArgumentParser(description="CV Sample Reweighting Benchmark")
    parser.add_argument('--arch', type=str, required=True,
                        choices=['resnet32', 'vgg16bn'])
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['cifar10', 'cifar100'])
    parser.add_argument('--imbalance-ratio', type=float, required=True,
                        help='Imbalance ratio between most and least frequent class')
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
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data
    train_loader, test_loader, num_classes, class_counts = get_dataloaders(
        args.dataset, args.data_root, args.imbalance_ratio, args.batch_size, seed=args.seed,
    )

    total_samples = int(class_counts.sum().item())
    print(f"Dataset: {args.dataset} (long-tail, imbalance_ratio={args.imbalance_ratio})", flush=True)
    print(f"Total training samples: {total_samples} (balanced would be "
          f"{num_classes * int(class_counts[0].item())})", flush=True)
    print(f"Class counts — max: {int(class_counts[0].item())}, "
          f"min: {int(class_counts[-1].item())}", flush=True)

    # Model
    model = build_model(args.arch, num_classes)
    initialize_weights(model)

    # Compute class weights
    config = {
        'imbalance_ratio': args.imbalance_ratio,
        'dataset': args.dataset,
        'arch': args.arch,
        'total_samples': total_samples,
    }
    weights = compute_class_weights(class_counts, num_classes, config)
    weights = weights.to(device)

    model = model.to(device)

    # Optimizer
    criterion = nn.CrossEntropyLoss(weight=weights)
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
