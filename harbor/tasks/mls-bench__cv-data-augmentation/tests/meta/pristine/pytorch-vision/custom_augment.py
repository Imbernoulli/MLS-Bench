"""CV Data Augmentation Benchmark.

Train vision models (ResNet, VGG, MobileNetV2) on CIFAR-10/100/FashionMNIST to evaluate
data augmentation strategies.

FIXED: Model architectures, weight initialization, test transform, data loading, training loop.
EDITABLE: build_train_transform() function.

Usage:
    python custom_augment.py --arch resnet20 --dataset cifar10 --seed 42
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
# Model Architectures (FIXED)
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
    """CIFAR-adapted ResNet (He et al., 2016).

    Uses 3x3 initial conv (no 7x7), no max pooling, global avg pool at end.
    Standard depths: ResNet-20 ([3,3,3]), ResNet-56 ([9,9,9]).
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


class InvertedResidual(nn.Module):
    """MobileNetV2 inverted residual block (Sandler et al., 2018)."""

    def __init__(self, inp, oup, stride, expand_ratio):
        super().__init__()
        self.stride = stride
        hidden = int(round(inp * expand_ratio))
        self.use_res = (stride == 1 and inp == oup)
        layers = []
        if expand_ratio != 1:
            layers += [
                nn.Conv2d(inp, hidden, 1, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU6(inplace=True),
            ]
        layers += [
            nn.Conv2d(hidden, hidden, 3, stride=stride, padding=1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden, oup, 1, bias=False),
            nn.BatchNorm2d(oup),
        ]
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res:
            return x + self.conv(x)
        return self.conv(x)


class MobileNetV2(nn.Module):
    """MobileNetV2 adapted for CIFAR/small-image input (Sandler et al., 2018).

    Uses stride-1 initial conv (no stride-2) for 32x32 input.
    Width multiplier = 1.0, ~2.2M parameters.
    """

    CFG = [
        # expand_ratio, channels, num_blocks, stride
        [1, 16, 1, 1],
        [6, 24, 2, 1],
        [6, 32, 3, 2],
        [6, 64, 4, 2],
        [6, 96, 3, 1],
        [6, 160, 3, 2],
        [6, 320, 1, 1],
    ]

    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
        )
        layers = []
        inp = 32
        for t, c, n, s in self.CFG:
            for i in range(n):
                stride = s if i == 0 else 1
                layers.append(InvertedResidual(inp, c, stride, t))
                inp = c
        self.layers = nn.Sequential(*layers)
        self.conv_last = nn.Sequential(
            nn.Conv2d(320, 1280, 1, bias=False),
            nn.BatchNorm2d(1280),
            nn.ReLU6(inplace=True),
        )
        self.fc = nn.Linear(1280, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layers(x)
        x = self.conv_last(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def build_model(arch, num_classes):
    """Build model by architecture name."""
    if arch == 'resnet20':
        return ResNet(BasicBlock, [3, 3, 3], num_classes)
    elif arch == 'resnet56':
        return ResNet(BasicBlock, [9, 9, 9], num_classes)
    elif arch == 'vgg16bn':
        return VGG(num_classes)
    elif arch == 'mobilenetv2':
        return MobileNetV2(num_classes)
    else:
        raise ValueError(f"Unknown architecture: {arch}")


# ============================================================================
# Weight Initialization (FIXED)
# ============================================================================

def initialize_weights(model):
    """Kaiming normal initialization (standard)."""
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
# Data Augmentation
# ============================================================================

# -- EDITABLE REGION START (lines 246-275) ------------------------------------
def build_train_transform(config):
    """Build training data transform pipeline.

    Called before creating the training dataset. Must return a complete
    transforms.Compose pipeline including ToTensor() and Normalize().

    Args:
        config: dict with keys:
            - img_size: int (32 for CIFAR)
            - mean: tuple of floats (per-channel mean)
            - std: tuple of floats (per-channel std)
            - dataset: str ('cifar10' or 'cifar100')

    Returns:
        transforms.Compose -- complete training transform pipeline.

    Design considerations:
        - Geometric transforms (crop, flip, rotation, affine)
        - Color/photometric transforms (jitter, equalize, posterize)
        - Erasing/masking strategies (cutout, random erasing)
        - Automated augmentation policies (AutoAugment, RandAugment, TrivialAugment)
        - Mixing strategies applied at the tensor level (after ToTensor)
        - Regularization via input perturbation
    """
    return transforms.Compose([
        transforms.RandomCrop(config['img_size'], padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(config['mean'], config['std']),
    ])
# -- EDITABLE REGION END (lines 246-275) --------------------------------------


# ============================================================================
# Data Loading (FIXED)
# ============================================================================

def get_dataloaders(dataset, data_root, batch_size=128, num_workers=4):
    """Create train/test dataloaders.

    Train transform is built by build_train_transform() (editable).
    Test transform is fixed (no augmentation).
    """
    if dataset == 'cifar10':
        mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        num_classes = 10
        Dataset = torchvision.datasets.CIFAR10
    elif dataset == 'cifar100':
        mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
        num_classes = 100
        Dataset = torchvision.datasets.CIFAR100
    elif dataset == 'fmnist':
        mean, std = (0.2860, 0.2860, 0.2860), (0.3530, 0.3530, 0.3530)
        num_classes = 10
        Dataset = torchvision.datasets.FashionMNIST
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    is_grayscale = (dataset == 'fmnist')
    _repeat3 = transforms.Lambda(lambda x: x.repeat(3, 1, 1))

    config = {
        'img_size': 32,
        'mean': mean,
        'std': std,
        'dataset': dataset,
    }
    train_transform = build_train_transform(config)
    # For grayscale datasets, wrap user transform: Resize + user pipeline + channel repeat
    if is_grayscale:
        user_ops = list(train_transform.transforms)
        # Insert Resize at the front (before any spatial augmentation)
        user_ops.insert(0, transforms.Resize(32))
        # Find where ToTensor is and insert channel repeat right after it
        for i, t in enumerate(user_ops):
            if isinstance(t, transforms.ToTensor):
                user_ops.insert(i + 1, _repeat3)
                break
        train_transform = transforms.Compose(user_ops)

    if is_grayscale:
        test_transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            _repeat3,
            transforms.Normalize(mean, std),
        ])
    else:
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
    parser = argparse.ArgumentParser(description="CV Data Augmentation Benchmark")
    parser.add_argument('--arch', type=str, required=True,
                        choices=['resnet20', 'resnet56', 'vgg16bn', 'mobilenetv2'])
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['cifar10', 'cifar100', 'fmnist'])
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
