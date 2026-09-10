"""Research-scale evaluation harness for poison-robust learning.

Train standard vision models (ResNet-20, VGG-16-BN, MobileNetV2) on
CIFAR-10/100/FashionMNIST with label-flip poisoning. The agent's custom
RobustLoss replaces nn.CrossEntropyLoss in the training loop.

FIXED: Model architectures, data pipeline, training schedule, poison injection.
EDITABLE: RobustLoss class in custom_robust_loss.py.
"""

import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

from custom_robust_loss import RobustLoss

_DATA_ROOT = os.environ.get("DATA_ROOT", "/data")


# ============================================================================
# Model Architectures (FIXED) -- identical to dl-activation-function but
# using nn.ReLU() instead of CustomActivation.
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
        self.act = nn.ReLU()
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion),
            )

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return self.act(out)


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
        self.act = nn.ReLU()
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
        out = self.act(self.bn1(self.conv1(x)))
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
            nn.ReLU(),
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
                    nn.ReLU(),
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
                nn.ReLU(),
            ]
        layers += [
            nn.Conv2d(hidden, hidden, 3, stride=stride, padding=1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(),
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
            nn.ReLU(),
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
            nn.ReLU(),
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
# Data Loading (FIXED)
# ============================================================================

def get_datasets(dataset, data_root):
    """Create train/test datasets with standard augmentation."""
    if dataset == 'cifar10':
        mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        num_classes = 10
        DatasetClass = torchvision.datasets.CIFAR10
    elif dataset == 'cifar100':
        mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
        num_classes = 100
        DatasetClass = torchvision.datasets.CIFAR100
    elif dataset == 'fmnist':
        mean, std = (0.2860, 0.2860, 0.2860), (0.3530, 0.3530, 0.3530)
        num_classes = 10
        DatasetClass = torchvision.datasets.FashionMNIST
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    is_grayscale = (dataset == 'fmnist')

    train_transform_list = [
        transforms.Resize(32),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
    if is_grayscale:
        train_transform_list.append(transforms.Lambda(lambda x: x.repeat(3, 1, 1)))
    train_transform_list.append(transforms.Normalize(mean, std))
    train_transform = transforms.Compose(train_transform_list)

    test_transform_list = [
        transforms.Resize(32),
        transforms.ToTensor(),
    ]
    if is_grayscale:
        test_transform_list.append(transforms.Lambda(lambda x: x.repeat(3, 1, 1)))
    test_transform_list.append(transforms.Normalize(mean, std))
    test_transform = transforms.Compose(test_transform_list)

    train_set = DatasetClass(root=data_root, train=True, download=False, transform=train_transform)
    test_set = DatasetClass(root=data_root, train=False, download=False, transform=test_transform)

    return train_set, test_set, num_classes


# ============================================================================
# Poison Injection (FIXED)
# ============================================================================

class PoisonedDataset(Dataset):
    """Wraps a dataset with label-flip poisoning on a random subset."""

    def __init__(self, base_dataset, num_classes, poison_fraction, seed):
        self.base_dataset = base_dataset
        self.num_classes = num_classes

        n = len(base_dataset)
        # Collect original labels (use .targets for speed if available)
        if hasattr(base_dataset, 'targets'):
            raw = base_dataset.targets
            self.clean_labels = torch.tensor(raw, dtype=torch.long) if not isinstance(raw, torch.Tensor) else raw.clone().long()
        else:
            labels = []
            for i in range(n):
                _, lbl = base_dataset[i]
                labels.append(lbl)
            self.clean_labels = torch.tensor(labels, dtype=torch.long)
        self.labels = self.clean_labels.clone()

        # Randomly select poison indices (seeded)
        g = torch.Generator().manual_seed(seed)
        n_poison = int(n * poison_fraction)
        self.poison_idx = torch.randperm(n, generator=g)[:n_poison]

        # Flip selected labels: label -> (label + 1) % num_classes
        self.labels[self.poison_idx] = (self.labels[self.poison_idx] + 1) % num_classes

        # Boolean mask for fast lookup
        self.is_poison = torch.zeros(n, dtype=torch.bool)
        self.is_poison[self.poison_idx] = True

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img, _ = self.base_dataset[idx]
        return img, self.labels[idx].item()


# ============================================================================
# Evaluation (FIXED)
# ============================================================================

@torch.no_grad()
def evaluate_test_acc(model, loader, device):
    """Accuracy on clean test set."""
    model.eval()
    correct, total = 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        preds = model(images).argmax(dim=1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)
    return correct / max(total, 1)


@torch.no_grad()
def evaluate_poison_fit(model, poisoned_dataset, device, batch_size):
    """Fraction of poisoned samples where model predicts the WRONG (poisoned) label."""
    model.eval()
    loader = DataLoader(poisoned_dataset, batch_size=batch_size, shuffle=False,
                        num_workers=4, pin_memory=True)
    poison_correct, poison_total = 0, 0
    offset = 0
    for images, labels in loader:
        bsz = images.size(0)
        mask = poisoned_dataset.is_poison[offset:offset + bsz]
        offset += bsz
        if not mask.any():
            continue
        images = images.to(device)
        preds = model(images).argmax(dim=1).cpu()
        # labels from __getitem__ are already the poisoned labels
        poison_correct += preds[mask].eq(labels[mask]).sum().item()
        poison_total += int(mask.sum().item())
    return poison_correct / max(poison_total, 1)


# ============================================================================
# Main (FIXED)
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Poison-Robust Learning Benchmark")
    parser.add_argument('--arch', type=str, required=True,
                        choices=['resnet20', 'resnet56', 'vgg16bn', 'mobilenetv2'])
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['cifar10', 'cifar100', 'fmnist'])
    parser.add_argument('--data-root', type=str, default='/data/cifar')
    parser.add_argument('--poison-fraction', type=float, required=True)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data
    train_set, test_set, num_classes = get_datasets(
        args.dataset, args.data_root,
    )

    # Poison the training set
    poisoned_train = PoisonedDataset(train_set, num_classes, args.poison_fraction, args.seed)
    train_loader = DataLoader(poisoned_train, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                             num_workers=4, pin_memory=True)

    print(f"Dataset: {args.dataset}, Arch: {args.arch}, "
          f"Poison fraction: {args.poison_fraction:.2f}, "
          f"Poisoned samples: {len(poisoned_train.poison_idx)}/{len(poisoned_train)}",
          flush=True)

    # Model
    model = build_model(args.arch, num_classes)
    initialize_weights(model)
    model = model.to(device)

    # Optimizer + scheduler
    optimizer = optim.SGD(
        model.parameters(), lr=args.lr,
        momentum=args.momentum, weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Robust loss (agent-editable)
    robust_loss = RobustLoss()

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = robust_loss.compute_loss(logits, labels, epoch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * labels.size(0)
            _, predicted = logits.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
        scheduler.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"TRAIN_METRICS epoch={epoch + 1} loss={total_loss / total:.4f} "
                f"train_acc={100.0 * correct / total:.2f} "
                f"lr={optimizer.param_groups[0]['lr']:.6f}",
                flush=True,
            )

    # Evaluation
    test_acc = evaluate_test_acc(model, test_loader, device)
    poison_fit = evaluate_poison_fit(model, poisoned_train, device, args.batch_size)
    robust_score = (test_acc + (1.0 - poison_fit)) / 2.0

    print(
        f"TEST_METRICS test_acc={test_acc:.4f} poison_fit={poison_fit:.4f} "
        f"robust_score={robust_score:.4f}",
        flush=True,
    )


if __name__ == '__main__':
    main()
