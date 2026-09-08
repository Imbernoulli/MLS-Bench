"""Fixed evaluation harness for security-membership-inference-defense.

Train vision models (ResNet, VGG, MobileNetV2) on CIFAR-10/100/FashionMNIST
to evaluate custom membership-inference defense losses.

FIXED: Model architectures, data pipeline, training loop, MIA evaluation.
EDITABLE: MembershipDefense.compute_loss() method.

Usage:
    python run_membership_defense.py --arch resnet20 --dataset cifar10 --seed 42
    python run_membership_defense.py --arch mobilenetv2 --dataset fmnist --seed 42
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

from custom_membership_defense import MembershipDefense

_DATA_ROOT = os.environ.get("DATA_ROOT", "/data")


# ============================================================================
# Model Architectures (FIXED) — same as dl-activation-function, using ReLU
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

def get_dataset(dataset, data_root, augment=False):
    """Load full train/test datasets.

    Training augmentation is OFF by default to match the RelaxLoss
    paper recipe (Chen et al., ICLR 2022 — see official config
    https://github.com/DingfanChen/RelaxLoss/blob/main/source/cifar/defense/configs/default.yml
    which sets ``if_data_augmentation: False``). Augmentation prevents
    the overfit required to exhibit measurable MIA leakage (paper ERM
    VGG-11 on CIFAR-100 yields MIA AUC ~0.96).
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

    train_transform_list = [transforms.Resize(32)]
    if augment:
        train_transform_list += [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ]
    train_transform_list.append(transforms.ToTensor())
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

    train_set = Dataset(root=data_root, train=True, download=False, transform=train_transform)
    test_set = Dataset(root=data_root, train=False, download=False, transform=test_transform)

    return train_set, test_set, num_classes


# ============================================================================
# MIA Evaluation (FIXED)
# ============================================================================

def auc_from_scores(member_scores, nonmember_scores):
    """Compute AUC via Mann-Whitney U statistic (no sklearn needed)."""
    scores = np.concatenate([member_scores, nonmember_scores])
    labels = np.concatenate([np.ones(len(member_scores)), np.zeros(len(nonmember_scores))])
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    pos_ranks = ranks[labels == 1].sum()
    n_pos = len(member_scores)
    n_neg = len(nonmember_scores)
    return float((pos_ranks - n_pos * (n_pos + 1) / 2.0) / max(n_pos * n_neg, 1))


@torch.no_grad()
def confidence_scores(model, loader, device):
    """Compute max softmax probability (membership signal) and accuracy."""
    scores = []
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        probs = torch.softmax(model(images), dim=1)
        scores.append(probs.max(dim=1).values.cpu().numpy())
        preds = probs.argmax(dim=1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)
    return np.concatenate(scores), correct / max(total, 1)


# ============================================================================
# Main Pipeline (FIXED)
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Membership Inference Defense Benchmark")
    parser.add_argument('--arch', type=str, required=True,
                        choices=['resnet20', 'vgg16bn', 'mobilenetv2'])
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['cifar10', 'cifar100', 'fmnist'])
    parser.add_argument('--data-root', type=str, default='/data/cifar')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--schedule-milestones', type=int, nargs='+',
                        default=[150, 225],
                        help='Step-LR decay milestones (paper: 150,225 for 300-epoch run).')
    parser.add_argument('--schedule-gamma', type=float, default=0.1)
    parser.add_argument('--augment', action='store_true',
                        help='Enable train-time augmentation (OFF by default to match paper).')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # Seed everything
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── 1. Load full dataset, split into train / non-train (50/50 by index) ──
    train_set, test_set, num_classes = get_dataset(
        args.dataset, args.data_root, augment=args.augment,
    )
    n_train = len(train_set)
    half = n_train // 2
    indices = list(range(n_train))
    rng = random.Random(args.seed)
    rng.shuffle(indices)
    member_idx = indices[:half]
    nonmember_idx = indices[half:]

    member_subset = torch.utils.data.Subset(train_set, member_idx)
    nonmember_subset = torch.utils.data.Subset(train_set, nonmember_idx)

    train_loader = torch.utils.data.DataLoader(
        member_subset, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True,
    )
    member_eval_loader = torch.utils.data.DataLoader(
        member_subset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True,
    )
    nonmember_eval_loader = torch.utils.data.DataLoader(
        nonmember_subset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    # ── 2. Build model ──
    model = build_model(args.arch, num_classes)
    initialize_weights(model)
    model = model.to(device)

    # ── 3. Train with custom defense loss ──
    defense = MembershipDefense()
    optimizer = optim.SGD(
        model.parameters(), lr=args.lr,
        momentum=args.momentum, weight_decay=args.weight_decay,
    )
    # Paper uses step-decay MultiStepLR (milestones [150, 225], gamma 0.1) — matches
    # official RelaxLoss default.yml
    # https://github.com/DingfanChen/RelaxLoss/blob/main/source/cifar/defense/configs/default.yml
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=args.schedule_milestones, gamma=args.schedule_gamma,
    )

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = defense.compute_loss(logits, targets, epoch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * targets.size(0)
            _, predicted = logits.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)
        scheduler.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"TRAIN_METRICS epoch={epoch + 1} "
                f"loss={total_loss / total:.4f} "
                f"train_acc={100.0 * correct / total:.2f} "
                f"lr={optimizer.param_groups[0]['lr']:.6f}",
                flush=True,
            )

    # ── 4. Evaluate: test accuracy ──
    model.eval()
    _, test_acc = confidence_scores(model, test_loader, device)

    # ── 5. MIA: confidence-based membership inference ──
    member_scores, _ = confidence_scores(model, member_eval_loader, device)
    nonmember_scores, _ = confidence_scores(model, nonmember_eval_loader, device)
    mia_auc = auc_from_scores(member_scores, nonmember_scores)

    # ── 6. Compute privacy metrics ──
    privacy_gap = float(member_scores.mean() - nonmember_scores.mean())
    privacy_score = test_acc - max(mia_auc - 0.5, 0.0)

    print(
        f"TEST_METRICS test_acc={test_acc:.4f} mia_auc={mia_auc:.4f} "
        f"privacy_gap={privacy_gap:.4f} privacy_score={privacy_score:.4f}",
        flush=True,
    )


if __name__ == '__main__':
    main()
