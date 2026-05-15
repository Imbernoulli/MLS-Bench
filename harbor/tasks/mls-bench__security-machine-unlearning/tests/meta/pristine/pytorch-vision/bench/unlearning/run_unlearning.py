"""Fixed evaluation harness for security-machine-unlearning.

Pipeline:
  1. Load full dataset with standard augmentation
  2. Split into retain set (all classes except forget_class) and forget set
  3. Pretrain model on FULL training set for --pretrain-epochs (SGD + CosineAnnealing)
  4. Run unlearning: agent method processes retain/forget batches for --unlearn-epochs
  5. Evaluate: retain_acc, forget_acc, forget_mia_auc
  6. Compute unlearn_score = (retain_acc + (1-forget_acc) + (1-forget_mia_auc)) / 3
"""

import argparse
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

from custom_unlearning import UnlearningMethod

_DATA_ROOT = os.environ.get("DATA_ROOT", "/data")


# ============================================================================
# Model Architectures (FIXED) -- same as dl-activation-function but with ReLU
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
        self.act = nn.ReLU(inplace=True)
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
        self.act = nn.ReLU(inplace=True)
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
            nn.ReLU(inplace=True),
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
                nn.ReLU(inplace=True),
            ]
        layers += [
            nn.Conv2d(hidden, hidden, 3, stride=stride, padding=1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
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
            nn.ReLU(inplace=True),
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
            nn.ReLU(inplace=True),
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
    """Load full train/test datasets with standard augmentation."""
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

    train_set = Dataset(root=data_root, train=True, download=False, transform=train_transform)
    test_set = Dataset(root=data_root, train=False, download=False, transform=test_transform)

    return train_set, test_set, num_classes


# ============================================================================
# Splitting helpers
# ============================================================================

def split_by_class(dataset, forget_class):
    """Split dataset into retain (all except forget_class) and forget (forget_class only)."""
    # Use .targets attribute for efficiency (avoids loading every image)
    if hasattr(dataset, 'targets'):
        targets = dataset.targets
        if isinstance(targets, torch.Tensor):
            targets = targets.tolist()
    else:
        targets = [int(dataset[i][1]) for i in range(len(dataset))]
    retain_idx = [i for i, t in enumerate(targets) if int(t) != forget_class]
    forget_idx = [i for i, t in enumerate(targets) if int(t) == forget_class]
    return Subset(dataset, retain_idx), Subset(dataset, forget_idx)


def cycle_loader(loader):
    """Infinitely cycle through a DataLoader."""
    while True:
        for batch in loader:
            yield batch


# ============================================================================
# Pretrain (FIXED)
# ============================================================================

def pretrain(model, loader, device, epochs, lr, momentum, weight_decay):
    """Pretrain model on full training set with SGD + CosineAnnealingLR."""
    optimizer = optim.SGD(
        model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * labels.size(0)
            correct += logits.argmax(1).eq(labels).sum().item()
            total += labels.size(0)
        scheduler.step()
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"TRAIN_METRICS phase=pretrain epoch={epoch + 1} "
                f"loss={total_loss / total:.4f} acc={100.0 * correct / total:.2f}",
                flush=True,
            )


# ============================================================================
# Evaluation helpers
# ============================================================================

@torch.no_grad()
def evaluate_accuracy(model, loader, device):
    """Return (accuracy, confidence_scores) on a DataLoader."""
    model.eval()
    correct, total = 0, 0
    all_confs = []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        probs = torch.softmax(model(images), dim=1)
        preds = probs.argmax(dim=1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)
        all_confs.append(probs.max(dim=1).values.cpu().numpy())
    acc = correct / max(total, 1)
    confs = np.concatenate(all_confs) if all_confs else np.array([])
    return acc, confs


def auc_from_scores(member_scores, nonmember_scores):
    """Compute AUC via Mann-Whitney U statistic (no scipy needed)."""
    scores = np.concatenate([member_scores, nonmember_scores])
    labels = np.concatenate([np.ones(len(member_scores)), np.zeros(len(nonmember_scores))])
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    pos_ranks = ranks[labels == 1].sum()
    n_pos = len(member_scores)
    n_neg = len(nonmember_scores)
    return float((pos_ranks - n_pos * (n_pos + 1) / 2.0) / max(n_pos * n_neg, 1))


# ============================================================================
# Main
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Machine Unlearning Benchmark")
    parser.add_argument('--arch', type=str, required=True,
                        choices=['resnet20', 'resnet56', 'vgg16bn', 'mobilenetv2'])
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['cifar10', 'cifar100', 'fmnist'])
    parser.add_argument('--data-root', type=str, default='/data/cifar')
    parser.add_argument('--forget-class', type=int, default=0)
    parser.add_argument('--pretrain-epochs', type=int, default=80)
    parser.add_argument('--unlearn-epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Data ----
    train_set, test_set, num_classes = get_datasets(args.dataset, args.data_root)

    # Full training loader (pretrain on ALL classes)
    full_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True,
    )

    # Split train and test by forget class
    retain_train, forget_train = split_by_class(train_set, args.forget_class)
    retain_test, forget_test = split_by_class(test_set, args.forget_class)

    retain_loader = DataLoader(
        retain_train, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True,
    )
    forget_loader = DataLoader(
        forget_train, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True,
    )

    # ---- Build & pretrain model ----
    model = build_model(args.arch, num_classes)
    initialize_weights(model)
    model = model.to(device)

    print(f"Pretraining {args.arch} on {args.dataset} for {args.pretrain_epochs} epochs...", flush=True)
    pretrain(model, full_loader, device, args.pretrain_epochs,
             args.lr, args.momentum, args.weight_decay)

    # ---- Unlearning ----
    method = UnlearningMethod()
    unlearn_optimizer = optim.Adam(model.parameters(), lr=0.001)
    retain_iter = cycle_loader(retain_loader)
    forget_iter = cycle_loader(forget_loader)
    steps_per_epoch = max(1, min(len(retain_loader), len(forget_loader)))

    print(f"Running unlearning for {args.unlearn_epochs} epochs...", flush=True)
    for epoch in range(args.unlearn_epochs):
        model.train()
        losses = []
        for step in range(steps_per_epoch):
            retain_batch = next(retain_iter)
            forget_batch = next(forget_iter)
            retain_batch = (retain_batch[0].to(device), retain_batch[1].to(device))
            forget_batch = (forget_batch[0].to(device), forget_batch[1].to(device))
            result = method.unlearn_step(
                model, retain_batch, forget_batch, unlearn_optimizer, step, epoch,
            )
            losses.append(float(result["loss"]))
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"TRAIN_METRICS phase=unlearn epoch={epoch + 1} loss={np.mean(losses):.4f}",
                flush=True,
            )

    # ---- Evaluation ----
    retain_acc, _ = evaluate_accuracy(
        model, DataLoader(retain_test, batch_size=args.batch_size, num_workers=4), device,
    )
    forget_acc, forget_test_scores = evaluate_accuracy(
        model, DataLoader(forget_test, batch_size=args.batch_size, num_workers=4), device,
    )
    _, forget_train_scores = evaluate_accuracy(
        model, DataLoader(forget_train, batch_size=args.batch_size, num_workers=4), device,
    )

    # MIA: members = forget train (seen during original training), non-members = forget test
    forget_mia_auc = auc_from_scores(forget_train_scores, forget_test_scores)
    unlearn_score = (retain_acc + (1.0 - forget_acc) + (1.0 - forget_mia_auc)) / 3.0

    print(
        f"TEST_METRICS retain_acc={retain_acc:.4f} forget_acc={forget_acc:.4f} "
        f"forget_mia_auc={forget_mia_auc:.4f} unlearn_score={unlearn_score:.4f}",
        flush=True,
    )


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    main()
