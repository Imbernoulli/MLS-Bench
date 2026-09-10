"""Fixed evaluation harness for security-backdoor-defense.

Train research-scale vision models (ResNet-20, VGG-16-BN, MobileNetV2) on
CIFAR-10/100/FashionMNIST with backdoor poisoning, then evaluate a custom
backdoor defense that scores and removes suspicious training samples.

FIXED: Model architectures, data pipeline, training loop, poison injection.
EDITABLE: BackdoorDefense class in custom_backdoor_defense.py.

Usage:
    python run_backdoor_defense.py --arch resnet20 --dataset cifar10 \
        --data-root /data/cifar --trigger badnets --poison-fraction 0.05 \
        --epochs 100 --seed 42
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
from torch.utils.data import DataLoader, Dataset

from custom_backdoor_defense import BackdoorDefense

_DATA_ROOT = os.environ.get("DATA_ROOT", "/data")


# ============================================================================
# Model Architectures (FIXED) — same as dl-activation-function, using nn.ReLU
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
    Standard depths: ResNet-20 ([3,3,3]).
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

    def forward_features(self, x):
        """Return penultimate layer features (before final FC)."""
        out = self.act(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.adaptive_avg_pool2d(out, 1)
        return out.view(out.size(0), -1)


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

    def forward_features(self, x):
        """Return features after conv layers + pool, before classifier."""
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, 1)
        return x.view(x.size(0), -1)


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
                nn.ReLU6(),
            ]
        layers += [
            nn.Conv2d(hidden, hidden, 3, stride=stride, padding=1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU6(),
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
            nn.ReLU6(),
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
            nn.ReLU6(),
        )
        self.fc = nn.Linear(1280, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layers(x)
        x = self.conv_last(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        return self.fc(x)

    def forward_features(self, x):
        """Return features after final conv+pool, before FC."""
        x = self.conv1(x)
        x = self.layers(x)
        x = self.conv_last(x)
        x = F.adaptive_avg_pool2d(x, 1)
        return x.view(x.size(0), -1)


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
# Data Loading (FIXED) — same transforms as dl-activation-function
# ============================================================================

def get_dataloaders(dataset, data_root, batch_size=128, num_workers=4):
    """Create train/test dataloaders with standard augmentation."""
    if dataset == 'cifar10':
        mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        num_classes = 10
        DatasetCls = torchvision.datasets.CIFAR10
    elif dataset == 'cifar100':
        mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
        num_classes = 100
        DatasetCls = torchvision.datasets.CIFAR100
    elif dataset == 'fmnist':
        mean, std = (0.2860, 0.2860, 0.2860), (0.3530, 0.3530, 0.3530)
        num_classes = 10
        DatasetCls = torchvision.datasets.FashionMNIST
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

    train_set = DatasetCls(root=data_root, train=True, download=False, transform=train_transform)
    test_set = DatasetCls(root=data_root, train=False, download=False, transform=test_transform)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_set, test_set, train_loader, test_loader, num_classes


# ============================================================================
# Backdoor Trigger Injection (FIXED)
# ============================================================================

class PoisonedDataset(Dataset):
    """Wraps a dataset with backdoor trigger injection on a subset of samples."""

    def __init__(self, base_dataset, poison_indices, target_class, trigger,
                 mean, std):
        self.base_dataset = base_dataset
        self.poison_set = set(poison_indices)
        self.target_class = target_class
        self.trigger = trigger
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)
        self.is_poison = [i in self.poison_set for i in range(len(base_dataset))]

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image, label = self.base_dataset[idx]
        if idx in self.poison_set:
            image = self._apply_trigger(image)
            label = self.target_class
        return image, label

    def _apply_trigger(self, image):
        """Apply trigger in normalized image space."""
        # Denormalize
        img = image * self.std + self.mean
        if self.trigger == 'badnets':
            # White 3x3 square in bottom-right corner
            img[:, -3:, -3:] = 1.0
        else:  # blend
            # 50% checkerboard overlay
            pattern = torch.zeros_like(img)
            pattern[:, ::2, ::2] = 1.0
            pattern[:, 1::2, 1::2] = 1.0
            img = 0.8 * img + 0.2 * pattern
        img = img.clamp(0.0, 1.0)
        # Re-normalize
        return (img - self.mean) / self.std


def select_poison_indices(dataset, poison_fraction, target_class):
    """Select indices of non-target-class samples to poison."""
    eligible = []
    for i in range(len(dataset)):
        _, label = dataset[i]
        if label != target_class:
            eligible.append(i)
    n_poison = max(1, int(len(dataset) * poison_fraction))
    random.shuffle(eligible)
    return eligible[:n_poison]


# ============================================================================
# Training Loop (FIXED) — SGD + CosineAnnealing
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
    """Evaluate on a loader. Returns (avg_loss, accuracy%)."""
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


def train_model(model, train_loader, device, epochs, lr, momentum, weight_decay,
                tag, test_loader=None):
    """Full training loop with SGD + CosineAnnealingLR."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), lr=lr,
        momentum=momentum, weight_decay=weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            msg = (
                f"TRAIN_METRICS: phase={tag} epoch={epoch+1} "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.2f} "
                f"lr={optimizer.param_groups[0]['lr']:.6f}"
            )
            if test_loader is not None:
                _, test_acc = evaluate(model, test_loader, criterion, device)
                msg += f" test_acc={test_acc:.2f}"
            print(msg, flush=True)


# ============================================================================
# Feature Extraction (FIXED)
# ============================================================================

@torch.no_grad()
def collect_features_logits(model, loader, device):
    """Extract penultimate-layer features and logits for all samples."""
    model.eval()
    all_features, all_logits, all_labels = [], [], []
    for inputs, targets in loader:
        inputs = inputs.to(device)
        feat = model.forward_features(inputs)
        logit = model.fc(feat) if hasattr(model, 'fc') else model.classifier(feat)
        all_features.append(feat.cpu().numpy())
        all_logits.append(logit.cpu().numpy())
        all_labels.append(targets.numpy())
    return np.concatenate(all_features), np.concatenate(all_logits), np.concatenate(all_labels)


# ============================================================================
# ASR Evaluation (FIXED)
# ============================================================================

@torch.no_grad()
def evaluate_asr(model, test_set, device, batch_size, target_class, trigger,
                 mean, std):
    """Compute attack success rate on triggered test inputs."""
    model.eval()
    mean_t = torch.tensor(mean).view(3, 1, 1)
    std_t = torch.tensor(std).view(3, 1, 1)

    triggered_images = []
    for i in range(len(test_set)):
        image, label = test_set[i]
        if label == target_class:
            continue
        # Denormalize, apply trigger, re-normalize
        img = image * std_t + mean_t
        if trigger == 'badnets':
            img[:, -3:, -3:] = 1.0
        else:
            pattern = torch.zeros_like(img)
            pattern[:, ::2, ::2] = 1.0
            pattern[:, 1::2, 1::2] = 1.0
            img = 0.8 * img + 0.2 * pattern
        img = img.clamp(0.0, 1.0)
        triggered_images.append((img - mean_t) / std_t)

    total, success = 0, 0
    for start in range(0, len(triggered_images), batch_size):
        batch = torch.stack(triggered_images[start:start + batch_size]).to(device)
        preds = model(batch).argmax(dim=1).cpu()
        total += preds.size(0)
        success += preds.eq(target_class).sum().item()
    return success / max(total, 1)


# ============================================================================
# Main Pipeline
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Backdoor Defense Benchmark")
    parser.add_argument('--arch', type=str, required=True,
                        choices=['resnet20', 'vgg16bn', 'mobilenetv2'])
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['cifar10', 'cifar100', 'fmnist'])
    parser.add_argument('--data-root', type=str, default=_DATA_ROOT)
    parser.add_argument('--trigger', type=str, required=True,
                        choices=['badnets', 'blend'])
    parser.add_argument('--poison-fraction', type=float, required=True)
    parser.add_argument('--target-class', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()

    # Seed everything
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Dataset normalization stats (must match data loading)
    if args.dataset == 'cifar10':
        mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    elif args.dataset == 'cifar100':
        mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
    else:  # fmnist
        mean, std = (0.2860, 0.2860, 0.2860), (0.3530, 0.3530, 0.3530)

    # Load full datasets
    train_set, test_set, _, test_loader, num_classes = get_dataloaders(
        args.dataset, args.data_root, args.batch_size,
    )

    # ---- Step 1: Poison the training set ----
    poison_indices = select_poison_indices(
        train_set, args.poison_fraction, args.target_class,
    )
    poisoned_train = PoisonedDataset(
        train_set, poison_indices, args.target_class, args.trigger, mean, std,
    )
    poisoned_loader = DataLoader(
        poisoned_train, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True,
    )
    n_poison = len(poison_indices)
    n_total = len(train_set)
    print(f"Poisoned {n_poison}/{n_total} training samples "
          f"({100*n_poison/n_total:.1f}%), trigger={args.trigger}, "
          f"target_class={args.target_class}", flush=True)

    # ---- Step 2: Train victim model on poisoned data (100 epochs) ----
    model = build_model(args.arch, num_classes)
    initialize_weights(model)
    model = model.to(device)
    train_model(
        model, poisoned_loader, device, args.epochs,
        args.lr, args.momentum, args.weight_decay,
        tag="poisoned_train", test_loader=test_loader,
    )

    # ---- Step 3: Extract features for defense analysis ----
    analysis_loader = DataLoader(
        poisoned_train, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True,
    )
    features, logits, labels = collect_features_logits(model, analysis_loader, device)

    # ---- Step 4: Run defense scoring ----
    defense = BackdoorDefense()
    defense.fit(features, labels, args.poison_fraction)
    scores = np.asarray(defense.score_samples(features, logits), dtype=np.float64)

    # ---- Step 5: Remove highest-scoring samples ----
    # Tran et al. (2018, Sec. 4.1) recommend removing 1.5 * epsilon * n points
    # (an over-estimate of the poison count) so that the filter has margin for
    # scoring errors.  Matches BackdoorBench's protocol of filter-then-retrain.
    filter_multiplier = 1.5
    n_remove = max(1, int(len(scores) * args.poison_fraction * filter_multiplier))
    n_remove = min(n_remove, len(scores) - 1)
    remove_idx = np.argsort(scores)[-n_remove:]
    is_poison_arr = np.array(poisoned_train.is_poison, dtype=bool)
    removed_poison = float(is_poison_arr[remove_idx].sum())
    poison_recall = removed_poison / max(float(is_poison_arr.sum()), 1.0)
    print(f"Defense removed {int(removed_poison)}/{int(is_poison_arr.sum())} "
          f"poison samples out of {n_remove} flagged "
          f"(recall={poison_recall:.4f}, filter_ratio={filter_multiplier}*eps)",
          flush=True)

    # Build filtered dataset (keep non-removed indices)
    keep_mask = np.ones(len(scores), dtype=bool)
    keep_mask[remove_idx] = False
    keep_indices = np.nonzero(keep_mask)[0].tolist()

    class FilteredDataset(Dataset):
        def __init__(self, base_dataset, indices):
            self.base_dataset = base_dataset
            self.indices = indices

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, idx):
            return self.base_dataset[self.indices[idx]]

    filtered_dataset = FilteredDataset(poisoned_train, keep_indices)
    filtered_loader = DataLoader(
        filtered_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True,
    )

    # ---- Step 6: Retrain defended model on filtered data (100 epochs) ----
    defended_model = build_model(args.arch, num_classes)
    initialize_weights(defended_model)
    defended_model = defended_model.to(device)
    train_model(
        defended_model, filtered_loader, device, args.epochs,
        args.lr, args.momentum, args.weight_decay,
        tag="defense_retrain", test_loader=test_loader,
    )

    # ---- Step 7: Final evaluation ----
    criterion = nn.CrossEntropyLoss()
    _, clean_acc_pct = evaluate(defended_model, test_loader, criterion, device)
    clean_acc = clean_acc_pct / 100.0
    asr = evaluate_asr(
        defended_model, test_set, device, args.batch_size,
        args.target_class, args.trigger, mean, std,
    )
    # defense_score follows BackdoorBench's post-filter-retrain convention:
    # combine clean accuracy on the retrained model with robustness against the
    # trigger (1 - ASR).  Filter-stage poison_recall is reported separately for
    # diagnostics but is NOT part of defense_score.  This aligns with Tran et
    # al. (2018) and BackdoorBench/defense/spectral.py, where only retrained
    # model ACC/ASR determine effectiveness.
    defense_score = 0.5 * clean_acc + 0.5 * (1.0 - asr)

    print(
        f"TEST_METRICS clean_acc={clean_acc:.4f} asr={asr:.4f} "
        f"poison_recall={poison_recall:.4f} defense_score={defense_score:.4f}",
        flush=True,
    )


if __name__ == '__main__':
    main()
