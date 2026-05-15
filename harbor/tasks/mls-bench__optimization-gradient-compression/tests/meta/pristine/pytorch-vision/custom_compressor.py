"""Gradient Compression for Communication-Efficient Distributed Training.

Self-contained benchmark: trains standard vision models on CIFAR datasets
using data-parallel SGD with a pluggable gradient compressor.

The script simulates distributed training on a single node by:
1. Computing gradients normally
2. Applying compress() -> decompress() to each gradient (simulating communication)
3. Using the decompressed gradient for the optimizer step

This faithfully measures the effect of gradient compression on convergence
quality, which is the core ML-science question, without requiring multi-node
infrastructure.
"""

import argparse
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ============================================================================
# Model Definitions (FIXED)
# ============================================================================


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()
        self.in_planes = 16
        self.conv1 = conv3x3(3, 16)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64 * block.expansion, num_classes)

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
        return self.linear(out)


class VGG(nn.Module):
    """VGG-11 with batch normalization."""

    def __init__(self, num_classes=100):
        super().__init__()
        cfg = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.extend([
                    nn.Conv2d(in_channels, v, kernel_size=3, padding=1),
                    nn.BatchNorm2d(v),
                    nn.ReLU(inplace=True),
                ])
                in_channels = v
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def build_model(model_name, num_classes, device):
    if model_name == 'resnet20':
        model = ResNet(BasicBlock, [3, 3, 3], num_classes=num_classes)
    elif model_name == 'resnet56':
        model = ResNet(BasicBlock, [9, 9, 9], num_classes=num_classes)
    elif model_name == 'vgg11':
        model = VGG(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return model.to(device)


# ============================================================================
# Data Loading (FIXED)
# ============================================================================

def get_dataloaders(dataset_name, batch_size, num_workers=2):
    if dataset_name == 'cifar10':
        mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        num_classes = 10
        Dataset = datasets.CIFAR10
    elif dataset_name == 'cifar100':
        mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
        num_classes = 100
        Dataset = datasets.CIFAR100
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

    _data_root = os.environ.get("DATA_ROOT", "/data")
    train_set = Dataset(_data_root + '/cifar', train=True, download=False,
                        transform=train_transform)
    test_set = Dataset(_data_root + '/cifar', train=False, download=False,
                       transform=test_transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader, num_classes


# ============================================================================
# EDITABLE SECTION — Gradient Compressor (lines 182-232)
# ============================================================================

class Compressor:
    """Gradient compressor base implementation.

    Interface contract:
    - compress(tensor) -> (compressed_tensors: list[Tensor], ctx: any)
        Compress a gradient tensor. Only `compressed_tensors` would be
        "communicated" in a real distributed setting. `ctx` stays local.
    - decompress(compressed_tensors, ctx) -> Tensor
        Reconstruct the gradient from compressed representation.
        Must return a tensor of the same shape as the original.
    - The compressor may maintain internal state (e.g., error feedback
        residuals) across calls for the same parameter.

    Default: identity (no compression). Replace with your method.
    """

    def __init__(self, compress_ratio=0.01):
        """Initialize the compressor.

        Args:
            compress_ratio: Target compression ratio (fraction of elements
                to keep for sparsification, or quantization level).
                0.01 = 100x compression, 0.1 = 10x compression.
        """
        self.compress_ratio = compress_ratio

    def compress(self, tensor, name):
        """Compress a gradient tensor.

        Args:
            tensor: Gradient tensor to compress (flattened or original shape).
            name: Parameter name (useful for maintaining per-parameter state).

        Returns:
            compressed_tensors: list of tensors that would be communicated.
            ctx: local context needed for decompression (not communicated).
        """
        return [tensor.clone()], tensor.shape

    def decompress(self, compressed_tensors, ctx):
        """Decompress gradients back to original shape.

        Args:
            compressed_tensors: list of tensors from compress().
            ctx: local context from compress().

        Returns:
            Decompressed gradient tensor matching original shape.
        """
        return compressed_tensors[0].view(ctx)


# ============================================================================
# FIXED SECTION — Training Loop
# ============================================================================

def cosine_lr(optimizer, epoch, total_epochs, warmup_epochs, base_lr, min_lr=0.0):
    """Cosine learning rate schedule with linear warmup."""
    if epoch < warmup_epochs:
        lr = base_lr * (epoch + 1) / (warmup_epochs + 1)
    else:
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def apply_gradient_compression(model, compressor):
    """Apply gradient compression to all model parameters.

    Simulates the compress -> communicate -> decompress pipeline of
    distributed training. In a real system, only compressed_tensors
    would be sent over the network.
    """
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        grad = param.grad.data
        compressed, ctx = compressor.compress(grad, name)
        decompressed = compressor.decompress(compressed, ctx)
        param.grad.data = decompressed


def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = F.cross_entropy(outputs, labels, reduction='sum')
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    acc = 100.0 * correct / total
    avg_loss = total_loss / total
    return acc, avg_loss


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)

    train_loader, test_loader, num_classes = get_dataloaders(
        args.dataset, args.batch_size)
    model = build_model(args.model, num_classes, device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.model}, Dataset: {args.dataset}, "
          f"Parameters: {n_params:,}, Compress ratio: {args.compress_ratio}")

    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=args.weight_decay)

    compressor = Compressor(compress_ratio=args.compress_ratio)

    best_acc = 0.0
    for epoch in range(args.epochs):
        lr = cosine_lr(optimizer, epoch, args.epochs, args.warmup_epochs,
                       args.lr, min_lr=args.lr * 0.01)
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()

            # Apply gradient compression before optimizer step
            apply_gradient_compression(model, compressor)

            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_acc = 100.0 * correct / total
        train_loss = running_loss / len(train_loader)

        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == args.epochs - 1:
            test_acc, test_loss = evaluate(model, test_loader, device)
            if test_acc > best_acc:
                best_acc = test_acc
            print(f"TRAIN_METRICS epoch={epoch+1} lr={lr:.6f} "
                  f"train_loss={train_loss:.4f} train_acc={train_acc:.2f} "
                  f"test_acc={test_acc:.2f} test_loss={test_loss:.4f}",
                  flush=True)
        else:
            print(f"TRAIN_METRICS epoch={epoch+1} lr={lr:.6f} "
                  f"train_loss={train_loss:.4f} train_acc={train_acc:.2f}",
                  flush=True)

    # Final evaluation
    test_acc, test_loss = evaluate(model, test_loader, device)
    if test_acc > best_acc:
        best_acc = test_acc
    print(f"TEST_METRICS test_acc={test_acc:.2f} best_acc={best_acc:.2f} "
          f"test_loss={test_loss:.4f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description='Gradient Compression Benchmark')
    parser.add_argument('--model', type=str, default='resnet20',
                        choices=['resnet20', 'resnet56', 'vgg11'])
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100'])
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--warmup-epochs', type=int, default=5)
    parser.add_argument('--compress-ratio', type=float, default=0.01,
                        help='Compression ratio (fraction of gradient to keep)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
