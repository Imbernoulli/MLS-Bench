"""
CIFAR-10 JEPA Self-Supervised Training Script (Self-Contained)

Trains a ResNet-18 backbone with a projector using a two-view augmentation
pipeline and an anti-collapse regularization loss. Evaluation is performed
via an online linear probe on CIFAR-10 validation set.

Usage:
    python custom_regularizer.py
"""

import sys; sys.path = [p for p in sys.path if not __import__('os').path.isfile(__import__('os').path.join(p, 'logging.py'))]
import os
import math
import time
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.amp import GradScaler, autocast
from torch.optim.optimizer import required
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10


# ── Custom Regularizer ─────────────────────────────────────────────────────
# EDITABLE REGION START
class CustomRegularizer(nn.Module):
    """Anti-collapse regularizer for self-supervised JEPA learning.

    Takes two projected embedding tensors from different augmented views
    and returns a loss dict that prevents representation collapse while
    encouraging useful feature learning.

    Args:
        z1: [B, D] projected embeddings from view 1
        z2: [B, D] projected embeddings from view 2

    Returns:
        dict with at least a "loss" key (scalar tensor)
    """

    def __init__(self):
        super().__init__()

    def forward(self, z1, z2):
        loss = torch.tensor(0.0, device=z1.device, requires_grad=True)
        return {"loss": loss}


# CONFIG_OVERRIDES: override training hyperparameters for your method.
# Allowed keys: proj_output_dim, proj_hidden_dim.
CONFIG_OVERRIDES = {}
# EDITABLE REGION END


# ── Backbone ──────────────────────────────────────────────────────────────

def build_backbone(arch="resnet18"):
    """Build a backbone modified for CIFAR-10 (small 3x3 conv1, no maxpool).

    Returns (backbone_module, features_dim).
    """
    builder = {
        "resnet18": (torchvision.models.resnet18, 512),
        "resnet34": (torchvision.models.resnet34, 512),
        "resnet50": (torchvision.models.resnet50, 2048),
    }
    if arch not in builder:
        raise ValueError(f"Unknown ARCH={arch!r}. Choose from {list(builder)}")
    fn, features_dim = builder[arch]
    model = fn()
    model.fc = nn.Identity()
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2, bias=False)
    model.maxpool = nn.Identity()
    return model, features_dim


# ── ImageSSL Model ──────────────────────────────────────────────────────────

class ImageSSL(nn.Module):
    """Image Self-Supervised Learning model with backbone + projector."""

    def __init__(
        self, backbone, features_dim, proj_hidden_dim=2048, proj_output_dim=2048
    ):
        super().__init__()
        self.backbone = backbone
        self.features_dim = features_dim
        self.projector = nn.Sequential(
            nn.Linear(features_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_output_dim),
        )

    def forward(self, x):
        features = self.backbone(x)
        projections = self.projector(features)
        return features, projections


# ── Linear Probe ────────────────────────────────────────────────────────────

class LinearProbe(nn.Module):
    """Linear probe classifier for evaluating representations."""

    def __init__(self, feature_dim, num_classes):
        super().__init__()
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        return self.classifier(x)


# ── LARS Optimizer ──────────────────────────────────────────────────────────

class LARS(optim.Optimizer):
    """LARS (Layer-wise Adaptive Rate Scaling) optimizer."""

    def __init__(
        self,
        params,
        lr=required,
        momentum=0,
        dampening=0,
        weight_decay=0,
        nesterov=False,
        eta=1e-3,
        eps=1e-8,
        clip_lr=False,
        exclude_bias_n_norm=False,
    ):
        if lr is not required and lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
            eta=eta,
            eps=eps,
            clip_lr=clip_lr,
            exclude_bias_n_norm=exclude_bias_n_norm,
        )
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            dampening = group["dampening"]
            nesterov = group["nesterov"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                d_p = p.grad
                p_norm = torch.norm(p.data)
                g_norm = torch.norm(p.grad.data)

                if p.ndim != 1 or not group["exclude_bias_n_norm"]:
                    if p_norm != 0 and g_norm != 0:
                        lars_lr = p_norm / (
                            g_norm + p_norm * weight_decay + group["eps"]
                        )
                        lars_lr *= group["eta"]

                        if group["clip_lr"]:
                            lars_lr = min(lars_lr / group["lr"], 1)

                        d_p = d_p.add(p, alpha=weight_decay)
                        d_p *= lars_lr

                if momentum != 0:
                    param_state = self.state[p]
                    if "momentum_buffer" not in param_state:
                        buf = param_state["momentum_buffer"] = torch.clone(
                            d_p
                        ).detach()
                    else:
                        buf = param_state["momentum_buffer"]
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf

                p.add_(d_p, alpha=-group["lr"])

        return loss


# ── Warmup Cosine Scheduler ────────────────────────────────────────────────

class WarmupCosineScheduler:
    """Warmup cosine learning rate scheduler."""

    def __init__(
        self,
        optimizer,
        warmup_epochs,
        max_epochs,
        base_lr,
        min_lr=0.0,
        warmup_start_lr=3e-5,
    ):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.warmup_start_lr = warmup_start_lr

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            lr = self.warmup_start_lr + epoch * (
                self.base_lr - self.warmup_start_lr
            ) / max(self.warmup_epochs - 1, 1)
        else:
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (
                1
                + math.cos(
                    (epoch - self.warmup_epochs)
                    / max(self.max_epochs - self.warmup_epochs, 1)
                    * math.pi
                )
            )

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr


# ── Data Augmentations ──────────────────────────────────────────────────────

def get_train_transforms():
    """Get training transforms for self-supervised learning on CIFAR-10."""
    transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                    )
                ],
                p=0.8,
            ),
            transforms.RandomGrayscale(p=0.2),
            # Solarization at p=0.1 — paper's upstream pipeline uses this and
            # its absence disproportionately hurts SIGReg (Gaussianity-on-
            # random-projections benefits from augmentation diversity). See
            # eb_jepa/examples/image_jepa/dataset.py:46-79.
            transforms.RandomSolarize(threshold=128, p=0.1),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
            ),
        ]
    )
    return transform


def get_val_transforms():
    """Get validation transforms."""
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
            ),
        ]
    )


class ImageDataset(Dataset):
    """Dataset that applies augmentations multiple times to create views."""

    def __init__(self, dataset, transform, num_crops=2):
        self.dataset = dataset
        self.transform = transform
        self.num_crops = num_crops

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        views = [self.transform(image) for _ in range(self.num_crops)]
        return views, label


# ── Evaluation ──────────────────────────────────────────────────────────────

def evaluate_linear_probe(model, linear_probe, val_loader, device, use_amp=True):
    """Evaluate linear probe on validation set."""
    model.eval()
    linear_probe.eval()

    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in val_loader:
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            with autocast("cuda", enabled=use_amp):
                features, _ = model(data)

            outputs = linear_probe(features.float())
            loss = F.cross_entropy(outputs, target)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    accuracy = 100.0 * correct / total
    avg_loss = total_loss / len(val_loader)
    return accuracy, avg_loss


# ── Training Loop ───────────────────────────────────────────────────────────

def train_epoch(
    model,
    train_loader,
    optimizer,
    scheduler,
    linear_probe,
    scaler,
    device,
    epoch,
    loss_fn,
    use_amp=True,
    dtype=torch.bfloat16,
):
    """Train for one epoch."""
    model.train()
    linear_probe.train()

    loss_totals = {}
    total_linear_loss = 0
    linear_correct = 0
    linear_total = 0

    for batch_idx, (views, target) in enumerate(train_loader):
        view1, view2 = views[0].to(device, non_blocking=True), views[1].to(
            device, non_blocking=True
        )
        target = target.to(device, non_blocking=True)

        with autocast(device.type, enabled=use_amp, dtype=dtype):
            features, z1 = model(view1)
            _, z2 = model(view2)
            loss_dict = loss_fn(z1, z2)
            loss = loss_dict["loss"]

        with torch.no_grad():
            features_frozen = features.detach().float()

        linear_outputs = linear_probe(features_frozen)
        linear_loss = F.cross_entropy(linear_outputs, target)

        _, predicted = linear_outputs.max(1)
        linear_correct_batch = predicted.eq(target).sum().item()

        total_loss_batch = loss + linear_loss

        optimizer.zero_grad()
        scaler.scale(total_loss_batch).backward()
        scaler.step(optimizer)
        scaler.update()

        for key, value in loss_dict.items():
            if key not in loss_totals:
                loss_totals[key] = 0
            loss_totals[key] += value.item() if torch.is_tensor(value) else value
        total_linear_loss += linear_loss.item()

        linear_total += target.size(0)
        linear_correct += linear_correct_batch

    scheduler.step(epoch)

    num_batches = len(train_loader)
    metrics = {key: total / num_batches for key, total in loss_totals.items()}
    metrics["linear_loss"] = total_linear_loss / num_batches
    metrics["linear_acc"] = 100.0 * linear_correct / linear_total

    return metrics


# ── Main ────────────────────────────────────────────────────────────────────

def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def make_generator(seed):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def main():
    # Configuration
    seed = int(os.environ.get("SEED", 42))
    arch = os.environ.get("ARCH", "resnet18")
    data_dir = os.environ.get("EBJEPA_DSETS", "/data/eb_jepa")
    # Match the upstream image_jepa SIGReg/VICReg comparisons, which train
    # for 300 epochs on CIFAR-10. At 100 epochs both baselines under-converge
    # and the expected SIGReg/VICReg ordering is not reliable.
    epochs = 300
    batch_size = 256
    lr = 0.3
    weight_decay = 1e-4
    warmup_epochs = 10
    min_lr = 0.0
    warmup_start_lr = 3e-5
    num_workers = 4
    proj_hidden_dim = 2048
    proj_output_dim = 2048
    use_amp = True
    dtype = torch.bfloat16
    log_every = 10

    # Setup
    seed_everything(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    # Data
    base_train_dataset = CIFAR10(
        root=data_dir, train=True, download=False, transform=None
    )
    train_dataset = ImageDataset(base_train_dataset, get_train_transforms(), num_crops=2)
    val_dataset = CIFAR10(
        root=data_dir, train=False, download=False, transform=get_val_transforms()
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=make_generator(seed),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=make_generator(seed + 1),
    )

    print(
        f"Data: CIFAR-10 | train={len(train_dataset)} | val={len(val_dataset)} "
        f"| batch_size={batch_size}",
        flush=True,
    )

    # Apply per-method hyperparameter overrides (CONFIG_OVERRIDES is in the
    # editable region; this loop is fixed infrastructure — agents cannot add
    # new keys, only set values for the whitelist below).
    for _k, _v in CONFIG_OVERRIDES.items():
        if _k == "proj_output_dim": proj_output_dim = _v
        elif _k == "proj_hidden_dim": proj_hidden_dim = _v

    # Model
    backbone, features_dim = build_backbone(arch)
    model = ImageSSL(
        backbone,
        features_dim=features_dim,
        proj_hidden_dim=proj_hidden_dim,
        proj_output_dim=proj_output_dim,
    ).to(device)

    linear_probe = LinearProbe(feature_dim=features_dim, num_classes=10).to(device)

    encoder_params = sum(p.numel() for p in backbone.parameters())
    projector_params = sum(p.numel() for p in model.projector.parameters())
    print(
        f"Model: {arch} | encoder={encoder_params:,} "
        f"| projector={projector_params:,} ({proj_hidden_dim}->{proj_output_dim})",
        flush=True,
    )

    # Optimizer
    scaler = GradScaler(device.type, enabled=use_amp)
    optimizer = LARS(
        [
            {"params": model.parameters(), "lr": lr},
            {"params": linear_probe.parameters(), "lr": 0.1},
        ],
        weight_decay=weight_decay,
        eta=0.02,
        clip_lr=True,
        exclude_bias_n_norm=True,
        momentum=0.9,
    )
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=warmup_epochs,
        max_epochs=epochs,
        base_lr=lr,
        min_lr=min_lr,
        warmup_start_lr=warmup_start_lr,
    )

    # Loss
    loss_fn = CustomRegularizer().to(device)

    # Training
    print(f"Starting training for {epochs} epochs...", flush=True)
    start_time = time.time()

    for epoch in range(epochs):
        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            linear_probe,
            scaler,
            device,
            epoch,
            loss_fn,
            use_amp,
            dtype,
        )

        val_acc, val_loss = evaluate_linear_probe(
            model, linear_probe, val_loader, device, use_amp
        )

        if epoch % log_every == 0 or epoch == epochs - 1:
            elapsed = time.time() - start_time
            metrics_str = " | ".join(
                f"{k}={v:.4f}" for k, v in train_metrics.items()
            )
            print(
                f"TRAIN_METRICS: epoch={epoch} | {metrics_str} "
                f"| val_acc={val_acc:.2f} | val_loss={val_loss:.4f} "
                f"| time={elapsed:.1f}s",
                flush=True,
            )

    # Final evaluation
    val_acc, val_loss = evaluate_linear_probe(
        model, linear_probe, val_loader, device, use_amp
    )
    print(f"TEST_METRICS: val_acc={val_acc:.2f}", flush=True)
    print(f"Training completed. Final val_acc={val_acc:.2f}%", flush=True)


if __name__ == "__main__":
    main()
