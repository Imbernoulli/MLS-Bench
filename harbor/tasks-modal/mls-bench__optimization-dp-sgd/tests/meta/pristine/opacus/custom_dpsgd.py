#!/usr/bin/env python3
"""DP-SGD benchmark for MLS-Bench: Differentially Private Stochastic Gradient Descent.

FIXED sections: model architecture, data loading, privacy accounting, evaluation loop.
EDITABLE section: DPMechanism class — gradient clipping strategy, noise calibration,
                  and per-step privacy mechanism modifications.

The agent must implement a DPMechanism that achieves better privacy-utility tradeoff
than standard DP-SGD while respecting the same total privacy budget (epsilon, delta).
"""
import argparse
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy import optimize as sp_optimize
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# =====================================================================
# FIXED: Model architectures (DO NOT MODIFY)
# =====================================================================

class MNISTNet(nn.Module):
    """Small ConvNet for MNIST / Fashion-MNIST (1-channel 28x28 images)."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 8, 2, padding=3)
        self.conv2 = nn.Conv2d(16, 32, 4, 2)
        self.fc1 = nn.Linear(32 * 4 * 4, 32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 1)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 1)
        x = x.view(-1, 32 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class CIFAR10Net(nn.Module):
    """ConvNet for CIFAR-10 (3-channel 32x32 images), using GroupNorm (DP-compatible)."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1, padding=1)
        self.gn1 = nn.GroupNorm(8, 32)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, padding=1)
        self.gn2 = nn.GroupNorm(8, 64)
        self.conv3 = nn.Conv2d(64, 64, 3, 1, padding=1)
        self.gn3 = nn.GroupNorm(8, 64)
        self.conv4 = nn.Conv2d(64, 128, 3, 1, padding=1)
        self.gn4 = nn.GroupNorm(8, 128)
        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.gn1(self.conv1(x)))
        x = F.avg_pool2d(x, 2, 2)
        x = F.relu(self.gn2(self.conv2(x)))
        x = F.avg_pool2d(x, 2, 2)
        x = F.relu(self.gn3(self.conv3(x)))
        x = F.avg_pool2d(x, 2, 2)
        x = F.relu(self.gn4(self.conv4(x)))
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# =====================================================================
# FIXED: Privacy accounting utilities (DO NOT MODIFY)
# =====================================================================

def _compute_rdp_single_epoch(q, sigma, alpha):
    """Compute RDP for a single epoch of subsampled Gaussian mechanism."""
    if sigma == 0:
        return float("inf")
    if q == 0:
        return 0.0
    if alpha == 1:
        return q * q / (2 * sigma * sigma)
    log_term = (
        math.lgamma(alpha + 1)
        - math.lgamma(alpha - 1 + 1)
        - math.lgamma(2)
        + (alpha - 1) * math.log(1 - q)
        + math.log(q * q * alpha / (2 * sigma * sigma))
    )
    # Simplified RDP bound for subsampled Gaussian
    return min(
        alpha * q * q / (2 * sigma * sigma),
        q * q * alpha / (2 * sigma * sigma) + q * q * q * alpha * (alpha - 1) / (6 * sigma * sigma),
    )


def compute_epsilon(steps, sigma, q, delta, alphas=None):
    """Compute (epsilon, best_alpha) via RDP accounting.

    Args:
        steps: number of training steps
        sigma: noise multiplier
        q: sampling probability (batch_size / dataset_size)
        delta: target delta
        alphas: list of RDP orders to try

    Returns:
        (epsilon, best_alpha)
    """
    if alphas is None:
        alphas = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))
    best_eps = float("inf")
    best_alpha = None
    for alpha in alphas:
        # RDP for subsampled Gaussian mechanism (tight bound)
        if alpha <= 1:
            continue
        rdp = steps * min(
            q * q * alpha / (2 * sigma * sigma),
            alpha * q * q / (2 * sigma * sigma),
        )
        # Convert RDP to (epsilon, delta)-DP
        eps = rdp - math.log(delta) / (alpha - 1) + math.log(1 - 1 / alpha)
        if eps < best_eps:
            best_eps = eps
            best_alpha = alpha
    return max(0, best_eps), best_alpha


def calibrate_noise_to_epsilon(target_epsilon, steps, q, delta, tol=1e-3):
    """Find the noise multiplier sigma that achieves target_epsilon.

    Uses binary search to find the right noise level.
    """
    sigma_low, sigma_high = 0.01, 100.0
    while sigma_high - sigma_low > tol:
        sigma_mid = (sigma_low + sigma_high) / 2
        eps, _ = compute_epsilon(steps, sigma_mid, q, delta)
        if eps > target_epsilon:
            sigma_low = sigma_mid
        else:
            sigma_high = sigma_mid
    return (sigma_low + sigma_high) / 2


# =====================================================================
# EDITABLE SECTION START (lines 152-233)
# =====================================================================
# DPMechanism: decides how per-sample gradients are clipped and how much
# noise each step gets. The FIXED harness (`privatize_step`, below) applies
# your per-sample scaling, checks the clipping bound, adds the Gaussian
# noise and does the privacy accounting with the values it actually used.
#
# Interface contract:
#   __init__(self, max_grad_norm, noise_multiplier, n_params, dataset_size,
#            batch_size, epochs, target_epsilon, target_delta)
#   clip(self, per_sample_grads, step, epoch) -> (scale, clip_norm)
#   get_noise_multiplier(self, step, epoch) -> float
#
# clip() receives a copy of the per-sample gradients (list of tensors, each
# [B, *param_shape]) and returns `scale`, the per-sample multipliers (a [B]
# tensor, or a list with one [B] tensor per parameter), and `clip_norm`, the
# L2 bound C_t that every scaled per-sample gradient satisfies. The harness
# averages the scaled gradients over the batch and adds N(0, (sigma_t*C_t/B)^2)
# noise, where sigma_t = get_noise_multiplier(step, epoch) (called after clip).
#
# IMPORTANT:
# - The total privacy budget (target_epsilon, target_delta) is FIXED. The
#   harness accounts every step with the sigma_t it applied and aborts a run
#   whose epsilon exceeds the target; a scaled per-sample gradient whose norm
#   exceeds clip_norm also aborts the run.

class DPMechanism:
    """Differentially private gradient mechanism.

    Standard DP-SGD: clip per-sample gradients to max_grad_norm and use the
    calibrated constant noise multiplier (noise std = sigma * C / B).
    """

    def __init__(self, max_grad_norm, noise_multiplier, n_params,
                 dataset_size, batch_size, epochs, target_epsilon, target_delta):
        self.max_grad_norm = max_grad_norm
        self.noise_multiplier = noise_multiplier
        self.n_params = n_params
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta

    def clip(self, per_sample_grads, step, epoch):
        """Choose per-sample clipping multipliers.

        Args:
            per_sample_grads: list of tensors, each [B, *param_shape] (a copy)
            step: current global training step
            epoch: current epoch number

        Returns:
            (scale, clip_norm): scale is a [B] tensor of per-sample multipliers
            (or a list of [B] tensors, one per parameter); clip_norm is the L2
            bound of every scaled per-sample gradient.
        """
        batch_size = per_sample_grads[0].shape[0]

        # Compute per-sample gradient norms (flat norm across all parameters)
        flat = torch.cat([g.reshape(batch_size, -1) for g in per_sample_grads], dim=1)
        norms = flat.norm(2, dim=1)  # [B]

        # Clip per-sample gradients to max_grad_norm
        clip_factor = (self.max_grad_norm / norms.clamp(min=1e-8)).clamp(max=1.0)  # [B]

        return clip_factor, self.max_grad_norm

    def get_noise_multiplier(self, step, epoch):
        """Return the noise multiplier sigma_t used (and accounted) this step."""
        return self.noise_multiplier

    # Design space: per-sample / per-layer clipping or normalization,
    # adaptive thresholds (C_t), and noise schedules (sigma_t). Helper
    # methods and module-level helpers may live in this section too.
    # The noise itself and the epsilon accounting are done by the harness.


# =====================================================================
# EDITABLE SECTION END
# =====================================================================


# =====================================================================
# FIXED: DP privatization and privacy accounting (DO NOT MODIFY)
# =====================================================================
# The Gaussian mechanism and the accountant live here, outside the editable
# section, so the reported epsilon is always computed from the clipping bound
# and noise multiplier that were actually applied. Modules are re-bound under
# private names so that names reassigned in the editable section cannot
# change the mechanism.
import math as _math
import torch as _torch

_CLIP_NORM_RTOL = 1e-4  # float slack on the per-sample L2 bound check
_EPSILON_RTOL = 1e-2    # slack for the sigma calibration's binary-search tolerance


def _rdp_epsilon(steps, sigma, q, delta):
    """Same RDP bound and conversion as compute_epsilon (kept private here)."""
    alphas = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))
    best_eps = float("inf")
    for alpha in alphas:
        if alpha <= 1:
            continue
        rdp = steps * min(
            q * q * alpha / (2 * sigma * sigma),
            alpha * q * q / (2 * sigma * sigma),
        )
        eps = rdp - _math.log(delta) / (alpha - 1) + _math.log(1 - 1 / alpha)
        if eps < best_eps:
            best_eps = eps
    return max(0, best_eps)


class _PrivacyLedger:
    """Composes the per-step noise multipliers the harness actually applied.

    Under this RDP bound a step with multiplier sigma_t costs
    alpha * q^2 / (2 sigma_t^2), so T steps compose to the cost of T steps at
    sigma_eff = sqrt(T / sum_t 1/sigma_t^2).
    """

    def __init__(self, q, delta):
        self.q = float(q)
        self.delta = float(delta)
        self.steps = 0
        self.inv_sq_sum = 0.0

    def record(self, sigma):
        self.steps += 1
        self.inv_sq_sum += 1.0 / (sigma * sigma)

    def effective_sigma(self):
        return (self.steps / self.inv_sq_sum) ** 0.5

    def epsilon(self):
        return _rdp_epsilon(self.steps, self.effective_sigma(), self.q, self.delta)


def _positive_finite(value, name):
    value = float(value)
    if not _math.isfinite(value) or value <= 0:
        raise RuntimeError(f"DPMechanism returned invalid {name}={value!r}; "
                           f"it must be a finite positive number")
    return value


def privatize_step(dp_mechanism, per_sample_grads, step, epoch, ledger):
    """One step of the Gaussian mechanism with the mechanism's C_t and sigma_t.

    The mechanism sees a copy of the per-sample gradients and returns
    per-sample multipliers plus the bound C_t; the harness scales the
    original gradients, verifies ||scaled_i|| <= C_t for every sample, averages
    over the batch, adds N(0, (sigma_t * C_t / B)^2) noise and records sigma_t.
    """
    batch_size = per_sample_grads[0].shape[0]
    scale, clip_norm = dp_mechanism.clip(
        [g.clone() for g in per_sample_grads], step, epoch
    )
    clip_norm = _positive_finite(clip_norm, "clip_norm")
    sigma = _positive_finite(dp_mechanism.get_noise_multiplier(step, epoch),
                             "noise multiplier")

    scales = list(scale) if isinstance(scale, (list, tuple)) else [scale] * len(per_sample_grads)
    if len(scales) != len(per_sample_grads):
        raise RuntimeError(f"DPMechanism.clip returned {len(scales)} scale tensors "
                           f"for {len(per_sample_grads)} parameters")
    clipped = []
    for g, s in zip(per_sample_grads, scales):
        if type(s) is not _torch.Tensor or tuple(s.shape) != (batch_size,):
            raise RuntimeError("DPMechanism.clip must return per-sample scales as "
                               f"plain torch.Tensor of shape [{batch_size}]")
        shape = [batch_size] + [1] * (g.dim() - 1)
        clipped.append(g * s.detach().reshape(shape))

    norms = _torch.cat([c.reshape(batch_size, -1) for c in clipped], dim=1).norm(2, dim=1)
    if not bool((norms <= clip_norm * (1 + _CLIP_NORM_RTOL)).all()):
        raise RuntimeError(
            f"DP violation at step {step}: a scaled per-sample gradient has norm "
            f"{norms.max().item():.6g} > clip_norm={clip_norm:.6g}"
        )

    noised_grads = []
    for c in clipped:
        avg = c.mean(dim=0)
        noise = _torch.randn_like(avg) * (sigma * clip_norm / batch_size)
        noised_grads.append(avg + noise)
    ledger.record(sigma)
    return noised_grads


# =====================================================================
# FIXED: Data loading (DO NOT MODIFY)
# =====================================================================

def get_data_loaders(dataset_name, batch_size, data_root=os.environ.get("DATA_ROOT", "/data")):
    """Create train and test data loaders."""
    if dataset_name == "mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        train_ds = datasets.MNIST(
            os.path.join(data_root, "mnist"), train=True, download=False, transform=transform
        )
        test_ds = datasets.MNIST(
            os.path.join(data_root, "mnist"), train=False, download=False, transform=transform
        )
        model_cls = MNISTNet
    elif dataset_name == "cifar10":
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_ds = datasets.CIFAR10(
            os.path.join(data_root, "cifar10"), train=True, download=False, transform=transform_train
        )
        test_ds = datasets.CIFAR10(
            os.path.join(data_root, "cifar10"), train=False, download=False, transform=transform_test
        )
        model_cls = CIFAR10Net
    elif dataset_name == "fmnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,)),
        ])
        train_ds = datasets.FashionMNIST(
            os.path.join(data_root, "fmnist"), train=True, download=False, transform=transform
        )
        test_ds = datasets.FashionMNIST(
            os.path.join(data_root, "fmnist"), train=False, download=False, transform=transform
        )
        model_cls = MNISTNet
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=1024, shuffle=False,
                             num_workers=2, pin_memory=True)
    return train_ds, train_loader, test_loader, model_cls


# =====================================================================
# FIXED: Per-sample gradient computation (DO NOT MODIFY)
# =====================================================================

def compute_per_sample_gradients(model, data, target, criterion):
    """Compute per-sample gradients using functorch-style vmap.

    Returns a list of tensors, each of shape [B, *param_shape].
    """
    params = [p for p in model.parameters() if p.requires_grad]

    # Manual per-sample gradient computation via backward on each sample
    batch_size = data.shape[0]
    per_sample_grads = [torch.zeros(batch_size, *p.shape, device=p.device) for p in params]

    for i in range(batch_size):
        model.zero_grad()
        output = model(data[i:i+1])
        loss = criterion(output, target[i:i+1])
        loss.backward()
        for j, p in enumerate(params):
            if p.grad is not None:
                per_sample_grads[j][i] = p.grad.clone()

    return per_sample_grads


def compute_per_sample_gradients_fast(model, data, target, criterion):
    """Efficient per-sample gradient computation using ghost clipping trick.

    Computes per-sample gradient norms first, then uses weighted loss for aggregation.
    Falls back to loop-based computation for small batches.
    """
    batch_size = data.shape[0]

    # For moderate batch sizes, use vectorized approach via autograd
    if batch_size <= 128:
        return compute_per_sample_gradients(model, data, target, criterion)

    # For larger batches, use microbatching for memory efficiency
    micro_bs = 64
    params = [p for p in model.parameters() if p.requires_grad]
    per_sample_grads = [torch.zeros(batch_size, *p.shape, device=p.device) for p in params]

    for start in range(0, batch_size, micro_bs):
        end = min(start + micro_bs, batch_size)
        micro_data = data[start:end]
        micro_target = target[start:end]
        for i in range(end - start):
            model.zero_grad()
            output = model(micro_data[i:i+1])
            loss = criterion(output, micro_target[i:i+1])
            loss.backward()
            for j, p in enumerate(params):
                if p.grad is not None:
                    per_sample_grads[j][start + i] = p.grad.clone()

    return per_sample_grads


# =====================================================================
# FIXED: Training and evaluation loops (DO NOT MODIFY)
# =====================================================================

def train_epoch(model, train_loader, optimizer, criterion, dp_mechanism, device,
                epoch, total_steps, ledger, log_interval=50):
    """Train one epoch with DP mechanism."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    step = total_steps

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        batch_size = data.shape[0]

        # Compute per-sample gradients
        per_sample_grads = compute_per_sample_gradients(model, data, target, criterion)

        # Apply the DP mechanism: clipping/schedule from DPMechanism (EDITABLE),
        # noise and accounting from privatize_step (FIXED)
        noised_grads = privatize_step(dp_mechanism, per_sample_grads, step, epoch, ledger)

        # Set model gradients
        optimizer.zero_grad()
        for param, grad in zip(
            [p for p in model.parameters() if p.requires_grad], noised_grads
        ):
            param.grad = grad

        optimizer.step()

        # Compute batch metrics (without grad)
        with torch.no_grad():
            output = model(data)
            loss = criterion(output, target)
            running_loss += loss.item() * batch_size
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += batch_size

        step += 1

        if (batch_idx + 1) % log_interval == 0:
            avg_loss = running_loss / total
            acc = 100.0 * correct / total
            print(
                f"TRAIN_METRICS epoch={epoch} step={step} loss={avg_loss:.6f} "
                f"accuracy={acc:.2f}",
                flush=True,
            )

    return step, running_loss / total, 100.0 * correct / total


def evaluate(model, test_loader, criterion, device):
    """Evaluate model on test set."""
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item() * data.shape[0]
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += data.shape[0]

    return test_loss / total, 100.0 * correct / total


# =====================================================================
# FIXED: Main entry point (DO NOT MODIFY)
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="DP-SGD Benchmark")
    parser.add_argument("--dataset", type=str, default="mnist",
                        choices=["mnist", "cifar10", "fmnist"],
                        help="Dataset to train on")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Training batch size")
    parser.add_argument("--lr", type=float, default=0.1,
                        help="Learning rate")
    parser.add_argument("--max-grad-norm", type=float, default=1.0,
                        help="Max per-sample gradient norm for clipping")
    parser.add_argument("--target-epsilon", type=float, default=3.0,
                        help="Target epsilon for privacy budget")
    parser.add_argument("--target-delta", type=float, default=1e-5,
                        help="Target delta for privacy budget")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use")
    args = parser.parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load data
    train_ds, train_loader, test_loader, model_cls = get_data_loaders(
        args.dataset, args.batch_size
    )
    dataset_size = len(train_ds)
    q = args.batch_size / dataset_size
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs

    # Calibrate noise to target epsilon
    sigma = calibrate_noise_to_epsilon(
        args.target_epsilon, total_steps, q, args.target_delta
    )
    print(f"Calibrated noise_multiplier sigma={sigma:.4f} for "
          f"epsilon={args.target_epsilon}, delta={args.target_delta}, "
          f"steps={total_steps}, q={q:.4f}", flush=True)

    # Create model
    model = model_cls().to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {model_cls.__name__}, Parameters: {n_params}", flush=True)

    # Create optimizer (SGD with momentum, standard for DP training)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    # Learning rate schedule: cosine annealing
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    criterion = nn.CrossEntropyLoss()

    # Initialize DP mechanism (EDITABLE)
    dp_mechanism = DPMechanism(
        max_grad_norm=args.max_grad_norm,
        noise_multiplier=sigma,
        n_params=n_params,
        dataset_size=dataset_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        target_epsilon=args.target_epsilon,
        target_delta=args.target_delta,
    )

    # Privacy ledger (FIXED): records the sigma_t applied at every step
    ledger = _PrivacyLedger(q, args.target_delta)
    epsilon_limit = args.target_epsilon * (1 + _EPSILON_RTOL)

    # Training loop
    global_step = 0
    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        global_step, train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, dp_mechanism, device,
            epoch, global_step, ledger, log_interval=50,
        )
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        # Compute current epsilon spend from the noise actually applied
        effective_sigma = ledger.effective_sigma()
        eps_spent = ledger.epsilon()

        print(
            f"Epoch {epoch}/{args.epochs}: "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.2f}% "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.2f}% "
            f"epsilon_spent={eps_spent:.2f} sigma={effective_sigma:.4f}",
            flush=True,
        )

        if eps_spent > epsilon_limit:
            raise RuntimeError(
                f"Privacy budget exceeded: epsilon={eps_spent:.4f} > "
                f"target_epsilon={args.target_epsilon} after {ledger.steps} steps"
            )

        if test_acc > best_acc:
            best_acc = test_acc

        scheduler.step()

    # Print final test metrics
    final_test_loss, final_test_acc = evaluate(model, test_loader, criterion, device)
    eps_final = ledger.epsilon()

    print(f"\nTEST_METRICS accuracy={final_test_acc:.4f} "
          f"epsilon={eps_final:.4f} best_accuracy={best_acc:.4f}",
          flush=True)

    print(f"\nFinal Results: accuracy={final_test_acc:.2f}%, "
          f"best_accuracy={best_acc:.2f}%, epsilon={eps_final:.2f}",
          flush=True)


if __name__ == "__main__":
    main()
