#!/usr/bin/env python3
"""Network-pruning harness (REAL): prune a REAL model on REAL data.

Compresses a trained CIFAR-10 classifier (torchvision ResNet-18 adapted for 32x32)
by removing weights or channels at a fixed, enforced budget, then runs a full
160-epoch recovery and measures accuracy on the official CIFAR-10 test split.
FLOPs (MACs) and parameter context are reported via Torch-Pruning.

This is the REAL grounding of the former synthetic toy: real CIFAR-10 (50k/10k), a real
architecture (ResNet-18), and faithful pruning — magnitude / Taylor (unstructured) and
L1/Taylor/BN-scale structured channel pruning (via Torch-Pruning's DependencyGraph,
which correctly handles ResNet residual coupling) — with a full 160-epoch recovery fine-tune.

The dense starting point is fixed. Verification strictly loads an independently
pinned checkpoint carrying at least 200 training epochs; it never trains a dense
model or substitutes a different checkpoint at runtime.

The pruning and recovery budgets are fixed and enforced by the harness. Final test
accuracy is the objective; dense accuracy, the 1/10 class-prior reference, measured
MACs, and parameter count are reported only as protocol evidence and context.

Surfaces (one per task, --surface):

  UNSTRUCTURED (global or per-layer weight sparsity, pure-torch):
    criterion       -> importance(name, weight, grad): per-weight importance SCORE.
    taylor_estimator-> estimate_importance(model, batches, params): data-aware
                       importance (how many grad batches, sign/scale) -> per-weight.
    second_order    -> importance2(name, weight, grad, fisher): magnitude / Taylor /
                       OBS-style (0.5 * w^2 * Fisher-diag) importance.
    layer_budget    -> layer_sparsity(layer_names): per-layer sparsity allocation
                       averaging to the global target (sensitivity-aware vs uniform).
    schedule        -> schedule(target_sparsity, total_steps): list of (sp, steps) for
                       one-shot vs iterative/gradual pruning.
    reg_prune       -> regularizer(model, params): extra sparsity loss during a fixed
                       full 160-epoch pre-prune phase, then magnitude-threshold.
  STRUCTURED (real FLOPs/latency drop, via Torch-Pruning DependencyGraph):
    structured_criterion -> importance_spec(): channel importance for structured
                       pruning: {type: l1|l2|taylor|random|bn, ...}.
    flops_budget    -> importance_spec(): channel importance used while the harness
                       enforces the measured-MAC upper budget.
  RECOVERY / REINIT (fixed mask; the agent controls the post-prune adaptation):
    recovery        -> recover(model, masked_finetune, cfg): the recovery fine-tune.
    recovery_distill-> recovery_loss(logits, targets, teacher_logits): CE vs KD-aware
                       recovery objective (teacher = the fixed dense model).
    reinit          -> reinit(): post-prune weight init for surviving weights:
                       keep | rewind (lottery-ticket) | random.

Metric line (one per run):
    PRUNE_METRICS surface=<S> setting=<L> acc=<..> sparsity=<..> dense_acc=<..> \
        pruned_acc_prefinetune=<..> nparams=<..> flops=<..> chance=<..>
acc (official test accuracy after the protocol, higher is better) is primary.
dense_acc (loaded dense checkpoint), chance (=0.1), nparams (pruned-model trainable
params), and flops (MACs of one 32x32
forward) are context. For unstructured pruning flops == dense flops (sparse != smaller);
for structured pruning flops drops for real (the model is genuinely narrower).

Every editable hook is fail-closed: malformed, crashing, missing, wrong-shape, or
non-finite output terminates verification without emitting a metric record.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
# Fixed protocol
# --------------------------------------------------------------------------- #
NUM_CLASSES = 10
IMG = 32
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)
BATCH = 128
RECOVERY_EPOCHS = 160          # community-scale CIFAR recovery schedule (FIXED)
RECOVERY_LR = 1e-2             # SGD lr for recovery (with cosine decay)
REG_PRETRAIN_EPOCHS = 160      # full regularized training phase (FIXED)
DEFAULT_SPARSITY = 0.70        # default enforced budget (overridden per task)
WEIGHT_DECAY = 5e-4
DENSE_PROTOCOL = "cifar10-resnet18-200ep-v1"
# Must match the independently approved artifact digest in prepare_data.py.
# Empty is deliberate until the artifact owner supplies that trusted digest.
EXPECTED_CHECKPOINT_SHA256 = ""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_pin() -> str:
    expected = EXPECTED_CHECKPOINT_SHA256.strip().lower()
    if len(expected) != 64 or any(ch not in "0123456789abcdef" for ch in expected):
        raise SystemExit(
            "EXPECTED_CHECKPOINT_SHA256 is not configured with the independently "
            "approved dense-checkpoint digest"
        )
    return expected


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# --------------------------------------------------------------------------- #
# Data: REAL CIFAR-10 (torchvision format, no download in the worker)
# --------------------------------------------------------------------------- #
def load_cifar10(root: str, batch: int, num_workers: int):
    import torchvision
    import torchvision.transforms as T
    train_tf = T.Compose([
        T.RandomCrop(IMG, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    test_tf = T.Compose([
        T.ToTensor(),
        T.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    train_set = torchvision.datasets.CIFAR10(root, train=True, download=False,
                                             transform=train_tf)
    test_set = torchvision.datasets.CIFAR10(root, train=False, download=False,
                                            transform=test_tf)
    if len(train_set) != 50_000 or len(test_set) != 10_000:
        raise SystemExit(
            f"full CIFAR-10 inventory required, got train={len(train_set)} test={len(test_set)}"
        )
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch, shuffle=True, num_workers=num_workers,
        pin_memory=True, drop_last=False)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=256, shuffle=False, num_workers=num_workers,
        pin_memory=True)
    return train_loader, test_loader


# --------------------------------------------------------------------------- #
# Model: REAL torchvision ResNet-18 adapted for CIFAR-10 (32x32, 10 classes)
# --------------------------------------------------------------------------- #
def build_resnet18_cifar(num_classes: int = NUM_CLASSES) -> nn.Module:
    import torchvision
    model = torchvision.models.resnet18(weights=None, num_classes=num_classes)
    # CIFAR stem: 3x3 stride-1 conv, no maxpool (standard for 32x32).
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


# --------------------------------------------------------------------------- #
# Prunable parameters (conv + linear weights; biases/BN not pruned)
# --------------------------------------------------------------------------- #
def prunable_params(model):
    out = []
    for name, mod in model.named_modules():
        if isinstance(mod, (nn.Conv2d, nn.Linear)):
            out.append((name + ".weight", mod.weight))
    return out


def conv_layers(model):
    """(name, Conv2d module) for structured channel pruning surfaces."""
    return [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Conv2d)]


# --------------------------------------------------------------------------- #
# Load the agent-editable surface
# --------------------------------------------------------------------------- #
def load_surface(sol_path: Path):
    spec = importlib.util.spec_from_file_location("agent_surface", str(sol_path))
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(sol_path.parent))
    spec.loader.exec_module(mod)  # type: ignore
    return mod


# --------------------------------------------------------------------------- #
# Train / eval
# --------------------------------------------------------------------------- #
def accuracy(model, loader, device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.numel()
    return correct / max(1, total)


def _sgd(model, lr, wd=WEIGHT_DECAY):
    return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                           weight_decay=wd, nesterov=True)


def _recover_loop(model, train_loader, epochs, lr, device, seed,
                  masks=None, teacher=None, loss_fn=None):
    """Full 160-epoch recovery schedule. If `masks` given, re-apply every step (sparsity
    stays fixed). If `teacher` given, `loss_fn(logits, targets, teacher_logits)`
    is used (KD); else cross_entropy. Returns nothing (model trained in place)."""
    model.train()
    if teacher is not None:
        teacher.eval()
    opt = _sgd(model, lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    if masks is not None:
        apply_masks(prunable_params(model), masks)
    for _ in range(epochs):
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad()
            logits = model(x)
            if loss_fn is not None and teacher is not None:
                with torch.no_grad():
                    tlogits = teacher(x)
                loss = loss_fn(logits, y, tlogits)
            else:
                loss = F.cross_entropy(logits, y)
            loss.backward()
            opt.step()
            if masks is not None:
                apply_masks(prunable_params(model), masks)
        sched.step()


# --------------------------------------------------------------------------- #
# Importance + masks (unstructured), with EXACT budget enforcement
# --------------------------------------------------------------------------- #
def magnitude_scores(params):
    return {name: p.detach().abs() for name, p in params}


def random_scores(params):
    """Uniform-random importance used only by explicit negative-control baselines."""
    return {name: torch.rand(p.shape, dtype=torch.float32, device=p.device)
            for name, p in params}


def compute_abs_grads(model, train_loader, device, nb=None):
    """Accumulate |grad| over the full train inventory unless explicitly bounded."""
    model.zero_grad()
    grads = {n: torch.zeros_like(p) for n, p in prunable_params(model)}
    cnt = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        F.cross_entropy(model(x), y).backward()
        for n, p in prunable_params(model):
            if p.grad is not None:
                grads[n] += p.grad.detach().abs()
        model.zero_grad()
        cnt += 1
        if nb is not None and cnt >= nb:
            break
    for n in grads:
        grads[n] /= max(1, cnt)
    return grads


def compute_fisher_diag(model, train_loader, device, nb=None):
    """Fisher-diagonal proxy over the full train inventory by default."""
    model.zero_grad()
    fish = {n: torch.zeros_like(p) for n, p in prunable_params(model)}
    cnt = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        F.cross_entropy(model(x), y).backward()
        for n, p in prunable_params(model):
            if p.grad is not None:
                fish[n] += p.grad.detach().pow(2)
        model.zero_grad()
        cnt += 1
        if nb is not None and cnt >= nb:
            break
    for n in fish:
        fish[n] /= max(1, cnt)
    return fish


def build_masks_global(scores, params, sparsity, eligible_masks=None):
    """Keep the globally top (1-sparsity) fraction by score; zero the rest.
    EXACT global keep count. When eligible_masks is supplied, removed weights
    cannot be selected again. Boundary ties use flattened parameter order."""
    flat_scores = []
    eligible = {}
    total = 0
    for name, p in params:
        s = _clean_score(scores.get(name), p)
        total += p.numel()
        if eligible_masks is None:
            mask = torch.ones_like(p, dtype=torch.bool)
        else:
            if name not in eligible_masks:
                raise ValueError(f"eligible mask is missing {name}")
            mask = torch.as_tensor(eligible_masks[name], device=p.device, dtype=torch.bool)
            if mask.shape != p.shape:
                raise ValueError(f"eligible mask shape for {name} does not match its weight")
        eligible[name] = mask
        flat_scores.append(s.reshape(-1)[mask.reshape(-1)])

    n_keep = max(1, int(round((1.0 - sparsity) * total)))
    candidate_scores = torch.cat(flat_scores)
    if n_keep > candidate_scores.numel():
        raise ValueError("requested mask would reactivate previously removed weights")

    threshold = torch.topk(
        candidate_scores, n_keep, largest=True, sorted=False
    ).values.min()
    candidate_keep = candidate_scores > threshold
    remaining = n_keep - int(candidate_keep.sum())
    if remaining:
        tied = torch.nonzero(candidate_scores == threshold, as_tuple=False).reshape(-1)
        if tied.numel() < remaining:
            raise RuntimeError("could not resolve the exact global importance boundary")
        candidate_keep[tied[:remaining]] = True

    masks = {}
    offset = 0
    for name, p in params:
        allowed = eligible[name].reshape(-1)
        count = int(allowed.sum())
        mask = torch.zeros(p.numel(), dtype=torch.bool, device=p.device)
        mask[allowed] = candidate_keep[offset:offset + count]
        masks[name] = mask.reshape(p.shape)
        offset += count
    if offset != candidate_scores.numel() or sum(int(mask.sum()) for mask in masks.values()) != n_keep:
        raise RuntimeError("global pruning did not enforce the exact keep count")
    return masks


def build_masks_per_layer(scores, params, layer_sp, default_sp):
    """Honor a relative per-layer allocation at an exact global budget."""
    total = sum(p.numel() for _, p in params)
    target_pruned = int(round(float(default_sp) * total))
    raw = {
        name: min(max(float(layer_sp.get(name, default_sp)), 0.0), 0.99)
        for name, _ in params
    }
    if not any(value > 0.0 for value in raw.values()):
        raise ValueError("layer sparsity allocation must prune at least one layer")

    # Scale the requested per-layer ratios by parameter count, not by number of
    # layers. Bisection handles clipping at 0.99 without changing their ordering.
    lo, hi = 0.0, 1.0
    while sum(min(0.99, raw[name] * hi) * p.numel() for name, p in params) < target_pruned:
        hi *= 2.0
        if hi > 1e6:
            raise ValueError("layer sparsity allocation cannot meet the global budget")
    for _ in range(80):
        mid = (lo + hi) / 2.0
        pruned = sum(min(0.99, raw[name] * mid) * p.numel() for name, p in params)
        if pruned < target_pruned:
            lo = mid
        else:
            hi = mid

    ideals = []
    counts = {}
    for name, p in params:
        ideal = min(0.99, raw[name] * hi) * p.numel()
        base = int(math.floor(ideal))
        counts[name] = base
        ideals.append((ideal - base, name, p.numel()))
    remaining = target_pruned - sum(counts.values())
    for _, name, capacity in sorted(ideals, reverse=True):
        if remaining <= 0:
            break
        if counts[name] < capacity - 1:
            counts[name] += 1
            remaining -= 1
    if remaining != 0:
        raise RuntimeError("could not allocate the exact global pruning budget")

    masks = {}
    for name, p in params:
        s = _clean_score(scores.get(name), p).reshape(-1)
        n = s.numel()
        n_keep = n - counts[name]
        if not 1 <= n_keep <= n:
            raise RuntimeError("per-layer budget left an invalid keep count")
        thresh = torch.topk(s, n_keep, largest=True).values.min()
        m = (s >= thresh)
        # exact within-layer
        k = int(m.sum())
        if k > n_keep:
            idx = torch.nonzero(m, as_tuple=False).reshape(-1)
            order = torch.argsort(s[idx])
            m[idx[order[:k - n_keep]]] = False
        elif k < n_keep:
            idx = torch.nonzero(~m, as_tuple=False).reshape(-1)
            order = torch.argsort(s[idx], descending=True)
            m[idx[order[:n_keep - k]]] = True
        masks[name] = m.reshape(p.shape)
    if sum(int((~mask).sum()) for mask in masks.values()) != target_pruned:
        raise RuntimeError("per-layer pruning did not enforce the exact global budget")
    return masks


def _clean_score(s, p):
    if s is None:
        raise ValueError("importance score is missing")
    s = torch.as_tensor(s, device=p.device, dtype=torch.float32)
    if s.shape != p.shape:
        raise ValueError(f"importance shape {tuple(s.shape)} != weight shape {tuple(p.shape)}")
    if not torch.isfinite(s).all() or torch.any(s < 0):
        raise ValueError("importance score must be finite and non-negative")
    return s.detach().to(p.device, p.dtype)


def apply_masks(params, masks):
    with torch.no_grad():
        for name, p in params:
            p.mul_(masks[name].to(p.dtype))


def measured_sparsity(params, masks):
    tot = sum(p.numel() for _, p in params)
    zeros = sum(int((~masks[n]).sum()) for n, _ in params)
    return zeros / max(1, tot)


# --------------------------------------------------------------------------- #
# FLOPs / params context (torch-pruning)
# --------------------------------------------------------------------------- #
def count_flops_params(model, device):
    """Return (macs, nparams). torch-pruning's op_counter deep-copies the model to
    CPU, so we always count on a CPU copy with a CPU example (CUDA example vs CPU
    copy mismatches otherwise)."""
    import torch_pruning as tp
    mcpu = copy.deepcopy(model).cpu().eval()
    example = torch.randn(1, 3, IMG, IMG)
    macs, _ = tp.utils.count_ops_and_params(mcpu, example)
    nparams = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if not math.isfinite(float(macs)) or int(macs) <= 0 or nparams <= 0:
        raise RuntimeError("FLOPs/parameter accounting did not complete")
    return int(macs), int(nparams)


# --------------------------------------------------------------------------- #
# Structured channel pruning via Torch-Pruning (handles ResNet residual coupling)
# --------------------------------------------------------------------------- #
def _build_importance(spec):
    import torch_pruning as tp
    if not isinstance(spec, dict) or "type" not in spec:
        raise TypeError("importance_spec() must return a dict containing 'type'")
    t = str(spec["type"]).lower()
    p = int(spec.get("p", 1 if t == "l1" else 2))
    if t in ("l1", "magnitude"):
        return tp.importance.MagnitudeImportance(p=1), "l1"
    if t == "l2":
        return tp.importance.MagnitudeImportance(p=2), "l2"
    if t == "taylor":
        return tp.importance.TaylorImportance(), "taylor"
    if t == "random":
        return tp.importance.RandomImportance(), "random"
    if t in ("bn", "bnscale", "bn_scale"):
        return tp.importance.BNScaleImportance(), "bn"
    raise ValueError(f"unsupported structured importance type: {t}")


def structured_prune(model, device, spec, ch_sparsity, train_loader=None):
    """Prune output channels globally to channel-sparsity `ch_sparsity` using
    torch-pruning's DependencyGraph (residual-aware). The graph build + channel
    removal run on CPU (torch-pruning's tracer/counter are CPU-only); the pruned
    model is moved back to `device` for recovery. `model.cpu()` is in-place, so the
    caller's `model` object is pruned in place and returned on `device`.
    Returns (macs0, macs1, nparams1)."""
    import torch_pruning as tp
    model.cpu()  # in-place; same object the caller holds
    example = torch.randn(1, 3, IMG, IMG)
    macs0, _ = tp.utils.count_ops_and_params(copy.deepcopy(model).eval(), example)
    imp, kind = _build_importance(spec)
    ignored = [model.fc] if hasattr(model, "fc") else None
    pruner = tp.pruner.MagnitudePruner(
        model, example, importance=imp, pruning_ratio=float(ch_sparsity),
        global_pruning=True, iterative_steps=1, round_to=8,
        ignored_layers=ignored,
    )
    if kind == "taylor" and train_loader is not None:
        # TaylorImportance consumes ordinary weight gradients. The regularizer
        # methods belong to sparsity-learning pruners and are not part of this API.
        model.train()
        cnt = 0
        for x, y in train_loader:
            x, y = x.cpu(), y.cpu()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            cnt += 1
            if cnt >= int(spec.get("batches", 4)):
                break
    pruner.step()
    macs1, nparams1 = tp.utils.count_ops_and_params(copy.deepcopy(model).eval(), example)
    model.to(device)
    return int(macs0), int(macs1), int(nparams1)


# --------------------------------------------------------------------------- #
# Surface runners
# --------------------------------------------------------------------------- #
def _get_importance(mod, params, grads, fn_name):
    scores = {}
    fn = getattr(mod, fn_name, None)
    if not callable(fn):
        raise TypeError(f"solution must define callable {fn_name}()")
    for name, p in params:
        g = grads.get(name)
        score = fn(name, p.detach(), None if g is None else g.detach())
        scores[name] = _clean_score(score, p)
    print(f"{fn_name.upper()}_APPLIED custom", flush=True)
    return scores


def run_unstructured(mod, model, train_loader, test_loader, device, sparsity, seed,
                     surface, recovery_epochs, teacher=None, init_sd=None):
    params = prunable_params(model)
    grads = {}
    fisher = {}
    if surface in ("criterion", "second_order"):
        grads = compute_abs_grads(model, train_loader, device)
    if surface == "second_order":
        fisher = compute_fisher_diag(model, train_loader, device)

    # --- importance scores ---
    if surface == "taylor_estimator":
        scores = _estimate_importance_surface(mod, model, train_loader, params,
                                              device)
    elif surface == "second_order":
        scores = _second_order_surface(mod, params, grads, fisher)
    elif surface == "reg_prune":
        _reg_pretrain(mod, model, train_loader, device)
        grads = compute_abs_grads(model, train_loader, device)
        scores = magnitude_scores(params)  # threshold by magnitude after reg
    elif surface == "criterion":
        scores = _get_importance(mod, params, grads, "importance")
    else:  # recovery / schedule / layer_budget / reinit / recovery_distill
        # These surfaces do not edit the mask; magnitude is the frozen scaffold.
        scores = magnitude_scores(params)

    # --- recovery objective (CE vs KD) ---
    loss_fn = None
    if surface == "recovery_distill":
        loss_fn = _recovery_loss_surface(mod)
        teacher = teacher  # dense teacher passed in

    # --- masks: global / per-layer / iterative ---
    if surface == "layer_budget":
        layer_sp = _layer_budget_surface(mod, [n for n, _ in params], sparsity)
        masks = build_masks_per_layer(scores, params, layer_sp, sparsity)
        apply_masks(params, masks)
        pre = accuracy(model, test_loader, device)
        _do_recovery(mod, model, train_loader, device, recovery_epochs, RECOVERY_LR,
                     seed, masks, surface, teacher, loss_fn, init_sd)
        return masks, pre

    if surface == "schedule":
        steps = _schedule_surface(mod, sparsity, recovery_epochs)
        masks = None
        pre = None
        for sp, ep in steps:
            # Re-rank the current surviving weights after each recovery rung.
            # eligible_masks makes the new mask nested even at exact-score ties.
            current_scores = magnitude_scores(params)
            next_masks = build_masks_global(
                current_scores, params, float(sp), eligible_masks=masks
            )
            if masks is not None and any(
                torch.any(next_masks[name] & ~masks[name]) for name in next_masks
            ):
                raise RuntimeError("pruning schedule attempted to reactivate a removed weight")
            masks = next_masks
            apply_masks(params, masks)
            if pre is None:
                pre = accuracy(model, test_loader, device)
            if ep > 0:
                _recover_loop(model, train_loader, int(ep), RECOVERY_LR, device,
                              seed, masks=masks, teacher=teacher, loss_fn=loss_fn)
        return masks, pre

    # one-shot (criterion / recovery / reinit / reg_prune / taylor_estimator /
    # second_order / recovery_distill)
    masks = build_masks_global(scores, params, sparsity)
    apply_masks(params, masks)
    pre = accuracy(model, test_loader, device)

    if surface == "reinit":
        _apply_reinit(mod, model, masks, init_sd)

    _do_recovery(mod, model, train_loader, device, recovery_epochs, RECOVERY_LR,
                 seed, masks, surface, teacher, loss_fn, init_sd)
    return masks, pre


def _do_recovery(mod, model, train_loader, device, epochs, lr, seed, masks,
                 surface, teacher, loss_fn, init_sd):
    if surface == "recovery":
        # agent controls the loop via recover(model, masked_finetune, cfg)
        cfg = dict(epochs=epochs, lr=lr, batch=BATCH)
        remaining_epochs = epochs

        def masked_finetune(epochs=epochs, lr=lr):
            nonlocal remaining_epochs
            if isinstance(epochs, bool):
                raise ValueError("recovery epochs must be an integer")
            requested = int(epochs)
            if requested < 0 or float(epochs) != requested:
                raise ValueError("recovery epochs must be a non-negative integer")
            learning_rate = float(lr)
            if not math.isfinite(learning_rate) or not 0.0 < learning_rate <= 0.1:
                raise ValueError("recovery learning rate must be finite and in (0,0.1]")
            if requested > remaining_epochs:
                raise ValueError(
                    f"recovery request exceeds the remaining budget: "
                    f"requested={requested}, remaining={remaining_epochs}"
                )
            if requested:
                _recover_loop(model, train_loader, requested, learning_rate, device,
                              seed, masks=masks)
            remaining_epochs -= requested

        fn = getattr(mod, "recover", None)
        if not callable(fn):
            raise TypeError("solution must define callable recover()")
        fn(model, masked_finetune, cfg)
        if remaining_epochs != 0:
            raise ValueError(
                f"recover() must consume the complete {epochs}-epoch budget; "
                f"{remaining_epochs} epochs remain"
            )
        apply_masks(prunable_params(model), masks)
        print(f"RECOVERY_APPLIED used_epochs={epochs - remaining_epochs}", flush=True)
    else:
        _recover_loop(model, train_loader, epochs, lr, device, seed, masks=masks,
                      teacher=teacher, loss_fn=loss_fn)


def _estimate_importance_surface(mod, model, train_loader, params, device):
    """Surface returns name->importance using one complete train pass."""
    batches = list(train_loader)
    fn = getattr(mod, "estimate_importance", None)
    if not callable(fn):
        raise TypeError("solution must define callable estimate_importance()")
    res = fn(model, batches, params)
    if not isinstance(res, dict):
        raise TypeError("estimate_importance() must return a dict")
    scores = {name: _clean_score(res.get(name), p) for name, p in params}
    print(f"ESTIMATE_IMPORTANCE_APPLIED batches={len(batches)}", flush=True)
    return scores


def _second_order_surface(mod, params, grads, fisher):
    scores = {}
    fn = getattr(mod, "importance2", None)
    if not callable(fn):
        raise TypeError("solution must define callable importance2()")
    for name, p in params:
        g = grads.get(name)
        h = fisher.get(name)
        score = fn(
            name,
            p.detach(),
            None if g is None else g.detach(),
            None if h is None else h.detach(),
        )
        scores[name] = _clean_score(score, p)
    print("SECOND_ORDER_APPLIED", flush=True)
    return scores


def _reg_pretrain(mod, model, train_loader, device):
    """Full fixed pre-prune phase with the agent's sparsity regularizer added."""
    reg = getattr(mod, "regularizer", None)
    if not callable(reg):
        raise TypeError("solution must define callable regularizer()")
    model.train()
    params = prunable_params(model)
    opt = _sgd(model, lr=0.05)
    for ep in range(REG_PRETRAIN_EPOCHS):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y)
            r = torch.as_tensor(reg(model, params), device=device, dtype=torch.float32)
            if r.numel() != 1 or not torch.isfinite(r).all() or float(r) < 0.0:
                raise ValueError("regularizer() must return one finite non-negative scalar")
            loss = loss + r.reshape(())
            loss.backward()
            opt.step()
    print("REG_PRUNE_APPLIED", flush=True)


def _recovery_loss_surface(mod):
    fn = getattr(mod, "recovery_loss", None)
    if not callable(fn):
        raise TypeError("solution must define callable recovery_loss()")

    def loss_fn(logits, targets, tlogits):
        loss = torch.as_tensor(
            fn(logits, targets, tlogits),
            device=logits.device,
            dtype=torch.float32,
        )
        if loss.numel() != 1 or not torch.isfinite(loss).all() or float(loss) < 0.0:
            raise ValueError("recovery_loss() must return one finite non-negative scalar")
        return loss.reshape(())

    print("RECOVERY_LOSS_APPLIED (KD/custom)", flush=True)
    return loss_fn


def _layer_budget_surface(mod, layer_names, target):
    fn = getattr(mod, "layer_sparsity", None)
    if not callable(fn):
        raise TypeError("solution must define callable layer_sparsity()")
    raw = fn(list(layer_names))
    if not isinstance(raw, dict):
        raise TypeError("layer_sparsity() must return a dict")
    unknown = set(raw) - set(layer_names)
    if unknown:
        raise ValueError(f"layer_sparsity() returned unknown layers: {sorted(unknown)}")
    res = {key: float(value) for key, value in raw.items()}
    if any(not math.isfinite(value) or not 0.0 <= value < 1.0 for value in res.values()):
        raise ValueError("layer sparsities must be finite values in [0,1)")
    # Missing entries use the global target. Exact parameter-weighted scaling is
    # applied by build_masks_per_layer(), which has access to tensor sizes.
    print("LAYER_BUDGET_APPLIED", flush=True)
    return res


def _schedule_surface(mod, target, total_epochs):
    """Returns list of (sparsity, epochs). The harness applies cumulative masks at
    each rung and fine-tunes `epochs` between. Total epochs is capped at the budget."""
    fn = getattr(mod, "schedule", None)
    if not callable(fn):
        raise TypeError("solution must define callable schedule()")
    raw = fn(float(target), int(total_epochs))
    if not isinstance(raw, (list, tuple)) or not raw:
        raise ValueError("schedule() must return at least one rung")
    steps = []
    previous = -1.0
    for item in raw:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError("each schedule rung must be (sparsity, epochs)")
        sp = float(item[0])
        if isinstance(item[1], bool):
            raise ValueError("schedule epochs must be integers")
        ep = int(item[1])
        if (
            not math.isfinite(sp)
            or not 0.0 <= sp <= target
            or sp < previous
            or ep < 0
            or float(item[1]) != ep
        ):
            raise ValueError("schedule sparsities must be finite/monotone and epochs non-negative")
        previous = sp
        steps.append((sp, ep))
    if abs(steps[-1][0] - target) > 1e-9:
        raise ValueError("schedule must end at the enforced target sparsity")
    if sum(ep for _, ep in steps) != total_epochs:
        raise ValueError("schedule must consume the complete fixed recovery budget")
    print(f"SCHEDULE_APPLIED rungs={steps}", flush=True)
    return steps


def _apply_reinit(mod, model, masks, init_sd):
    fn = getattr(mod, "reinit", None)
    if not callable(fn):
        raise TypeError("solution must define callable reinit()")
    choice = str(fn()).strip().lower()
    if choice not in {"keep", "rewind", "random"}:
        raise ValueError("reinit() must return keep, rewind, or random")
    params = dict(prunable_params(model))
    sd = init_sd or {}
    with torch.no_grad():
        for name, p in params.items():
            m = masks[name]
            if choice == "random":
                new = torch.empty_like(p)
                nn.init.kaiming_uniform_(new.reshape(p.shape[0], -1)
                                         if p.ndim >= 2 else new.unsqueeze(0))
                new = new.reshape(p.shape)
                p.copy_(torch.where(m, new, torch.zeros_like(p)))
            elif choice == "rewind":
                if name not in sd:
                    raise ValueError(f"rewind checkpoint is missing {name}")
                base = sd[name].to(p.device, p.dtype)
                p.copy_(torch.where(m, base, torch.zeros_like(p)))
            # keep leaves trained surviving weights masked in place.
    print(f"REINIT_APPLIED choice={choice}", flush=True)


def run_structured_criterion(mod, model, train_loader, test_loader, device,
                             ch_sparsity, recovery_epochs, seed):
    fn = getattr(mod, "importance_spec", None)
    if not callable(fn):
        raise TypeError("solution must define callable importance_spec()")
    spec = fn()
    _build_importance(spec)  # validate before touching the model
    print(f"STRUCTURED_CRITERION_APPLIED spec={spec}", flush=True)
    _, _, _ = structured_prune(model, device, spec, ch_sparsity, train_loader)
    pre = accuracy(model, test_loader, device)
    _recover_loop(model, train_loader, recovery_epochs, RECOVERY_LR, device, seed)
    return None, pre


def run_flops_budget(mod, model, train_loader, test_loader, device,
                     flops_budget_frac, recovery_epochs, seed):
    """Agent returns a channel-importance spec; harness enforces an approximate FLOPs
    (MACs) BUDGET by pruning channels to a global channel-sparsity derived from the
    budget (conv MACs scale ~quadratically with channels, so ch_sparsity ~
    1 - sqrt(keep_frac)), using the surface's importance. Realized MACs are reported
    as the `flops` metric. Structured prune runs on CPU (torch-pruning tracer), then
    the model is moved back to `device` for recovery."""
    fn = getattr(mod, "importance_spec", None)
    if not callable(fn):
        raise TypeError("solution must define callable importance_spec()")
    spec = fn()
    if isinstance(spec, dict) and str(spec.get("type", "")).lower() == "taylor":
        raise ValueError("flops-budget surface does not provide gradients for Taylor importance")
    _build_importance(spec)  # validate before touching the model
    print(f"FLOPS_BUDGET_APPLIED spec={spec}", flush=True)
    model.cpu()
    example = torch.randn(1, 3, IMG, IMG)
    target_ch_sp = max(0.0, min(0.95, 1.0 - float(flops_budget_frac) ** 0.5))
    import torch_pruning as tp
    imp, _ = _build_importance(spec)
    ignored = [model.fc] if hasattr(model, "fc") else None
    pruner = tp.pruner.MagnitudePruner(
        model, example, importance=imp, pruning_ratio=target_ch_sp,
        global_pruning=True, iterative_steps=1, round_to=8,
        ignored_layers=ignored,
    )
    pruner.step()
    model.to(device)
    pre = accuracy(model, test_loader, device)
    _recover_loop(model, train_loader, recovery_epochs, RECOVERY_LR, device, seed)
    return None, pre


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
STRUCTURED_SURFACES = {"structured_criterion", "flops_budget"}
TASK_SURFACES = {
    "prune-criterion": "criterion",
    "prune-flops-budget": "flops_budget",
    "prune-layer-budget": "layer_budget",
    "prune-recovery": "recovery",
    "prune-recovery-distill": "recovery_distill",
    "prune-reg-prune": "reg_prune",
    "prune-reinit": "reinit",
    "prune-schedule": "schedule",
    "prune-second-order": "second_order",
    "prune-structured-criterion": "structured_criterion",
    "prune-taylor-estimator": "taylor_estimator",
}
ALL_SURFACES = set(TASK_SURFACES.values())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--solution", required=True)
    ap.add_argument("--task-id", required=True, choices=sorted(TASK_SURFACES))
    ap.add_argument("--surface", required=True, choices=sorted(ALL_SURFACES))
    ap.add_argument("--label", default="run")
    ap.add_argument("--sparsity", type=float, default=DEFAULT_SPARSITY)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dense-ckpt", default="")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--recovery-epochs", type=int, default=RECOVERY_EPOCHS)
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--flops-budget", type=float, default=0.5,
                    help="(flops_budget surface) fraction of dense MACs to keep")
    args = ap.parse_args()
    if TASK_SURFACES[args.task_id] != args.surface:
        raise SystemExit("task-id/surface mismatch")

    expected_seed = {"cifar10": 42, "cifar10_seed1": 1}.get(args.label)
    if expected_seed is None or args.seed != expected_seed:
        raise SystemExit("label/seed does not match the required two-setting protocol")
    if args.recovery_epochs != RECOVERY_EPOCHS or args.batch != BATCH:
        raise SystemExit(
            f"full protocol requires recovery_epochs={RECOVERY_EPOCHS}, batch={BATCH}"
        )
    if not 0.0 < args.sparsity < 1.0 or not 0.0 < args.flops_budget < 1.0:
        raise SystemExit("invalid pruning budget")

    set_all_seeds(args.seed)
    if not torch.cuda.is_available():
        raise SystemExit("network-pruning verification requires CUDA")
    device = torch.device("cuda")
    print(f"DEVICE {device} torch {torch.__version__}", flush=True)

    expected_checkpoint_sha256 = _checkpoint_pin()
    manifest_path = Path(args.manifest).resolve()
    if not manifest_path.is_file():
        raise SystemExit("image-prepared prune-lab manifest is missing")
    manifest = json.loads(manifest_path.read_text())
    required_manifest = {
        "checkpoint",
        "checkpoint_sha256",
        "cifar10_files_md5",
        "protocol",
        "test_count",
        "train_count",
    }
    if not isinstance(manifest, dict) or set(manifest) != required_manifest:
        raise SystemExit("prune-lab manifest schema is invalid")
    if (
        manifest["protocol"] != DENSE_PROTOCOL
        or int(manifest["train_count"]) != 50_000
        or int(manifest["test_count"]) != 10_000
    ):
        raise SystemExit("prune-lab manifest protocol/counts are invalid")
    checkpoint_sha256 = str(manifest["checkpoint_sha256"]).lower()
    if len(checkpoint_sha256) != 64 or any(
        ch not in "0123456789abcdef" for ch in checkpoint_sha256
    ):
        raise SystemExit("prune-lab manifest checkpoint digest is invalid")
    if checkpoint_sha256 != expected_checkpoint_sha256:
        raise SystemExit("prune-lab manifest does not match the trusted checkpoint digest")
    manifest_sha256 = _sha256(manifest_path)

    train_loader, test_loader = load_cifar10(args.data_root, args.batch,
                                             args.num_workers)
    chance = 1.0 / NUM_CLASSES
    n_train = len(train_loader.dataset)
    n_test = len(test_loader.dataset)
    print(
        f"DATA task={args.task_id} surface={args.surface} cifar10 "
        f"train={n_train} test={n_test} classes={NUM_CLASSES} "
        f"chance={chance:.3f} target_budget={args.sparsity} "
        f"manifest_sha256={manifest_sha256}",
        flush=True,
    )

    # ---- dense model: strictly load the fixed checkpoint ----
    set_all_seeds(args.seed)
    model = build_resnet18_cifar().to(device)
    init_sd: dict = {}
    teacher = None
    if not args.dense_ckpt or not os.path.isfile(args.dense_ckpt):
        raise SystemExit("pinned dense checkpoint is missing; runtime pretraining is forbidden")
    checkpoint_path = Path(args.dense_ckpt).resolve()
    if checkpoint_path.name != manifest["checkpoint"]:
        raise SystemExit("dense checkpoint path does not match the prepared manifest")
    if _sha256(checkpoint_path) != checkpoint_sha256:
        raise SystemExit("dense checkpoint SHA-256 does not match the prepared manifest")
    ck = torch.load(args.dense_ckpt, map_location=device)
    required_checkpoint_keys = {
        "state_dict", "early_state_dict", "dense_acc", "protocol", "train_epochs"
    }
    if not isinstance(ck, dict) or not required_checkpoint_keys.issubset(ck):
        raise SystemExit("dense checkpoint lacks required provenance/state")
    if ck["protocol"] != DENSE_PROTOCOL or int(ck["train_epochs"]) < 200:
        raise SystemExit("dense checkpoint does not match the 200-epoch protocol")
    dense_acc = float(ck["dense_acc"])
    if not math.isfinite(dense_acc) or not 0.0 <= dense_acc <= 1.0:
        raise SystemExit("dense checkpoint accuracy is invalid")
    model.load_state_dict(ck["state_dict"], strict=True)
    init_sd = ck["early_state_dict"]
    measured_dense_acc = accuracy(model, test_loader, device)
    if abs(measured_dense_acc - dense_acc) > 5e-4:
        raise SystemExit(
            "dense checkpoint accuracy metadata does not match full CIFAR-10 evaluation"
        )
    dense_acc = measured_dense_acc
    teacher = build_resnet18_cifar().to(device)
    teacher.load_state_dict(ck["state_dict"], strict=True)
    teacher.eval()
    print(
        f"DENSE_LOADED task={args.task_id} surface={args.surface} "
        f"protocol={DENSE_PROTOCOL} epochs={int(ck['train_epochs'])} "
        f"dense_acc={dense_acc:.4f} checkpoint_sha256={checkpoint_sha256}",
        flush=True,
    )

    params = prunable_params(model)
    nparams_dense = sum(p.numel() for _, p in params)
    dense_flops, _ = count_flops_params(model, device)
    print(
        f"MODEL task={args.task_id} surface={args.surface} "
        f"resnet18_cifar prunable_params={nparams_dense}",
        flush=True,
    )

    mod = load_surface(Path(args.solution))

    if args.surface in STRUCTURED_SURFACES:
        if args.surface == "structured_criterion":
            masks, pre = run_structured_criterion(
                mod, model, train_loader, test_loader, device, args.sparsity,
                args.recovery_epochs, args.seed)
        else:
            masks, pre = run_flops_budget(
                mod, model, train_loader, test_loader, device, args.flops_budget,
                args.recovery_epochs, args.seed)
        sp = 1.0 - (sum(p.numel() for p in model.parameters() if p.requires_grad)
                    / sum(p.numel() for p in build_resnet18_cifar().parameters()))
        flops, nparams = count_flops_params(model, device)
        if flops >= dense_flops:
            raise RuntimeError("structured pruning did not reduce measured MACs")
        if args.surface == "flops_budget" and flops > dense_flops * (args.flops_budget + 0.05):
            raise RuntimeError("structured result exceeds the enforced MAC budget")
    else:
        masks, pre = run_unstructured(
            mod, model, train_loader, test_loader, device, args.sparsity,
            args.seed, args.surface, args.recovery_epochs,
            teacher=teacher, init_sd=init_sd)
        sp = measured_sparsity(prunable_params(model), masks) if masks else 0.0
        flops, nparams = count_flops_params(model, device)
        if args.surface not in ("flops_budget",):
            # unstructured: report DENSE flops as context (sparse != smaller)
            flops, _ = count_flops_params(build_resnet18_cifar().to(device), device)

    acc = accuracy(model, test_loader, device)
    values = (acc, sp, dense_acc, pre, float(nparams), float(flops), chance)
    if not all(math.isfinite(float(value)) for value in values):
        raise RuntimeError("non-finite pruning result")
    print(f"PRUNE_METRICS protocol={DENSE_PROTOCOL} task={args.task_id} "
          f"surface={args.surface} setting={args.label} "
          f"acc={acc:.4f} sparsity={sp:.4f} dense_acc={dense_acc:.4f} "
          f"pruned_acc_prefinetune={pre:.4f} nparams={nparams} flops={flops} "
          f"dense_flops={dense_flops} flops_budget={args.flops_budget:.4f} "
          f"chance={chance:.4f} train={n_train} test={n_test} "
          f"recovery_epochs={args.recovery_epochs} seed={args.seed} "
          f"manifest_sha256={manifest_sha256} checkpoint_sha256={checkpoint_sha256}", flush=True)


if __name__ == "__main__":
    main()
