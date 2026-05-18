# Custom inner-loop optimizer for gradient-based meta-learning
#
# EDITABLE section: InnerLoopOptimizer class and helper modules.
# FIXED sections: everything else (config, data loading, backbone, outer loop, evaluation).
#
# Research question: Design the inner-loop adaptation algorithm that determines
# HOW model parameters are updated during fast adaptation to a new task.
import os
import sys
import copy
import random
from statistics import mean
from typing import Optional, Tuple, Dict, List

# Fix import path: exclude the learn2learn source tree so that
# ``import learn2learn`` resolves to the pip-installed package,
# not the source checkout at /workspace/learn2learn/learn2learn/.
_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path = [p for p in sys.path if os.path.abspath(p) != os.path.abspath(_script_dir)]

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import learn2learn as l2l
from learn2learn.data.transforms import NWays, KShots, LoadData, RemapLabels, ConsecutiveLabels
from torchvision import transforms as tv_transforms


# =====================================================================
# FIXED: Configuration
# =====================================================================
SEED = int(os.environ.get("SEED", "42"))
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./output")
SETTING = os.environ.get("ENV", "mini_imagenet_5shot")

# Parse setting: dataset_Nshot
_parts = SETTING.rsplit("_", 1)
DATASET_NAME = _parts[0]            # e.g. "mini_imagenet" or "cifar_fs"
N_SHOT = int(_parts[1].replace("shot", ""))  # e.g. 1 or 5

# Few-shot settings
N_WAY = 5
N_QUERY = 15
IMAGE_SIZE = 84
HIDDEN_SIZE = 64  # CNN4 channel width

# Training settings — 1-shot converges slower, use 4x iterations.
N_META_ITERS = 60000 if N_SHOT == 1 else 15000
META_BATCH_SIZE = 4        # tasks per meta-update
INNER_STEPS_TRAIN = 5      # adaptation steps during training
INNER_STEPS_TEST = 10      # adaptation steps during evaluation (more steps at test)
META_LR = 0.003            # outer-loop learning rate
INNER_LR = 0.5             # default inner-loop learning rate

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EVAL_INTERVAL = 500        # meta-iterations between evaluations
N_EVAL_TASKS = 200         # tasks for validation
N_TEST_TASKS = 600         # tasks for final test


# =====================================================================
# FIXED: Dataset loading via learn2learn benchmarks
# =====================================================================
def get_tasksets(dataset_name: str, n_way: int, n_shot: int, n_query: int,
                 root: str = os.environ.get("L2L_DATA_ROOT", "/workspace/l2l_data")):
    """Create train/val/test TaskDataset objects using learn2learn."""
    total_samples = n_shot + n_query

    if dataset_name == "mini_imagenet":
        dataset_cls = l2l.vision.datasets.MiniImagenet
    elif dataset_name == "cifar_fs":
        dataset_cls = l2l.vision.datasets.CIFARFS
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # CIFAR-FS returns PIL Images (32x32) — needs ToTensor + Resize to 84x84.
    # MiniImagenet already returns [3,84,84] tensors — no transform needed.
    if dataset_name == "cifar_fs":
        img_transform = tv_transforms.Compose([
            tv_transforms.Resize((84, 84)),
            tv_transforms.ToTensor(),
        ])
    else:
        img_transform = None

    splits = {}
    for mode in ["train", "validation", "test"]:
        ds = dataset_cls(root=root, mode=mode, download=False,
                         transform=img_transform)
        meta_ds = l2l.data.MetaDataset(ds)
        transforms = [
            NWays(meta_ds, n_way),
            KShots(meta_ds, total_samples),
            LoadData(meta_ds),
            RemapLabels(meta_ds),
            ConsecutiveLabels(meta_ds),
        ]
        splits[mode] = l2l.data.TaskDataset(meta_ds, transforms, num_tasks=-1)

    return splits["train"], splits["validation"], splits["test"]


def split_support_query(data: Tensor, labels: Tensor, n_way: int, n_shot: int):
    """Split a task batch into support and query sets.

    Args:
        data: images [n_way * (n_shot + n_query), C, H, W]
        labels: labels [n_way * (n_shot + n_query)]
        n_way: number of classes
        n_shot: number of support examples per class

    Returns:
        support_x, support_y, query_x, query_y
    """
    sort_idx = torch.sort(labels).indices
    data = data[sort_idx]
    labels = labels[sort_idx]

    n_query_per_class = len(labels) // n_way - n_shot
    support_idx = []
    query_idx = []
    for cls in range(n_way):
        start = cls * (n_shot + n_query_per_class)
        support_idx.extend(range(start, start + n_shot))
        query_idx.extend(range(start + n_shot, start + n_shot + n_query_per_class))

    return data[support_idx], labels[support_idx], data[query_idx], labels[query_idx]


# =====================================================================
# FIXED: CNN4 Backbone (shared by all methods)
# =====================================================================
def make_model(n_way: int, hidden_size: int = HIDDEN_SIZE) -> nn.Module:
    """Create a CNN4 model for few-shot classification.

    Returns a CNN4 model with:
    - 4 convolutional blocks (each: Conv2d -> BN -> ReLU -> MaxPool)
    - hidden_size channels per block (default 64)
    - A linear classifier head mapping features to n_way classes
    - Input: [B, 3, 84, 84], Output: [B, n_way]

    The feature dimension before the head is hidden_size * 5 * 5 = 1600.
    """
    return l2l.vision.models.CNN4(
        output_size=n_way,
        hidden_size=hidden_size,
        embedding_size=hidden_size * 5 * 5,
    )


FEATURE_DIM = HIDDEN_SIZE * 5 * 5  # 1600 for CNN4 with hidden_size=64


# =====================================================================
# FIXED: Utility functions
# =====================================================================
def accuracy(predictions: Tensor, targets: Tensor) -> float:
    """Compute classification accuracy."""
    return (predictions.argmax(dim=1) == targets).float().mean().item()


def compute_loss_and_acc(model: nn.Module, data: Tensor, labels: Tensor):
    """Compute cross-entropy loss and accuracy for a batch."""
    logits = model(data)
    loss = F.cross_entropy(logits, labels)
    acc = accuracy(logits, labels)
    return loss, acc


# =====================================================================
# EDITABLE: Inner-Loop Optimizer for Gradient-Based Meta-Learning
# =====================================================================
class InnerLoopOptimizer:
    """Inner-loop adaptation algorithm for gradient-based meta-learning.

    This class defines HOW model parameters are updated during fast adaptation
    to a new task. The outer loop (meta-optimizer) is fixed; only this inner
    loop is editable.

    The default implementation is vanilla MAML: simple SGD with a fixed
    learning rate applied to all parameters.

    You may redesign:
    - Per-parameter or per-layer learning rates (Meta-SGD)
    - Which parameters to adapt (full model vs. head-only, ANIL)
    - Preconditioning / curvature information (Meta-Curvature)
    - Momentum, second-order corrections, or learned update rules
    - Any combination of the above

    Interface contract:
    - __init__(model, inner_lr): initialize the optimizer, may create
      learnable parameters that will be meta-learned by the outer loop
    - adapt(model, support_x, support_y, n_steps): perform n_steps of
      inner-loop adaptation on the support set, return adapted model
    - meta_parameters(): return any learnable parameters of the optimizer
      itself (e.g., per-parameter learning rates) for outer-loop optimization

    IMPORTANT:
    - The model passed to adapt() is a clone (via l2l.clone_module).
      You must use differentiable operations so gradients flow to the outer loop.
    - Use l2l.algorithms.maml.maml_update(model, lr, grads) or manual
      parameter updates. Do NOT use torch.optim optimizers (they break the
      computational graph).
    - meta_parameters() must return all learnable optimizer state so the
      outer loop can optimize them.
    """

    def __init__(self, model: nn.Module, inner_lr: float = INNER_LR):
        """Initialize the inner-loop optimizer.

        Args:
            model: the base model (used to inspect parameter shapes/counts).
                   Do NOT store a reference to this model — a fresh clone
                   is passed to adapt() each time.
            inner_lr: default inner-loop learning rate.
        """
        self.inner_lr = inner_lr

    def adapt(self, model: nn.Module, support_x: Tensor, support_y: Tensor,
              n_steps: int) -> nn.Module:
        """Perform inner-loop adaptation.

        Args:
            model: a CLONED model (via l2l.clone_module) — safe to modify in-place.
            support_x: support images [n_way * n_shot, C, H, W]
            support_y: support labels [n_way * n_shot]
            n_steps: number of inner-loop gradient steps

        Returns:
            The adapted model (may be the same object, modified in-place).
        """
        model.train()
        for _ in range(n_steps):
            loss = F.cross_entropy(model(support_x), support_y)
            grads = torch.autograd.grad(
                loss, model.parameters(), create_graph=True
            )
            # Vanilla SGD update using learn2learn's differentiable update
            model = l2l.algorithms.maml.maml_update(
                model, lr=self.inner_lr, grads=grads
            )
        return model

    def meta_parameters(self) -> List[Tensor]:
        """Return learnable parameters of the optimizer for outer-loop training.

        For vanilla MAML, the inner LR is fixed, so this returns [].
        For Meta-SGD, this would return the per-parameter learning rates.
        """
        return []


# =====================================================================
# FIXED: Meta-Training and Evaluation Loop
# =====================================================================
def meta_train_step(model, inner_opt, meta_optimizer,
                    taskset, n_way, n_shot, n_query, meta_batch_size,
                    inner_steps, device):
    """One meta-training iteration: sample tasks, adapt, compute meta-loss."""
    meta_train_loss = 0.0
    meta_train_acc = 0.0

    for _ in range(meta_batch_size):
        # Clone model for this task
        learner = l2l.clone_module(model)

        # Sample a task
        task_data = taskset.sample()
        data, labels = task_data
        data, labels = data.to(device), labels.to(device)

        # Split into support / query
        support_x, support_y, query_x, query_y = split_support_query(
            data, labels, n_way, n_shot
        )

        # Inner-loop adaptation (uses the shared inner_opt instance)
        learner = inner_opt.adapt(learner, support_x, support_y, inner_steps)

        # Evaluate on query set (for meta-gradient)
        loss, acc = compute_loss_and_acc(learner, query_x, query_y)
        meta_train_loss += loss
        meta_train_acc += acc

    meta_train_loss /= meta_batch_size
    meta_train_acc /= meta_batch_size

    # Meta-update
    meta_optimizer.zero_grad()
    meta_train_loss.backward()
    meta_optimizer.step()

    return meta_train_loss.item(), meta_train_acc


def meta_evaluate(model, inner_opt, taskset,
                  n_way, n_shot, n_query, n_tasks, inner_steps, device):
    """Evaluate on a set of tasks."""
    accs = []
    for _ in range(n_tasks):
        learner = l2l.clone_module(model)

        task_data = taskset.sample()
        data, labels = task_data
        data, labels = data.to(device), labels.to(device)

        support_x, support_y, query_x, query_y = split_support_query(
            data, labels, n_way, n_shot
        )

        learner = inner_opt.adapt(learner, support_x, support_y, inner_steps)

        with torch.no_grad():
            _, acc = compute_loss_and_acc(learner, query_x, query_y)
        accs.append(acc)

    mean_acc = np.mean(accs)
    ci95 = 1.96 * np.std(accs) / np.sqrt(len(accs))
    return mean_acc, ci95


# =====================================================================
# FIXED: Main Script
# =====================================================================
if __name__ == "__main__":
    # Reproducibility
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Dataset: {DATASET_NAME}, N-way: {N_WAY}, N-shot: {N_SHOT}, Seed: {SEED}", flush=True)
    print(f"Setting: {SETTING}", flush=True)
    print(f"Meta-LR: {META_LR}, Inner-LR: {INNER_LR}", flush=True)
    print(f"Inner steps train/test: {INNER_STEPS_TRAIN}/{INNER_STEPS_TEST}", flush=True)

    # Load tasksets
    train_tasks, val_tasks, test_tasks = get_tasksets(
        DATASET_NAME, N_WAY, N_SHOT, N_QUERY
    )

    # Build model
    model = make_model(N_WAY).to(DEVICE)

    # ── FIXED: Parameter count check ────────────────────────────────
    # Budget: CNN4 model (~112K) + inner-loop optimizer learnable params
    # Meta-SGD adds one scalar per parameter (~112K extra).
    # Budget is 1.2x of (model params + Meta-SGD optimizer params).
    _model_params = sum(p.numel() for p in model.parameters())
    _optimizer_budget = _model_params  # Meta-SGD needs one LR per param
    _budget = int((_model_params + _optimizer_budget) * 1.2)

    # Create inner-loop optimizer (persistent across all iterations)
    inner_opt = InnerLoopOptimizer(model, INNER_LR)
    _opt_params = sum(p.numel() for p in inner_opt.meta_parameters())
    _total_params = _model_params + _opt_params
    print(f"Model params: {_model_params:,}, Optimizer params: {_opt_params:,}, "
          f"Total: {_total_params:,} (budget: {_budget:,})", flush=True)
    # ────────────────────────────────────────────────────────────────

    # Collect all meta-learnable parameters: model params + optimizer params
    all_meta_params = list(model.parameters()) + list(inner_opt.meta_parameters())
    meta_optimizer = torch.optim.Adam(all_meta_params, lr=META_LR)

    # Meta-training loop
    best_val_acc = 0.0
    best_state = copy.deepcopy(model.state_dict())
    best_inner_meta_state = [
        p.detach().clone() for p in inner_opt.meta_parameters()
    ]

    for iteration in range(1, N_META_ITERS + 1):
        model.train()
        train_loss, train_acc = meta_train_step(
            model, inner_opt, meta_optimizer,
            train_tasks, N_WAY, N_SHOT, N_QUERY, META_BATCH_SIZE,
            INNER_STEPS_TRAIN, DEVICE,
        )

        if iteration % EVAL_INTERVAL == 0:
            model.eval()
            val_acc, val_ci = meta_evaluate(
                model, inner_opt, val_tasks,
                N_WAY, N_SHOT, N_QUERY, N_EVAL_TASKS,
                INNER_STEPS_TEST, DEVICE,
            )
            print(
                f"TRAIN_METRICS iter={iteration} "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"val_acc={val_acc:.4f} val_ci95={val_ci:.4f}",
                flush=True,
            )
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = copy.deepcopy(model.state_dict())
                best_inner_meta_state = [
                    p.detach().clone() for p in inner_opt.meta_parameters()
                ]
                print(f"  New best val accuracy: {val_acc:.4f} +/- {val_ci:.4f}", flush=True)

    # Load best model and evaluate on test set
    model.load_state_dict(best_state)
    with torch.no_grad():
        for p, saved in zip(inner_opt.meta_parameters(), best_inner_meta_state):
            p.copy_(saved.to(device=p.device, dtype=p.dtype))
    model.eval()
    TEST_RNG_SEED = 0xBEEF
    random.seed(TEST_RNG_SEED)
    np.random.seed(TEST_RNG_SEED)
    torch.manual_seed(TEST_RNG_SEED)
    torch.cuda.manual_seed_all(TEST_RNG_SEED)
    test_acc, test_ci = meta_evaluate(
        model, inner_opt, test_tasks,
        N_WAY, N_SHOT, N_QUERY, N_TEST_TASKS,
        INNER_STEPS_TEST, DEVICE,
    )
    print(f"TEST_METRICS accuracy={test_acc:.4f} ci95={test_ci:.4f}", flush=True)
    print(f"Test accuracy: {100 * test_acc:.2f}% +/- {100 * test_ci:.2f}%", flush=True)
