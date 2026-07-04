"""Held-out scoring/data module for optimization-parity (NOT agent-visible).

This module lives OUTSIDE every path that is bind-mounted into the agent's
container (workspace package dir, task dir, data binds). It is imported only by
the host-side ``parser.py`` (native) and by the task-level ``mid_edit.py`` at
setup time (host) and by the Harbor verifier (tests/) — never by the
agent-editable ``custom_strategy.py``. It holds the hidden-secret sampler, the
parity labeling function, the test-set construction, and the metric, so the
agent's process can never reach the secret subset ``S`` or the held-out test
labels.

The function bodies that build the data (``sample_hidden_secrets``,
``parity_labels``, ``make_test_set``) and the training-pool generator are
BYTE-IDENTICAL to the originals that used to live in
``edits/custom_template.py`` — so the data, and therefore every honest result,
is reproduced exactly.

What the host gives the container (via the runner + mid_edit) is, per hidden
secret:
  * the *unlabeled* training x-pool — regenerated in-process by the runner from
    a public generator seed (binary noise carries no information about S); and
  * a bit-packed *label* vector ``y_train`` for that pool, pre-generated here
    (the only artifact that depends on S).
The held-out test labels ``y_test`` are NEVER shipped; the runner emits its
predictions on a deterministically-regenerated ``X_test`` and this module
regenerates ``y_test`` host-side to score them.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, replace

import numpy as np
import torch


# =====================================================================
# Benchmark configuration (mirror of the agent-visible TaskConfig; the
# secret-generation fields are byte-identical so secrets/data match exactly).
# =====================================================================
@dataclass(frozen=True)
class TaskConfig:
    n_features: int = 32
    secret_size: int = 8
    hidden_width: int = 512
    batch_size: int = 128
    max_steps: int = 30_000
    max_train_examples: int = 12_800_000
    num_hidden_secrets: int = 5
    num_orderings: int = 3
    test_set_size: int = 16_384
    log_interval: int = 250
    min_steps_before_stop: int = 1_000
    early_stop_acc: float = 0.999
    early_stop_windows: int = 4


def make_config(n_features=None, secret_size=None, smoke=False) -> TaskConfig:
    """Build a TaskConfig the same way the runner does (overrides + smoke).

    Keeps the host-side label/test-truth generation aligned with the exact
    config the container ran under.
    """
    config = TaskConfig()
    if smoke:
        config = replace(
            config,
            num_hidden_secrets=2,
            num_orderings=2,
            test_set_size=2_048,
            max_steps=4_000,
            log_interval=100,
            min_steps_before_stop=400,
            early_stop_windows=3,
        )
    if n_features is not None:
        config = replace(config, n_features=n_features)
    if secret_size is not None:
        config = replace(config, secret_size=secret_size)
    return config


# =====================================================================
# Hidden-secret sampler + parity labeling (byte-identical to original)
# =====================================================================

def sample_hidden_secrets(config, seed: int) -> list[tuple[int, ...]]:
    max_unique = math.comb(config.n_features, config.secret_size)
    if config.num_hidden_secrets > max_unique:
        raise ValueError("Requested more hidden secrets than unique subsets.")

    rng = random.Random(seed)
    seen: set[tuple[int, ...]] = set()
    secrets: list[tuple[int, ...]] = []
    while len(secrets) < config.num_hidden_secrets:
        secret = tuple(sorted(rng.sample(range(config.n_features), config.secret_size)))
        if secret not in seen:
            seen.add(secret)
            secrets.append(secret)
    return secrets


def parity_labels(x: torch.Tensor, secret: tuple[int, ...]) -> torch.Tensor:
    secret_index = torch.tensor(secret, dtype=torch.long)
    return (x.index_select(dim=1, index=secret_index).sum(dim=1).remainder(2)).to(
        torch.float32
    )


def make_test_set(
    secret: tuple[int, ...],
    config,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    x = torch.randint(
        low=0,
        high=2,
        size=(config.test_set_size, config.n_features),
        generator=generator,
        dtype=torch.int64,
    ).to(torch.float32)
    y = parity_labels(x, secret)
    return x, y


# =====================================================================
# Training x-pool generator (secret-free; identical draw to original
# make_dataset / baselines, which all share one freshly-seeded generator)
# =====================================================================

def gen_train_pool_x(config, train_dataset_seed: int) -> torch.Tensor:
    """Return the full unlabeled training x-pool for one secret.

    This is exactly the draw the original ``make_dataset`` / baselines made
    (``torch.Generator().manual_seed(train_dataset_seed)`` then
    ``torch.randint(0, 2, (num_examples, n_features))``), sized to
    ``max_train_examples``. Thanks to the prefix property of ``torch.randint``,
    any baseline that requested a smaller ``num_examples`` is reproduced
    byte-identically by slicing ``[:num_examples]`` of this pool. No secret is
    involved, so the runner regenerates this in-process inside the container.
    """
    generator = torch.Generator().manual_seed(train_dataset_seed)
    x = torch.randint(
        low=0,
        high=2,
        size=(config.max_train_examples, config.n_features),
        generator=generator,
        dtype=torch.int64,
    ).to(torch.float32)
    return x


# =====================================================================
# Host-side helpers: secret resolution, label pre-gen, test truth, metric
# =====================================================================

def secret_for(config, seed: int, secret_index: int) -> tuple[int, ...]:
    """Resolve the hidden secret for one (config, seed, secret_index).

    Uses the SAME ``seed + 17`` offset the runner used to sample secrets.
    Host-only — never returned to the container.
    """
    secrets = sample_hidden_secrets(config, seed + 17)
    return secrets[secret_index]


def gen_train_labels_packed(config, seed: int, secret_index: int) -> bytes:
    """Bit-pack the full training-pool labels for one (config, seed, secret).

    The x-pool is regenerated host-side with the same public seed the runner
    uses (``seed * 10_000 + secret_index``); the labels are ``parity_labels``
    over the hidden secret. Only the bit-packed labels leave this module (into
    the workspace), never the secret or the x-pool.
    """
    secret = secret_for(config, seed, secret_index)
    train_dataset_seed = seed * 10_000 + secret_index
    x = gen_train_pool_x(config, train_dataset_seed)
    y = parity_labels(x, secret).to(torch.uint8).numpy()
    return np.packbits(y).tobytes()


def truth(config, seed: int, secret_index: int, test_set_size: int = None) -> np.ndarray:
    """Return the held-out test labels ``y_test`` for the host-side scorer.

    The test inputs are drawn from a freshly-seeded generator, so by the prefix
    property of ``torch.randint`` the first ``m`` rows of the full-size draw equal
    an ``m``-row draw from the same seed. The host scorer therefore passes the
    number of predictions the runner actually emitted (``test_set_size``) so the
    regenerated labels line up exactly with the runner's test set, independent of
    whether the run used the default or smoke ``test_set_size``.
    """
    secret = secret_for(config, seed, secret_index)
    test_seed = seed * 20_000 + secret_index
    if test_set_size is None:
        test_set_size = config.test_set_size
    generator = torch.Generator().manual_seed(test_seed)
    x = torch.randint(
        low=0,
        high=2,
        size=(test_set_size, config.n_features),
        generator=generator,
        dtype=torch.int64,
    ).to(torch.float32)
    return parity_labels(x, secret).numpy()


def accuracy(preds: np.ndarray, y_test: np.ndarray) -> float:
    """Same accuracy as the original ``evaluate_accuracy``: threshold preds and
    test labels at 0.5 and take the fraction that match."""
    pred_bool = preds >= 0.5
    true_bool = y_test >= 0.5
    correct = int((pred_bool == true_bool).sum())
    total = int(true_bool.size)
    return correct / max(total, 1)
