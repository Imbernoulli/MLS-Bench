"""Held-out data-generating-process module for mdn-density (NOT agent-visible).

This module lives OUTSIDE every path that is bind-mounted into the agent's
container (the vendored package dir, the per-task scaffold). It is imported
only by ``holdout/mdn-density/generate_data.py``, a host-side, one-off script
that pre-generates FIXED train/test samples for every (target, seed) actually
used by any shipped mdn-* task and writes ONLY the samples (never this source
file) into ``vendor/mdn-density/_mdn_data/``.

It holds the one secret that must never reach the agent's editable scope: the
EXACT forward-map formulas + hardcoded noise scales / rotation / covariance
constants for each of the four generating targets (inverse_sine / two_branch
/ spiral / rot_bimodal). Knowing these exact numbers would let an evaluated
model hand-derive the analytically-optimal conditional density p(y|x)
directly (e.g. "two_branch is y=+-sqrt(x-noise) so p(y|x) is a mixture of two
Gaussians centred at +-sqrt(x) with variance 0.04^2") and hardcode it,
scoring near-perfect held-out NLL WITHOUT training a real mixture density
network -- i.e. get a perfect score without doing the science.

The setting names are disclosed benchmark labels. ``inverse_sine`` is adapted
from Bishop's 1994 MDN inverse problem; ``two_branch``, ``spiral``, and
``rot_bimodal`` are local synthetic extensions. They are not claimed to be
canonical community datasets. Only exact generator code and numeric constants
are held out from the evaluated agent.

``_sample_inverse_sine`` / ``_sample_two_branch`` / ``_sample_spiral`` /
``_sample_rot_bimodal`` / ``TARGETS`` / ``TARGETS_2D`` are byte-identical to
the originals that used to live in ``vendor/mdn-density/common.py`` before
the oracle-leakage fix -- so the sampled train/test arrays, and therefore
every honest downstream result, are reproduced exactly.
"""
from __future__ import annotations

import math

import numpy as np

# ---------------------------------------------------------------------------
# Deterministic 1-D -> 1-D multimodal inverse targets
# ---------------------------------------------------------------------------
# We build a forward map t -> y (single-valued + noise), then swap axes so
# x := y_forward, y := t. Because the forward map is non-monotone, the inverse
# p(y|x) is one-to-many (multimodal). No download; fully synthetic & fixed.
# Byte-identical to the pre-fix vendor/mdn-density/common.py.

def _sample_inverse_sine(n: int, rng: np.random.Generator) -> np.ndarray:
    """Bishop's classic inverse problem.

    Forward: t ~ U(0,1);  yf = t + 0.3*sin(2*pi*t) + eps, eps~N(0,0.05^2).
    Then x := yf, y := t. For x in the middle of the range there are up to THREE
    valid t branches -> p(y|x) has up to 3 modes. This is *the* textbook MDN
    demo (Bishop, "Mixture Density Networks", 1994, Fig. 5).
    """
    t = rng.uniform(0.0, 1.0, size=n)
    yf = t + 0.3 * np.sin(2.0 * math.pi * t) + rng.normal(scale=0.05, size=n)
    x = yf.astype(np.float32)
    y = t.astype(np.float32)
    return np.stack([x, y], axis=1)


def _sample_two_branch(n: int, rng: np.random.Generator) -> np.ndarray:
    """A cleaner exactly-two-branch one-to-many map.

    Forward: t ~ U(-1,1); yf = t^2 + eps, eps~N(0,0.04^2). Inverse: for each
    x=yf>0 there are exactly two valid y = +-sqrt(x) branches -> strictly
    bimodal p(y|x). The conditional mean (~0) is never a valid value, so a point
    predictor / K=1 Gaussian is maximally bad here.
    """
    t = rng.uniform(-1.0, 1.0, size=n)
    yf = t * t + rng.normal(scale=0.04, size=n)
    x = yf.astype(np.float32)
    y = t.astype(np.float32)
    return np.stack([x, y], axis=1)


def _sample_spiral(n: int, rng: np.random.Generator) -> np.ndarray:
    """One-to-many spiral-angle map (multi-branch, up to ~3 modes).

    Forward: t ~ U(0,3); r = t; angle wraps -> yf = r*cos(3*t)+noise as the x,
    y := t. Multiple t map to a similar x, giving several conditional modes with
    input-dependent spacing (heteroscedastic multimodality).
    """
    t = rng.uniform(0.0, 3.0, size=n)
    yf = (t * np.cos(3.0 * t)) * 0.4 + rng.normal(scale=0.05, size=n)
    x = yf.astype(np.float32)
    y = (t / 3.0).astype(np.float32)   # scale target to ~[0,1]
    return np.stack([x, y], axis=1)


TARGETS = {
    "inverse_sine": _sample_inverse_sine,
    "two_branch": _sample_two_branch,
    "spiral": _sample_spiral,
}


# ---------------------------------------------------------------------------
# 2-D MULTIMODAL target (for the covariance / full-vs-diag surface)
# ---------------------------------------------------------------------------
# x in R^1 -> y in R^2. The conditional p(y|x) is a mixture of TWO Gaussians
# whose modes lie on ROTATED, CORRELATED ellipses, so a diagonal-covariance
# component (axis-aligned) fits worse than a full-covariance one.

def _sample_rot_bimodal(n: int, rng: np.random.Generator) -> np.ndarray:
    """1->2 map with two correlated, rotated Gaussian branches per x.

    For each x~U(-1,1): pick a branch b in {+1,-1}; the mode centre is
    (b*sqrt(|x|), b*x) and the local noise is drawn along a direction ROTATED by
    an x-dependent angle with anisotropic spread (major axis 0.15, minor 0.02).
    So each conditional mode is a tilted, highly-correlated ellipse: a diagonal
    Gaussian must inflate both axes to cover it (wasting mass), while a full
    covariance rotates to hug the ellipse -> strictly lower NLL.

    Returns array (n, 3): [x, y1, y2].
    """
    x = rng.uniform(-1.0, 1.0, size=n).astype(np.float32)
    b = rng.integers(0, 2, size=n) * 2 - 1                 # +-1 branch
    cx = b * np.sqrt(np.abs(x))
    cy = b * x
    theta = 1.2 * x + 0.6                                   # x-dependent tilt
    u = rng.normal(scale=0.15, size=n)                     # major axis
    v = rng.normal(scale=0.02, size=n)                     # minor axis
    y1 = cx + np.cos(theta) * u - np.sin(theta) * v
    y2 = cy + np.sin(theta) * u + np.cos(theta) * v
    return np.stack([x, y1.astype(np.float32), y2.astype(np.float32)], axis=1)


TARGETS_2D = {
    "rot_bimodal": _sample_rot_bimodal,
}


def make_dataset_arrays(target: str, n_train: int, n_test: int, seed: int) -> dict:
    """FIXED train/test RAW samples for a 1-D->1-D `target`, as plain numpy.

    Byte-identical sampling procedure to the pre-fix
    ``common.py:make_dataset`` (same RNG streams: train uses ``seed``, test
    uses ``seed + 10_000``), BEFORE the (public) x-standardization step -- so
    honest downstream results are unchanged once the agent-visible
    ``common.py`` re-applies that standardization. Returns a flat dict with
    train_raw / test_raw keys ready to ``np.savez``.
    """
    if target not in TARGETS:
        raise KeyError(f"unknown mdn-density target {target!r}; choose from {list(TARGETS)}")
    fn = TARGETS[target]
    rng_tr = np.random.default_rng(seed)
    rng_te = np.random.default_rng(seed + 10_000)
    tr = fn(n_train, rng_tr)
    te = fn(n_test, rng_te)
    return {"train_raw": tr, "test_raw": te}


def make_dataset_arrays_2d(target: str, n_train: int, n_test: int, seed: int) -> dict:
    """FIXED train/test RAW samples for a 1-D->2-D `target`, as plain numpy.

    Byte-identical sampling procedure to the pre-fix
    ``common.py:make_dataset_2d``. Returns a flat dict with train_raw /
    test_raw keys ready to ``np.savez``.
    """
    if target not in TARGETS_2D:
        raise KeyError(f"unknown mdn-density 2-D target {target!r}; choose from {list(TARGETS_2D)}")
    fn = TARGETS_2D[target]
    rng_tr = np.random.default_rng(seed)
    rng_te = np.random.default_rng(seed + 10_000)
    tr = fn(n_train, rng_tr)
    te = fn(n_test, rng_te)
    return {"train_raw": tr, "test_raw": te}
