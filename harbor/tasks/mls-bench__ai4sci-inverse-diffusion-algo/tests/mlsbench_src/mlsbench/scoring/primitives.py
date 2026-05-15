"""Pure math primitives for score normalization.

Two normalization functions:
- bounded_power: metrics with a theoretical bound (accuracy, loss, FID, ...)
- sigmoid_score: metrics without a theoretical bound (reward, throughput, ...)

Plus constraint penalty functions for hard requirements (e.g., cost <= 25).
"""

from __future__ import annotations

import math
import warnings

GAMMA_MIN = 0.1
GAMMA_MAX = 10.0


# ---------------------------------------------------------------------------
# bounded_power
# ---------------------------------------------------------------------------

def bounded_power(x: float, floor: float, bound: float, gamma: float) -> float:
    """Normalize *x* into [0, 1] via a power curve between *floor* and *bound*.

    After direction unification (higher-is-better):

    * When ``bound`` is on the "better" side of ``floor`` (``bound > floor``) —
      the standard case used for metrics with a well-defined theoretical
      ceiling such as loss (``bound=0``) — the curve is::

          r = clip((x - floor) / (bound - floor), 0, 1)
          score = r ** gamma

      i.e. ``floor`` maps to score 0 and ``bound`` maps to score 1.

    * When ``bound`` is on the "worse" side of ``floor`` (``bound < floor``) —
      the pattern used for higher-is-better metrics where the spec anchors
      ``bound`` at a hard sanity floor such as random-guessing accuracy
      (``bound=25`` for arc_easy / hellaswag) — the score is inverted so that
      ``floor`` (best-baseline reference) still maps to 1 and ``bound``
      (random floor) maps to 0. Values worse than ``bound`` clip to 0
      instead of being silently inverted to 1.
    """
    if math.isnan(x):
        return 0.0
    if bound == floor:
        return 1.0
    if bound > floor:
        # Standard orientation: floor=worst, bound=best.
        r = (x - floor) / (bound - floor)
    else:
        # Inverted spec convention: bound is a hard sanity floor (worse side),
        # floor anchors the baseline reference.
        r = (x - bound) / (floor - bound)
    r = max(0.0, min(1.0, r))
    return r ** gamma


def solve_gamma(floor: float, bound: float, ref: float, ref_score: float) -> float:
    """Solve gamma such that ``bounded_power(ref, floor, bound, gamma) == ref_score``.

    Returns gamma clamped to [GAMMA_MIN, GAMMA_MAX].
    """
    if bound == floor:
        return 1.0
    # Mirror the inverted-spec convention used in ``bounded_power``.
    if bound > floor:
        r_ref = (ref - floor) / (bound - floor)
    else:
        r_ref = (ref - bound) / (floor - bound)
    r_ref = max(0.0, min(1.0, r_ref))
    # Degenerate: ref at floor or at bound
    if r_ref <= 0.0 or r_ref >= 1.0:
        warnings.warn(
            f"solve_gamma: r(ref)={r_ref:.4f} is degenerate (ref={ref}, "
            f"floor={floor}, bound={bound}). Falling back to gamma=1.",
            stacklevel=2,
        )
        return 1.0
    ref_score = max(1e-9, min(1.0 - 1e-9, ref_score))
    gamma = math.log(ref_score) / math.log(r_ref)
    gamma = max(GAMMA_MIN, min(GAMMA_MAX, gamma))
    return gamma


# ---------------------------------------------------------------------------
# sigmoid_score
# ---------------------------------------------------------------------------

def sigmoid_score(y: float, floor: float, scale: float) -> float:
    """Normalize *y* into [0, 1) via a shifted sigmoid.

    score(y) = 2 * sigma((y - floor) / scale) - 1   for y >= floor
    score(y) = 0                                      for y < floor

    Maps floor -> 0, approaches 1 as y -> +inf.
    """
    if math.isnan(y):
        return 0.0
    if y <= floor:
        return 0.0
    if scale <= 0:
        return 0.0
    z = (y - floor) / scale
    # Prevent overflow in exp
    if z > 30:
        return 1.0
    sig = 1.0 / (1.0 + math.exp(-z))
    return 2.0 * sig - 1.0


def solve_scale(floor: float, ref: float, ref_score: float) -> float:
    """Solve *scale* such that ``sigmoid_score(ref, floor, scale) == ref_score``.

    From: ref_score = 2 * sigma((ref - floor) / scale) - 1
    =>    sigma(z) = (ref_score + 1) / 2
    =>    z = logit((ref_score + 1) / 2)
    =>    scale = (ref - floor) / z
    """
    if ref <= floor:
        warnings.warn(
            f"solve_scale: ref={ref} <= floor={floor}. Using scale=1.0.",
            stacklevel=2,
        )
        return 1.0
    ref_score = max(0.05, min(0.95, ref_score))
    p = (ref_score + 1.0) / 2.0
    # logit(p) = log(p / (1-p))
    z = math.log(p / (1.0 - p))
    if z <= 0:
        return 1.0
    return (ref - floor) / z


# ---------------------------------------------------------------------------
# Constraint penalties
# ---------------------------------------------------------------------------

def penalty_upper(x: float, target: float, sharpness: float = 0.15) -> float:
    """Penalty for upper-bound constraint ``x <= target``.

    Returns 1.0 if satisfied, exponential decay otherwise.
    """
    if math.isnan(x):
        return 0.0
    if x <= target:
        return 1.0
    return math.exp(-sharpness * (x - target))


def penalty_lower(x: float, target: float, sharpness: float = 0.15) -> float:
    """Penalty for lower-bound constraint ``x >= target``.

    Returns 1.0 if satisfied, exponential decay otherwise.
    """
    if math.isnan(x):
        return 0.0
    if x >= target:
        return 1.0
    return math.exp(-sharpness * (target - x))


# ---------------------------------------------------------------------------
# Direction + transform helpers
# ---------------------------------------------------------------------------

TRANSFORMS = {
    "id": lambda x: x,
    "log": lambda x: math.log(max(x, 1e-30)),
    "log1p": lambda x: math.log1p(max(x, 0.0)),
}


def apply_direction_and_transform(
    x: float, direction: str, transform: str,
) -> float:
    """Unify raw metric to higher-is-better internal space.

    y = sign * transform(x)
    """
    if math.isnan(x):
        return float("nan")
    tfn = TRANSFORMS.get(transform)
    if tfn is None:
        raise ValueError(f"Unknown transform: {transform!r}")
    val = tfn(x)
    if direction == "lower":
        val = -val
    elif direction != "higher":
        raise ValueError(f"Unknown direction: {direction!r}")
    return val
