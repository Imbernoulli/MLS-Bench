"""DSL for writing score_spec.py files.

Usage in a score_spec.py::

    from mlsbench.scoring.dsl import *

    term("my_metric",
        col("leaderboard_column")
        .lower()
        .log()
        .bounded_power(bound=0.0)
    )

    setting("env1", weighted_mean(("my_metric", 1.0)))
    task(gmean("env1"))
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mlsbench.scoring.spec import (
    AnchorRef,
    DEFAULT_REF_SCORE,
    SettingSpec,
    TaskScoreSpec,
    TermSpec,
)

__all__ = [
    "col",
    "term",
    "setting",
    "task",
    "weighted_mean",
    "gmean",
    "penalty_upper",
    "penalty_lower",
    "bl_worst",
    "bl_best",
    "const",
    "quick_spec",
]


# ---------------------------------------------------------------------------
# Registry (collects declarations during score_spec.py execution)
# ---------------------------------------------------------------------------

class _Registry:
    def __init__(self) -> None:
        self.terms: dict[str, TermSpec] = {}
        self.settings: dict[str, SettingSpec] = {}
        self.task_agg: str = "gmean"

    def to_task_spec(self) -> TaskScoreSpec:
        return TaskScoreSpec(
            terms=dict(self.terms),
            settings=dict(self.settings),
            task_agg=self.task_agg,
        )


_REGISTRY: _Registry = _Registry()


def _new_registry() -> _Registry:
    return _Registry()


# ---------------------------------------------------------------------------
# Anchor references
# ---------------------------------------------------------------------------

def bl_worst(metric: str) -> AnchorRef:
    """Symbolic reference to the worst baseline value for *metric*."""
    return AnchorRef(kind="bl_worst", metric=metric)


def bl_best(metric: str) -> AnchorRef:
    """Symbolic reference to the best baseline value for *metric*."""
    return AnchorRef(kind="bl_best", metric=metric)


def const(value: float) -> AnchorRef:
    """Concrete constant value."""
    return AnchorRef(kind="const", value=value)


# ---------------------------------------------------------------------------
# Column expression builder
# ---------------------------------------------------------------------------

class ColExpr:
    """Fluent builder for a metric term."""

    def __init__(self, metric: str) -> None:
        self._metric = metric
        self._direction: str = "higher"
        self._transform: str = "id"

    def higher(self) -> ColExpr:
        self._direction = "higher"
        return self

    def lower(self) -> ColExpr:
        self._direction = "lower"
        return self

    def id(self) -> ColExpr:
        self._transform = "id"
        return self

    def log(self) -> ColExpr:
        self._transform = "log"
        return self

    def log1p(self) -> ColExpr:
        self._transform = "log1p"
        return self

    def bounded_power(
        self,
        bound: float | AnchorRef,
        ref: float | AnchorRef | None = None,
        ref_score: float = DEFAULT_REF_SCORE,
    ) -> TermSpec:
        """Create a bounded_power term spec."""
        bound_val = bound.value if isinstance(bound, AnchorRef) and bound.kind == "const" else bound
        return TermSpec(
            name="",  # filled by term()
            metric=self._metric,
            role="objective",
            direction=self._direction,
            transform=self._transform,
            norm_type="bounded_power",
            bound=float(bound_val) if isinstance(bound_val, (int, float)) else None,
            ref=ref,
            ref_score=ref_score,
        )

    def sigmoid(
        self,
        ref: float | AnchorRef | None = None,
        ref_score: float = DEFAULT_REF_SCORE,
        scale: float | None = None,
    ) -> TermSpec:
        """Create a sigmoid term spec."""
        return TermSpec(
            name="",  # filled by term()
            metric=self._metric,
            role="objective",
            direction=self._direction,
            transform=self._transform,
            norm_type="sigmoid",
            ref=ref,
            ref_score=ref_score,
            scale=scale,
        )


def col(metric: str) -> ColExpr:
    """Start building a term from leaderboard column *metric*."""
    return ColExpr(metric)


# ---------------------------------------------------------------------------
# Constraint helpers
# ---------------------------------------------------------------------------

def penalty_upper(
    col_expr: ColExpr,
    target: float,
    sharpness: float = 0.15,
) -> TermSpec:
    """Create an upper-bound constraint term (x <= target)."""
    return TermSpec(
        name="",
        metric=col_expr._metric,
        role="constraint",
        direction=col_expr._direction,
        transform=col_expr._transform,
        norm_type="penalty_upper",
        constraint_target=target,
        constraint_sharpness=sharpness,
    )


def penalty_lower(
    col_expr: ColExpr,
    target: float,
    sharpness: float = 0.15,
) -> TermSpec:
    """Create a lower-bound constraint term (x >= target)."""
    return TermSpec(
        name="",
        metric=col_expr._metric,
        role="constraint",
        direction=col_expr._direction,
        transform=col_expr._transform,
        norm_type="penalty_lower",
        constraint_target=target,
        constraint_sharpness=sharpness,
    )


# ---------------------------------------------------------------------------
# Registration functions
# ---------------------------------------------------------------------------

def term(name: str, spec: TermSpec) -> None:
    """Register a named term in the current score spec."""
    spec.name = name
    _REGISTRY.terms[name] = spec


# ---------------------------------------------------------------------------
# Aggregation expressions
# ---------------------------------------------------------------------------

@dataclass
class _WeightedMeanExpr:
    items: list[tuple[str, float]]


@dataclass
class _GmeanExpr:
    setting_names: list[str]


def weighted_mean(*term_weights: tuple[str, float]) -> _WeightedMeanExpr:
    """Weighted average of objective terms within a setting."""
    return _WeightedMeanExpr(items=list(term_weights))


def gmean(*setting_names: str) -> _GmeanExpr:
    """Geometric mean across settings."""
    return _GmeanExpr(setting_names=list(setting_names))


def setting(
    name: str,
    agg: _WeightedMeanExpr,
    constraints: list[str] | None = None,
) -> None:
    """Register a named setting in the current score spec."""
    _REGISTRY.settings[name] = SettingSpec(
        name=name,
        terms=agg.items,
        constraints=constraints or [],
    )


def task(agg: _GmeanExpr) -> None:
    """Register the task-level aggregation."""
    _REGISTRY.task_agg = "gmean"
    # Ensure all referenced settings exist
    for sn in agg.setting_names:
        if sn not in _REGISTRY.settings:
            raise ValueError(f"task() references undefined setting '{sn}'")


# ---------------------------------------------------------------------------
# quick_spec: shortcut for common patterns
# ---------------------------------------------------------------------------

def quick_spec(
    metrics: dict[str, dict[str, Any]],
    settings_from: str = "labels",
    task_agg: str = "gmean",
) -> None:
    """Generate a full spec from a compact declaration.

    *metrics* maps a pattern (possibly with ``*`` wildcard) to a dict with:
      - direction: "higher" | "lower"
      - norm: "bounded" | "sigmoid"
      - bound: float (for bounded)
      - ref: float (optional, auto from current best baseline if omitted)
      - ref_score: float (default 0.5)
      - weight: float (default 1.0)
      - role: "objective" | "constraint" | "drop" (default "objective")

    When ``settings_from="labels"``, the wildcard ``*`` in the metric key
    is expanded at evaluation time against actual leaderboard columns,
    and the matched suffix is used as the setting name.

    This function is expanded by the evaluator, not at spec load time.
    It stores the raw declaration for the evaluator to process.
    """
    # Store the quick_spec data directly in the registry for later expansion
    _REGISTRY._quick_spec = {  # type: ignore[attr-defined]
        "metrics": metrics,
        "settings_from": settings_from,
        "task_agg": task_agg,
    }
