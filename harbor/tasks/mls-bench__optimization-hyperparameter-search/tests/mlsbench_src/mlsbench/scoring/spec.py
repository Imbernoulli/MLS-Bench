"""Data model for score specifications and loader for score_spec.py files."""

from __future__ import annotations

import importlib.util
import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_REF_SCORE = 0.5
META_COLS = {"timestamp", "model", "is_final", "seed"}
NON_METRIC_COLS = {
    "baseline",
    "workload",
    "budget",
    "regime",
    "trace_mode",
    "n_prompts",
    "budget_scale",
}
NON_METRIC_PREFIXES = ("elapsed_", "n_samples", "n_prompts")


# ---------------------------------------------------------------------------
# Symbolic anchor references (resolved at evaluation time)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AnchorRef:
    """Symbolic reference to a baseline anchor value, resolved at eval time."""
    kind: str          # "bl_worst", "bl_best", "const"
    metric: str = ""   # leaderboard column name (for bl_* kinds)
    value: float = 0.0 # concrete value (for "const" kind)


# ---------------------------------------------------------------------------
# Spec dataclasses
# ---------------------------------------------------------------------------

@dataclass
class TermSpec:
    """Specification for a single metric term."""
    name: str
    metric: str                           # leaderboard column name
    role: str = "objective"               # "objective" | "constraint" | "drop"
    direction: str = "higher"             # "higher" | "lower"
    transform: str = "id"                 # "id" | "log" | "log1p"
    norm_type: str = "sigmoid"            # "bounded_power" | "sigmoid"
    # bounded_power params
    bound: float | None = None            # theoretical bound (in raw space)
    # shared calibration
    ref: float | AnchorRef | None = None  # reference value or symbolic ref
    ref_score: float = DEFAULT_REF_SCORE # target score at ref
    scale: float | None = None            # direct scale for sigmoid
    # constraint params
    constraint_target: float | None = None
    constraint_sharpness: float = 0.15


@dataclass
class SettingSpec:
    """Specification for one evaluation setting (env / dataset / bench)."""
    name: str
    terms: list[tuple[str, float]] = field(default_factory=list)  # (term_name, weight)
    constraints: list[str] = field(default_factory=list)           # term_names


@dataclass
class TaskScoreSpec:
    """Full score specification for a task."""
    terms: dict[str, TermSpec] = field(default_factory=dict)
    settings: dict[str, SettingSpec] = field(default_factory=dict)
    task_agg: str = "gmean"


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_score_spec(task_dir: Path) -> TaskScoreSpec | None:
    """Execute ``score_spec.py`` in *task_dir* and return the collected spec.

    Returns None if no score_spec.py exists.
    """
    spec_path = Path(task_dir) / "score_spec.py"
    if not spec_path.exists():
        return None

    # Import the DSL module to set up the registry
    from mlsbench.scoring import dsl as dsl_mod

    registry = dsl_mod._new_registry()
    prev = dsl_mod._REGISTRY
    dsl_mod._REGISTRY = registry
    try:
        mod_spec = importlib.util.spec_from_file_location(
            f"score_spec_{task_dir.name}", spec_path,
        )
        module = importlib.util.module_from_spec(mod_spec)
        mod_spec.loader.exec_module(module)
    finally:
        dsl_mod._REGISTRY = prev

    return registry.to_task_spec()


def validate_score_spec(
    spec: TaskScoreSpec,
    available_metrics: list[str],
) -> list[str]:
    """Return a list of warnings about the spec."""
    warns: list[str] = []
    avail = set(available_metrics)
    for tname, tspec in spec.terms.items():
        if tspec.role == "drop":
            continue
        if tspec.metric not in avail:
            warns.append(f"Term '{tname}': metric '{tspec.metric}' not found in leaderboard")
        if tspec.norm_type == "bounded_power" and tspec.bound is None:
            warns.append(f"Term '{tname}': bounded_power requires 'bound'")

    # Check all term references in settings exist
    defined_terms = set(spec.terms.keys())
    for sname, sspec in spec.settings.items():
        for term_name, _ in sspec.terms:
            if term_name not in defined_terms:
                warns.append(f"Setting '{sname}': references undefined term '{term_name}'")
        for cname in sspec.constraints:
            if cname not in defined_terms:
                warns.append(f"Setting '{sname}': references undefined constraint '{cname}'")

    return warns


def leaderboard_declared_metrics(task_dir: Path) -> list[str]:
    """Return metric-like columns declared in leaderboard.csv's header."""
    lb_path = Path(task_dir) / "leaderboard.csv"
    if not lb_path.exists():
        return []
    try:
        with lb_path.open(newline="") as f:
            header = next(csv.reader(f), [])
    except StopIteration:
        return []

    out: list[str] = []
    for col in header:
        if col in META_COLS or col in NON_METRIC_COLS:
            continue
        if col.startswith(NON_METRIC_PREFIXES):
            continue
        if col.endswith("_std"):
            continue
        out.append(col)
    return out
