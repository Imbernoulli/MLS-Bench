"""Data model for score specifications and loader for score_spec.py files."""

from __future__ import annotations

import importlib.util
import csv
import math
from dataclasses import dataclass, field
from pathlib import Path

from mlsbench.scoring._numeric import is_finite_real

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


def _is_valid_anchor(value: object) -> bool:
    """Return whether *value* has a well-formed anchor representation."""
    if value is None:
        return True
    if not isinstance(value, AnchorRef):
        return is_finite_real(value)
    if not isinstance(value.kind, str) or not isinstance(value.metric, str):
        return False
    if value.kind == "const":
        return value.metric == "" and is_finite_real(value.value)
    if value.kind in ("bl_worst", "bl_best"):
        return (
            bool(value.metric)
            and is_finite_real(value.value)
            and float(value.value) == 0.0
        )
    return False


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
    # explicit floor (generic / baseline-free anchor). When set, overrides the
    # baseline-derived worst anchor in _score_setting. Accepts a raw float or a
    # symbolic AnchorRef (e.g. const(...)). None -> fall back to worst baseline.
    floor: float | AnchorRef | None = None
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
    """Return fail-closed validation errors for a score specification."""
    errors: list[str] = []
    avail = {
        metric for metric in available_metrics
        if isinstance(metric, str)
    }

    for tname, tspec in spec.terms.items():
        if not isinstance(tname, str) or not tname:
            errors.append(f"Term name must be a nonempty string, got {tname!r}")
        if not isinstance(tspec, TermSpec):
            errors.append(f"Term '{tname}': declaration must be a TermSpec")
            continue
        if not isinstance(tspec.role, str) or tspec.role not in (
            "objective", "constraint", "drop",
        ):
            errors.append(f"Term '{tname}': invalid role '{tspec.role}'")
        metric_valid = isinstance(tspec.metric, str) and bool(tspec.metric)
        if not metric_valid:
            errors.append(f"Term '{tname}': metric must be a nonempty string")
        if tspec.role == "drop":
            continue
        if metric_valid and tspec.metric not in avail:
            errors.append(f"Term '{tname}': metric '{tspec.metric}' not found in leaderboard")
        if not isinstance(tspec.direction, str) or tspec.direction not in (
            "higher", "lower",
        ):
            errors.append(f"Term '{tname}': invalid direction '{tspec.direction}'")
        if not isinstance(tspec.transform, str) or tspec.transform not in (
            "id", "log", "log1p",
        ):
            errors.append(f"Term '{tname}': invalid transform '{tspec.transform}'")

        if tspec.role == "constraint":
            if not isinstance(tspec.norm_type, str) or tspec.norm_type not in (
                "penalty_upper", "penalty_lower",
            ):
                errors.append(
                    f"Term '{tname}': constraint has invalid norm '{tspec.norm_type}'"
                )
            if not is_finite_real(tspec.constraint_target):
                errors.append(f"Term '{tname}': constraint target must be finite")
            if (
                not is_finite_real(tspec.constraint_sharpness)
                or float(tspec.constraint_sharpness) <= 0.0
            ):
                errors.append(
                    f"Term '{tname}': constraint sharpness must be finite and > 0"
                )
            continue

        if tspec.role != "objective":
            continue
        if not isinstance(tspec.norm_type, str) or tspec.norm_type not in (
            "bounded_power", "sigmoid",
        ):
            errors.append(f"Term '{tname}': invalid objective norm '{tspec.norm_type}'")
        if tspec.norm_type == "bounded_power":
            if tspec.bound is None:
                errors.append(f"Term '{tname}': bounded_power requires 'bound'")
            elif not is_finite_real(tspec.bound):
                errors.append(f"Term '{tname}': bounded_power bound must be finite")
        if not _is_valid_anchor(tspec.floor):
            errors.append(f"Term '{tname}': floor anchor is invalid or non-finite")
        if not _is_valid_anchor(tspec.ref):
            errors.append(f"Term '{tname}': reference anchor is invalid or non-finite")
        if (
            not is_finite_real(tspec.ref_score)
            or not 0.0 < float(tspec.ref_score) < 1.0
        ):
            errors.append(f"Term '{tname}': ref_score must be finite and in (0, 1)")
        if tspec.scale is not None and (
            not is_finite_real(tspec.scale) or float(tspec.scale) <= 0.0
        ):
            errors.append(f"Term '{tname}': scale must be finite and > 0")

    # Check all term references in settings exist
    defined_terms = {
        name for name, term_spec in spec.terms.items()
        if isinstance(name, str) and isinstance(term_spec, TermSpec)
    }
    if not spec.settings:
        errors.append("Score spec declares no settings")
    for sname, sspec in spec.settings.items():
        if not isinstance(sname, str) or not sname:
            errors.append(f"Setting name must be a nonempty string, got {sname!r}")
        if not isinstance(sspec, SettingSpec):
            errors.append(f"Setting '{sname}': declaration must be a SettingSpec")
            continue
        if not isinstance(sspec.terms, (list, tuple)) or not sspec.terms:
            errors.append(f"Setting '{sname}': must declare at least one objective")
            weighted_items: list | tuple = []
        else:
            weighted_items = sspec.terms
        total_weight = 0.0
        for item in weighted_items:
            if not isinstance(item, (tuple, list)) or len(item) != 2:
                errors.append(f"Setting '{sname}': invalid weighted objective entry {item!r}")
                continue
            term_name, weight = item
            if not isinstance(term_name, str) or not term_name:
                errors.append(
                    f"Setting '{sname}': objective name must be a nonempty string"
                )
            elif term_name not in defined_terms:
                errors.append(f"Setting '{sname}': references undefined term '{term_name}'")
            elif spec.terms[term_name].role != "objective":
                errors.append(
                    f"Setting '{sname}': weighted term '{term_name}' is not an objective"
                )
            if not is_finite_real(weight) or float(weight) <= 0.0:
                errors.append(
                    f"Setting '{sname}': weight for '{term_name}' must be finite and > 0"
                )
            else:
                total_weight += float(weight)
        if not math.isfinite(total_weight) or total_weight <= 0.0:
            errors.append(f"Setting '{sname}': objective weights must have a finite positive total")
        if not isinstance(sspec.constraints, (list, tuple)):
            errors.append(f"Setting '{sname}': constraints must be a list or tuple")
            constraint_names: list | tuple = []
        else:
            constraint_names = sspec.constraints
        for cname in constraint_names:
            if not isinstance(cname, str) or not cname:
                errors.append(
                    f"Setting '{sname}': constraint name must be a nonempty string"
                )
            elif cname not in defined_terms:
                errors.append(f"Setting '{sname}': references undefined constraint '{cname}'")
            elif spec.terms[cname].role != "constraint":
                errors.append(
                    f"Setting '{sname}': constraint '{cname}' is not a constraint term"
                )

    if spec.task_agg != "gmean":
        errors.append(f"Score spec has unsupported task aggregation '{spec.task_agg}'")

    return errors


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
