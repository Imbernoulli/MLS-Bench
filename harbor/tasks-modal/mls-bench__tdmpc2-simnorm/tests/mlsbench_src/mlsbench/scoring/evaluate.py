"""Core evaluation engine: score tasks using score_spec.py and leaderboard data."""

from __future__ import annotations

import csv
import fnmatch
import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mlsbench import PROJECT_ROOT
from mlsbench.scoring.anchors import BaselineAnchors
from mlsbench.scoring.primitives import (
    apply_direction_and_transform,
    bounded_power,
    penalty_lower,
    penalty_upper,
    sigmoid_score,
    solve_gamma,
    solve_scale,
)
from mlsbench.scoring.spec import (
    DEFAULT_REF_SCORE,
    AnchorRef,
    SettingSpec,
    TaskScoreSpec,
    TermSpec,
    load_score_spec,
    leaderboard_declared_metrics,
    validate_score_spec,
)

GMEAN_EPS = 0.01
SHORT_ELAPSED_MEDIAN_RATIO = 0.5
HIGH_NEAR_WORST_FRAC = 0.05
LOW_BOUND_ATOL = 1e-9
PATHOLOGICAL_BOUNDED_POWER_EDGE = 0.05
_PATHOLOGICAL_BOUNDED_POWER_WARNED: set[tuple[str, str]] = set()


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class TermResult:
    name: str
    metric: str
    raw: float
    transformed: float
    score: float
    params: dict = field(default_factory=dict)


@dataclass
class SettingResult:
    name: str
    objective_score: float
    penalty: float
    score: float
    valid: bool = True
    invalid_reason: str | None = None
    terms: list[TermResult] = field(default_factory=list)


@dataclass
class TaskResult:
    task: str
    model: str
    score: float
    settings: list[SettingResult] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Anchor resolution
# ---------------------------------------------------------------------------

def _resolve_anchor(
    ref: float | AnchorRef | None,
    anchors: BaselineAnchors,
    metric: str,
    direction: str,
) -> float | None:
    if ref is None:
        return None
    if isinstance(ref, (int, float)):
        return float(ref)
    if isinstance(ref, AnchorRef):
        if ref.kind == "const":
            return ref.value
        if ref.kind == "bl_worst":
            return anchors.worst_for(ref.metric or metric, direction)
        if ref.kind == "bl_best":
            return anchors.best_for(ref.metric or metric, direction)
    return None


def _default_ref(anchors: BaselineAnchors, metric: str, direction: str) -> float | None:
    """Default calibration anchor: best baseline for the metric direction."""
    return anchors.best_for(metric, direction)


# ---------------------------------------------------------------------------
# quick_spec expansion
# ---------------------------------------------------------------------------

def _expand_quick_spec(
    raw: dict[str, Any],
    anchors: BaselineAnchors,
    task_dir: Path,
) -> TaskScoreSpec:
    """Expand a quick_spec declaration into a full TaskScoreSpec."""
    import json

    metric_patterns: dict[str, dict] = raw["metrics"]
    settings_from: str = raw.get("settings_from", "labels")
    task_agg: str = raw.get("task_agg", "gmean")

    avail_cols = anchors.metric_columns()

    # Load test_cmd labels for setting names
    labels: list[str] = []
    cfg_path = task_dir / "config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
        for tc in cfg.get("test_cmds", []):
            lbl = tc.get("label", "")
            if lbl and not tc.get("hidden", False):
                labels.append(lbl)

    spec = TaskScoreSpec(task_agg=task_agg)
    settings_terms: dict[str, list[tuple[str, float]]] = {}

    for pattern, mdesc in metric_patterns.items():
        direction = mdesc.get("direction", "higher")
        norm = mdesc.get("norm", "sigmoid")
        bound = mdesc.get("bound")
        ref = mdesc.get("ref")
        ref_score = mdesc.get("ref_score", DEFAULT_REF_SCORE)
        weight = mdesc.get("weight", 1.0)
        role = mdesc.get("role", "objective")
        transform = mdesc.get("transform", "id")

        if "*" in pattern:
            # Expand wildcard against available columns
            matched = fnmatch.filter(avail_cols, pattern)
        else:
            matched = [pattern] if pattern in avail_cols else []

        for col_name in matched:
            # Determine setting name from suffix
            setting_name = "default"
            if settings_from == "labels" and "*" in pattern:
                prefix = pattern.replace("*", "")
                suffix = col_name
                if prefix and col_name.startswith(prefix):
                    suffix = col_name[len(prefix):]
                elif prefix and col_name.endswith(prefix.rstrip("_")):
                    suffix = col_name
                # Match suffix against labels
                for lbl in labels:
                    if lbl in suffix or suffix in lbl:
                        setting_name = lbl
                        break
                else:
                    setting_name = suffix.strip("_") or "default"

            term_name = col_name.replace("-", "_").replace(".", "_")

            tspec = TermSpec(
                name=term_name,
                metric=col_name,
                role=role,
                direction=direction,
                transform=transform,
                norm_type="bounded_power" if norm == "bounded" else "sigmoid",
                bound=bound,
                ref=ref,
                ref_score=ref_score,
            )
            spec.terms[term_name] = tspec

            if role == "objective":
                settings_terms.setdefault(setting_name, []).append((term_name, weight))

    for sname, items in settings_terms.items():
        spec.settings[sname] = SettingSpec(name=sname, terms=items)

    return spec


# ---------------------------------------------------------------------------
# Single-term scoring
# ---------------------------------------------------------------------------

def _task_name_from_anchors(anchors: BaselineAnchors) -> str:
    task_dir = getattr(anchors, "_task_dir", None)
    if task_dir is None:
        return "<unknown>"
    return Path(task_dir).name


def _bounded_power_ref_ratio(floor: float, bound: float, ref: float) -> float | None:
    """Return the standard bounded_power reference ratio, if applicable.

    This is intentionally limited to the standard orientation where ``bound``
    is the better-side theoretical limit. Existing inverted specs use
    ``bound < floor`` as a hard sanity floor and must keep the legacy
    ``bounded_power`` behavior.
    """
    if bound <= floor:
        return None
    denom = bound - floor
    if denom == 0.0:
        return None
    return (ref - floor) / denom


def _is_pathological_bounded_power_ref(r_ref: float | None) -> bool:
    if r_ref is None:
        return False
    if r_ref <= 0.0 or r_ref >= 1.0:
        return False
    return (
        r_ref < PATHOLOGICAL_BOUNDED_POWER_EDGE
        or r_ref > 1.0 - PATHOLOGICAL_BOUNDED_POWER_EDGE
    )


def _warn_pathological_bounded_power(
    anchors: BaselineAnchors,
    term_name: str,
    r_ref: float,
) -> None:
    task_name = _task_name_from_anchors(anchors)
    key = (task_name, term_name)
    if key in _PATHOLOGICAL_BOUNDED_POWER_WARNED:
        return
    _PATHOLOGICAL_BOUNDED_POWER_WARNED.add(key)
    warnings.warn(
        f"Task {task_name}, term {term_name}: r_ref={r_ref:.4f} is pathological "
        "for bounded_power; falling back to sigmoid_score calibration so the "
        "reference baseline maps to ref_score.",
        stacklevel=2,
    )

def _score_term(
    tspec: TermSpec,
    raw_value: float | None,
    floor_raw: float | None,
    anchors: BaselineAnchors,
) -> TermResult:
    """Score a single term given its raw metric value."""
    if raw_value is None or (isinstance(raw_value, float) and math.isnan(raw_value)):
        return TermResult(
            name=tspec.name, metric=tspec.metric,
            raw=float("nan"), transformed=float("nan"), score=0.0,
            params={"reason": "missing_value"},
        )

    raw_f = float(raw_value)

    # Apply direction + transform to raw value
    y = apply_direction_and_transform(raw_f, tspec.direction, tspec.transform)

    # Compute floor in transformed space
    if floor_raw is not None:
        y_floor = apply_direction_and_transform(float(floor_raw), tspec.direction, tspec.transform)
    else:
        y_floor = y  # no floor info; score will be 0

    if tspec.role == "constraint":
        # Constraint terms produce penalty, not score
        target = tspec.constraint_target
        if target is None:
            return TermResult(
                name=tspec.name, metric=tspec.metric,
                raw=raw_f, transformed=y, score=1.0,
                params={"reason": "no_constraint_target"},
            )
        if tspec.norm_type == "penalty_upper":
            p = penalty_upper(raw_f, target, tspec.constraint_sharpness)
        else:
            p = penalty_lower(raw_f, target, tspec.constraint_sharpness)
        return TermResult(
            name=tspec.name, metric=tspec.metric,
            raw=raw_f, transformed=y, score=p,
            params={"target": target, "sharpness": tspec.constraint_sharpness},
        )

    # Objective terms
    if tspec.norm_type == "bounded_power":
        # Transform bound to internal space
        bound_raw = tspec.bound
        if bound_raw is None:
            return TermResult(
                name=tspec.name, metric=tspec.metric,
                raw=raw_f, transformed=y, score=0.0,
                params={"reason": "no_bound"},
            )
        y_bound = apply_direction_and_transform(float(bound_raw), tspec.direction, tspec.transform)

        # Resolve ref
        ref_resolved = _resolve_anchor(tspec.ref, anchors, tspec.metric, tspec.direction)
        if ref_resolved is None:
            ref_resolved = _default_ref(anchors, tspec.metric, tspec.direction)
        r_ref = None
        if ref_resolved is not None:
            y_ref = apply_direction_and_transform(float(ref_resolved), tspec.direction, tspec.transform)
            r_ref = _bounded_power_ref_ratio(y_floor, y_bound, y_ref)
            if _is_pathological_bounded_power_ref(r_ref):
                # When the best baseline is nearly at a bounded theoretical
                # limit, the gamma needed to keep score(ref)=0.5 can be far
                # outside the clipped [0.1, 10] range. A sigmoid calibration
                # preserves the ref_score anchor instead of inflating Human SOTA.
                _warn_pathological_bounded_power(anchors, tspec.name, r_ref)
                sc = solve_scale(y_floor, y_ref, tspec.ref_score)
                score = sigmoid_score(y, y_floor, sc)
                return TermResult(
                    name=tspec.name, metric=tspec.metric,
                    raw=raw_f, transformed=y, score=score,
                    params={
                        "floor": y_floor,
                        "bound": y_bound,
                        "scale": sc,
                        "ref": ref_resolved,
                        "r_ref": r_ref,
                        "fallback": "sigmoid_pathological_bounded_power",
                    },
                )
            gamma = solve_gamma(y_floor, y_bound, y_ref, tspec.ref_score)
        else:
            gamma = 1.0  # linear fallback

        score = bounded_power(y, y_floor, y_bound, gamma)
        return TermResult(
            name=tspec.name, metric=tspec.metric,
            raw=raw_f, transformed=y, score=score,
            params={
                "floor": y_floor,
                "bound": y_bound,
                "gamma": gamma,
                "ref": ref_resolved,
                "r_ref": r_ref,
            },
        )

    elif tspec.norm_type == "sigmoid":
        if tspec.scale is not None:
            sc = tspec.scale
        else:
            ref_resolved = _resolve_anchor(tspec.ref, anchors, tspec.metric, tspec.direction)
            if ref_resolved is None:
                ref_resolved = _default_ref(anchors, tspec.metric, tspec.direction)
            if ref_resolved is not None:
                y_ref = apply_direction_and_transform(float(ref_resolved), tspec.direction, tspec.transform)
                sc = solve_scale(y_floor, y_ref, tspec.ref_score)
            else:
                sc = 1.0
                warnings.warn(
                    f"Term '{tspec.name}': no ref or scale for sigmoid; using scale=1.0",
                    stacklevel=2,
                )

        score = sigmoid_score(y, y_floor, sc)
        return TermResult(
            name=tspec.name, metric=tspec.metric,
            raw=raw_f, transformed=y, score=score,
            params={"floor": y_floor, "scale": sc},
        )

    return TermResult(
        name=tspec.name, metric=tspec.metric,
        raw=raw_f, transformed=y, score=0.0,
        params={"reason": f"unknown_norm_type:{tspec.norm_type}"},
    )


def _parse_csv_value(col: str, val: Any) -> Any:
    if val in ("", None):
        return None
    if col in {"timestamp", "model", "seed"}:
        return val
    if col == "is_final":
        return str(val).lower() == "true"
    try:
        f = float(val)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return val


def _load_leaderboard_records(lb_path: Path) -> list[dict[str, Any]]:
    if not lb_path.exists():
        return []
    with lb_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        return [{c: _parse_csv_value(c, row.get(c, "")) for c in cols} for row in reader]


def _is_missing_value(value: Any) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def _near_high_worst_default(raw: float, bound: float) -> bool:
    b = abs(float(bound))
    thresh = max(HIGH_NEAR_WORST_FRAC * b, HIGH_NEAR_WORST_FRAC) if b > 0 else HIGH_NEAR_WORST_FRAC
    return float(raw) <= thresh


def _near_lower_bound(raw: float, bound: float) -> bool:
    return abs(float(raw) - float(bound)) <= LOW_BOUND_ATOL


def _setting_elapsed_is_suspicious(
    setting_name: str,
    record: dict,
    anchors: BaselineAnchors,
) -> bool:
    elapsed_key = f"elapsed_{setting_name}"
    elapsed = record.get(elapsed_key)
    if _is_missing_value(elapsed):
        return False
    elapsed_anchor = anchors.get(elapsed_key)
    if not elapsed_anchor or not elapsed_anchor.values:
        return False
    return float(elapsed) < SHORT_ELAPSED_MEDIAN_RATIO * float(sorted(elapsed_anchor.values)[len(elapsed_anchor.values) // 2])


def _validate_setting(
    sspec: SettingSpec,
    all_terms: dict[str, TermSpec],
    record: dict,
    anchors: BaselineAnchors,
) -> tuple[bool, str | None]:
    # missing-objective: strictly required terms absent → invalid.
    # partial-mean rows from older CSVs: NOT invalid.
    # Real partial-coverage runs (e.g. 1/3 seeds succeeding on a given env)
    # carry true measurements; zeroing them treats them like crash defaults.
    # Crash defaults are still caught by the higher+lower bound + short-elapsed
    # heuristic below.
    higher_default_hits: list[bool] = []
    lower_bound_hits: list[bool] = []

    for term_name, _weight in sspec.terms:
        tspec = all_terms.get(term_name)
        if tspec is None:
            continue
        raw = record.get(tspec.metric)
        if _is_missing_value(raw):
            return False, f"missing_objective:{tspec.metric}"

        if tspec.norm_type != "bounded_power" or tspec.bound is None:
            continue
        if tspec.direction == "higher":
            higher_default_hits.append(_near_high_worst_default(float(raw), float(tspec.bound)))
        elif tspec.direction == "lower":
            lower_bound_hits.append(_near_lower_bound(float(raw), float(tspec.bound)))

    if (
        higher_default_hits
        and lower_bound_hits
        and all(higher_default_hits)
        and all(lower_bound_hits)
        and _setting_elapsed_is_suspicious(sspec.name, record, anchors)
    ):
        return False, "crash_default_pattern"

    return True, None


# ---------------------------------------------------------------------------
# Setting and task scoring
# ---------------------------------------------------------------------------

def _score_setting(
    sspec: SettingSpec,
    all_terms: dict[str, TermSpec],
    record: dict,
    anchors: BaselineAnchors,
) -> SettingResult:
    """Score a single setting from one model record."""
    term_results: list[TermResult] = []

    # Score objective terms
    obj_scores: list[float] = []
    obj_weights: list[float] = []
    for term_name, weight in sspec.terms:
        tspec = all_terms.get(term_name)
        if tspec is None:
            continue
        raw_val = record.get(tspec.metric)
        floor_raw = anchors.worst_for(tspec.metric, tspec.direction)
        tr = _score_term(tspec, raw_val, floor_raw, anchors)
        term_results.append(tr)
        obj_scores.append(tr.score)
        obj_weights.append(weight)

    # Weighted mean of objectives
    if obj_weights:
        total_w = sum(obj_weights)
        obj_score = sum(s * w for s, w in zip(obj_scores, obj_weights)) / total_w if total_w > 0 else 0.0
    else:
        obj_score = 0.0

    # Score constraint terms
    penalty = 1.0
    for cname in sspec.constraints:
        tspec = all_terms.get(cname)
        if tspec is None:
            continue
        raw_val = record.get(tspec.metric)
        tr = _score_term(tspec, raw_val, None, anchors)
        term_results.append(tr)
        penalty *= tr.score

    valid, invalid_reason = _validate_setting(sspec, all_terms, record, anchors)
    final = obj_score * penalty if valid else 0.0
    return SettingResult(
        name=sspec.name,
        objective_score=obj_score,
        penalty=penalty,
        score=final,
        valid=valid,
        invalid_reason=invalid_reason,
        terms=term_results,
    )


def _gmean(values: list[float], eps: float = GMEAN_EPS) -> float:
    """Geometric mean with epsilon floor to avoid zero-collapse."""
    if not values:
        return 0.0
    if all(v <= 0.0 for v in values):
        return 0.0
    log_sum = sum(math.log(max(v, eps)) for v in values)
    return math.exp(log_sum / len(values))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_expanded_spec(
    task_dir: Path,
    anchors: BaselineAnchors,
) -> TaskScoreSpec | None:
    """Load a task's score_spec.py and expand any quick_spec into a full spec.

    Returns None if no score_spec.py exists or the spec is empty/invalid.
    Shared between `evaluate_task` and downstream tools that need to score
    arbitrary records (e.g. scripts/build_maintab.py).
    """
    raw_spec = load_score_spec(task_dir)
    if raw_spec is None:
        return None

    from mlsbench.scoring import dsl as dsl_mod
    spec = raw_spec

    spec_path = task_dir / "score_spec.py"
    if spec_path.exists():
        registry = dsl_mod._new_registry()
        prev = dsl_mod._REGISTRY
        dsl_mod._REGISTRY = registry
        try:
            import importlib.util
            mod_spec = importlib.util.spec_from_file_location(
                f"score_spec_{task_dir.name}", spec_path,
            )
            module = importlib.util.module_from_spec(mod_spec)
            mod_spec.loader.exec_module(module)
        finally:
            dsl_mod._REGISTRY = prev

        if hasattr(registry, '_quick_spec'):
            spec = _expand_quick_spec(registry._quick_spec, anchors, task_dir)
        else:
            spec = registry.to_task_spec()

    if not spec.settings:
        return None
    return spec


def score_record(
    spec: TaskScoreSpec,
    record: dict,
    anchors: BaselineAnchors,
) -> float:
    """Score a single leaderboard record against a spec. Returns gmean across settings."""
    return score_record_details(spec, record, anchors)[0]


def score_record_details(
    spec: TaskScoreSpec,
    record: dict,
    anchors: BaselineAnchors,
) -> tuple[float, list[SettingResult], bool]:
    """Score a record AND return per-setting results + record-level validity.

    A record is valid if all its settings are valid (no missing objective,
    no partial seed=mean, no crash-default pattern). When invalid, the
    aggregated score is forced to 0 so that crash-defaulted vanilla rows
    don't outscore real agent runs."""
    setting_scores: list[float] = []
    setting_results: list[SettingResult] = []
    for _sname, sspec in spec.settings.items():
        sr = _score_setting(sspec, spec.terms, record, anchors)
        setting_results.append(sr)
        setting_scores.append(sr.score)
    record_valid = all(sr.valid for sr in setting_results)
    return (_gmean(setting_scores) if record_valid else 0.0), setting_results, record_valid


def evaluate_task(
    task_name: str,
    model: str | None = None,
    tasks_dir: Path | None = None,
) -> list[TaskResult]:
    """Score one task. Returns one TaskResult per agent model in leaderboard."""
    if tasks_dir is None:
        tasks_dir = PROJECT_ROOT / "tasks"
    task_dir = tasks_dir / task_name

    # Load anchors
    anchors = BaselineAnchors(task_dir)

    # Load + expand spec (handles quick_spec)
    spec = load_expanded_spec(task_dir, anchors)
    if spec is None:
        return []

    # Validate
    available_metrics = sorted(set(anchors.metric_columns()) | set(leaderboard_declared_metrics(task_dir)))
    warns = validate_score_spec(spec, available_metrics)

    # Load leaderboard records (direct CSV read so we see seed/is_final raw)
    records = _load_leaderboard_records(task_dir / "leaderboard.csv")

    # Find agent model rows (non-baseline, seed=mean preferred, is_final=true)
    baseline_prefixes = {"baseline:"}
    bl_names = set(anchors.baseline_names())

    def _is_baseline(r: dict) -> bool:
        m = str(r.get("model", ""))
        if any(m.startswith(p) for p in baseline_prefixes):
            return True
        if m in bl_names:
            return True
        return False

    # Group records by model
    model_records: dict[str, list[dict]] = {}
    for r in records:
        m = str(r.get("model", ""))
        if _is_baseline(r):
            continue
        if model is not None and m != model:
            continue
        model_records.setdefault(m, []).append(r)

    results: list[TaskResult] = []
    for model_name, recs in model_records.items():
        # Selection priority for an agent's "final" submission row:
        #   1. seed=mean AND is_final=true (multi-seed tasks)
        #   2. is_final=true at any seed (single-seed tasks; falls through here)
        #   3. seed=mean at any is_final state (multi-seed tasks where agent never marked final)
        #   4. latest row of any kind (fallback)
        # Within each tier prefer the most metric-complete row, then latest by time.
        # Without the completeness preference, an agent's later mid-iteration row
        # (e.g. partial env coverage from a killed test) would supersede an
        # earlier completed final.
        final_mean = [r for r in recs if r.get("seed") == "mean" and str(r.get("is_final", "")).lower() == "true"]
        final_any = [r for r in recs if str(r.get("is_final", "")).lower() == "true"]
        nonfinal_mean = [r for r in recs if r.get("seed") == "mean"]

        def _completeness(rec: dict) -> int:
            count = 0
            for k, v in rec.items():
                if (
                    k in {"timestamp", "model", "is_final", "seed"}
                    or k.startswith("elapsed_")
                    or k.endswith("_std")
                ):
                    continue
                if v in ("", None):
                    continue
                if isinstance(v, float) and math.isnan(v):
                    continue
                # Defensive: CSV cells that escaped float() conversion as
                # literal "nan" / "NaN" / "null" strings should not count as
                # populated. Leaderboard.all_records currently coerces "nan"
                # to float NaN, but harden against schema drift.
                if isinstance(v, str) and v.strip().lower() in {"nan", "null", "none"}:
                    continue
                count += 1
            return count

        def _pick(rows: list[dict]) -> dict:
            best = max(rows, key=lambda r: (_completeness(r), str(r.get("timestamp", ""))))
            return best

        from mlsbench.agent.leaderboard import Leaderboard

        def _record_valid(rec: dict) -> bool:
            if not Leaderboard.has_real_metrics(rec):
                return False
            return score_record_details(spec, rec, anchors)[2]

        # Two-pass selection: prefer valid rows in each tier; fall back to
        # picker default only if no tier has a valid candidate.
        record = None
        fallback = None
        for tier in (final_mean, final_any, nonfinal_mean, recs):
            if not tier:
                continue
            valid_rows = [r for r in tier if _record_valid(r)]
            if valid_rows:
                record = _pick(valid_rows)
                break
            if fallback is None:
                fallback = _pick(tier)
        if record is None:
            record = fallback
        if record is None:
            continue

        if not Leaderboard.has_real_metrics(record):
            # Invalid/failed agent run → score 0
            results.append(TaskResult(
                task=task_name, model=model_name, score=0.0,
                warnings=["No metric values found (agent method likely failed)"],
            ))
            continue

        task_score, setting_results, record_valid = score_record_details(spec, record, anchors)
        record_warns = list(warns)
        if not record_valid:
            record_warns.append("Selected leaderboard row is incomplete or crash-defaulted; score forced to 0.")

        results.append(TaskResult(
            task=task_name,
            model=model_name,
            score=task_score,
            settings=setting_results,
            warnings=record_warns,
        ))

    return results


def evaluate_all(
    tasks_dir: Path | None = None,
    models: list[str] | None = None,
) -> dict[str, list[TaskResult]]:
    """Score all tasks that have a score_spec.py.

    Returns {task_name: [TaskResult, ...]}.
    """
    if tasks_dir is None:
        tasks_dir = PROJECT_ROOT / "tasks"

    all_results: dict[str, list[TaskResult]] = {}
    for task_dir in sorted(tasks_dir.iterdir()):
        if not task_dir.is_dir():
            continue
        spec_path = task_dir / "score_spec.py"
        if not spec_path.exists():
            continue

        task_name = task_dir.name
        if models:
            task_results = []
            for m in models:
                task_results.extend(evaluate_task(task_name, model=m, tasks_dir=tasks_dir))
        else:
            task_results = evaluate_task(task_name, tasks_dir=tasks_dir)

        if task_results:
            all_results[task_name] = task_results

    return all_results
