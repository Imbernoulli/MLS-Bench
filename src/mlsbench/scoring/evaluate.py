"""Core evaluation engine: score tasks using score_spec.py and leaderboard data."""

from __future__ import annotations

import csv
import fnmatch
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mlsbench import PROJECT_ROOT
from mlsbench.scoring._numeric import is_finite_real
from mlsbench.scoring.anchors import BaselineAnchors
from mlsbench.scoring.primitives import (
    apply_direction_and_transform,
    bounded_power,
    logistic_score,
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
    _is_valid_anchor,
    load_score_spec,
    leaderboard_declared_metrics,
    validate_score_spec,
)

SHORT_ELAPSED_MEDIAN_RATIO = 0.5
HIGH_NEAR_WORST_FRAC = 0.05
LOW_BOUND_ATOL = 1e-9


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
    valid: bool = True
    invalid_reason: str | None = None


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
    if not _is_valid_anchor(ref):
        return None
    if is_finite_real(ref):
        return float(ref)
    if isinstance(ref, AnchorRef):
        if ref.kind == "const":
            return float(ref.value)
        if ref.kind == "bl_worst":
            return anchors.worst_for(ref.metric, direction)
        if ref.kind == "bl_best":
            return anchors.best_for(ref.metric, direction)
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
            if lbl:
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

def _score_term(
    tspec: TermSpec,
    raw_value: float | None,
    floor_raw: float | None,
    anchors: BaselineAnchors,
) -> TermResult:
    """Score a single term given its raw metric value."""
    def invalid(raw: float, transformed: float, reason: str) -> TermResult:
        return TermResult(
            name=tspec.name,
            metric=tspec.metric,
            raw=raw,
            transformed=transformed,
            score=0.0,
            params={"reason": reason},
            valid=False,
            invalid_reason=reason,
        )

    def transform_value(value: float, field: str) -> tuple[float | None, str | None]:
        if not is_finite_real(value):
            return None, f"invalid_{field}_type"
        try:
            transformed = apply_direction_and_transform(
                float(value), tspec.direction, tspec.transform
            )
        except (TypeError, ValueError, OverflowError) as exc:
            return None, f"invalid_{field}_transform:{exc}"
        if not math.isfinite(transformed):
            return None, f"nonfinite_{field}_transform"
        return transformed, None

    if _is_missing_value(raw_value):
        return invalid(float("nan"), float("nan"), "missing_value")

    raw_f = float(raw_value)
    y, transform_error = transform_value(raw_f, "value")
    if transform_error is not None or y is None:
        return invalid(raw_f, float("nan"), transform_error or "invalid_value_transform")

    if tspec.role == "constraint":
        target = tspec.constraint_target
        if not is_finite_real(target):
            return invalid(raw_f, y, "invalid_constraint_target")
        if (
            not is_finite_real(tspec.constraint_sharpness)
            or float(tspec.constraint_sharpness) <= 0.0
        ):
            return invalid(raw_f, y, "invalid_constraint_sharpness")
        if tspec.norm_type == "penalty_upper":
            p = penalty_upper(raw_f, target, tspec.constraint_sharpness)
        elif tspec.norm_type == "penalty_lower":
            p = penalty_lower(raw_f, target, tspec.constraint_sharpness)
        else:
            return invalid(
                raw_f,
                y,
                f"invalid_constraint_norm:{tspec.norm_type}",
            )
        return TermResult(
            name=tspec.name, metric=tspec.metric,
            raw=raw_f, transformed=y, score=p,
            params={"target": target, "sharpness": tspec.constraint_sharpness},
        )

    explicit_midpoint = (
        tspec.norm_type == "sigmoid"
        and tspec.floor is None
        and tspec.ref is not None
        and tspec.scale is not None
    )
    if explicit_midpoint:
        ref_resolved = _resolve_anchor(
            tspec.ref, anchors, tspec.metric, tspec.direction
        )
        if ref_resolved is None or _is_missing_value(ref_resolved):
            return invalid(raw_f, y, "unresolved_logistic_midpoint")
        if not is_finite_real(tspec.scale) or float(tspec.scale) <= 0.0:
            return invalid(raw_f, y, "invalid_logistic_scale")
        scale = float(tspec.scale)
        y_mid, transform_error = transform_value(float(ref_resolved), "reference")
        if transform_error is not None or y_mid is None:
            return invalid(raw_f, y, transform_error or "invalid_reference_transform")
        score = logistic_score(y, y_mid, scale)
        return TermResult(
            name=tspec.name,
            metric=tspec.metric,
            raw=raw_f,
            transformed=y,
            score=score,
            params={"midpoint": y_mid, "scale": scale},
        )

    if floor_raw is None or _is_missing_value(floor_raw):
        return invalid(raw_f, y, "missing_floor")
    y_floor, transform_error = transform_value(float(floor_raw), "floor")
    if transform_error is not None or y_floor is None:
        return invalid(raw_f, y, transform_error or "invalid_floor_transform")

    if tspec.norm_type == "bounded_power":
        bound_raw = tspec.bound
        if bound_raw is None or _is_missing_value(bound_raw):
            return invalid(raw_f, y, "no_bound")
        y_bound, transform_error = transform_value(float(bound_raw), "bound")
        if transform_error is not None or y_bound is None:
            return invalid(raw_f, y, transform_error or "invalid_bound_transform")
        if y_bound == y_floor:
            return invalid(raw_f, y, "degenerate_bound")

        ref_resolved: float | None = None
        r_ref: float | None = None
        if tspec.floor is not None and tspec.ref is None:
            # An explicit floor and bound fully define the baseline-free linear
            # curve. Legacy terms without an explicit floor still calibrate from
            # the best baseline below.
            gamma = 1.0
        else:
            ref_resolved = _resolve_anchor(
                tspec.ref, anchors, tspec.metric, tspec.direction
            )
            if tspec.ref is not None and ref_resolved is None:
                return invalid(raw_f, y, "unresolved_bounded_power_ref")
            if ref_resolved is None:
                ref_resolved = _default_ref(
                    anchors, tspec.metric, tspec.direction
                )
            if ref_resolved is None or _is_missing_value(ref_resolved):
                return invalid(raw_f, y, "missing_bounded_power_ref")
            y_ref, transform_error = transform_value(
                ref_resolved, "reference"
            )
            if transform_error is not None or y_ref is None:
                return invalid(
                    raw_f,
                    y,
                    transform_error or "invalid_reference_transform",
                )
            r_ref = _bounded_power_ref_ratio(y_floor, y_bound, y_ref)
            try:
                gamma = solve_gamma(
                    y_floor, y_bound, y_ref, tspec.ref_score
                )
            except ValueError as exc:
                return invalid(
                    raw_f,
                    y,
                    f"invalid_bounded_power_calibration:{exc}",
                )

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

    if tspec.norm_type == "sigmoid":
        if tspec.scale is not None:
            if not is_finite_real(tspec.scale) or float(tspec.scale) <= 0.0:
                return invalid(raw_f, y, "invalid_sigmoid_scale")
            sc = float(tspec.scale)
        else:
            ref_resolved = _resolve_anchor(tspec.ref, anchors, tspec.metric, tspec.direction)
            if tspec.ref is not None and ref_resolved is None:
                return invalid(raw_f, y, "unresolved_sigmoid_ref")
            if ref_resolved is None:
                ref_resolved = _default_ref(anchors, tspec.metric, tspec.direction)
            if ref_resolved is None or _is_missing_value(ref_resolved):
                return invalid(raw_f, y, "missing_sigmoid_ref_or_scale")
            y_ref, transform_error = transform_value(float(ref_resolved), "reference")
            if transform_error is not None or y_ref is None:
                return invalid(raw_f, y, transform_error or "invalid_reference_transform")
            try:
                sc = solve_scale(y_floor, y_ref, tspec.ref_score)
            except ValueError as exc:
                return invalid(
                    raw_f,
                    y,
                    f"invalid_sigmoid_calibration:{exc}",
                )

        score = sigmoid_score(y, y_floor, sc)
        return TermResult(
            name=tspec.name, metric=tspec.metric,
            raw=raw_f, transformed=y, score=score,
            params={"floor": y_floor, "scale": sc},
        )

    return invalid(raw_f, y, f"unknown_norm_type:{tspec.norm_type}")


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
    return not is_finite_real(value)


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
    # Every score input is required and must be finite. In particular, do not
    # let objective or constraint keys from different seeds combine into an
    # apparently complete mean record after a partial verifier failure.
    higher_default_hits: list[bool] = []
    lower_bound_hits: list[bool] = []

    for term_name, _weight in sspec.terms:
        tspec = all_terms.get(term_name)
        if tspec is None:
            return False, f"undefined_objective:{term_name}"
        raw = record.get(tspec.metric)
        if _is_missing_value(raw):
            return False, f"missing_objective:{tspec.metric}"

        if tspec.norm_type != "bounded_power" or tspec.bound is None:
            continue
        if tspec.direction == "higher":
            higher_default_hits.append(_near_high_worst_default(float(raw), float(tspec.bound)))
        elif tspec.direction == "lower":
            lower_bound_hits.append(_near_lower_bound(float(raw), float(tspec.bound)))

    for term_name in sspec.constraints:
        tspec = all_terms.get(term_name)
        if tspec is None:
            return False, f"undefined_constraint:{term_name}"
        if _is_missing_value(record.get(tspec.metric)):
            return False, f"missing_constraint:{tspec.metric}"

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
        # Generic / baseline-free anchor: an explicit term.floor overrides the
        # baseline-derived worst anchor so a task can be scored with no baseline
        # rows in leaderboard.csv. Falls back to worst baseline when unset.
        if tspec.floor is not None:
            floor_raw = _resolve_anchor(tspec.floor, anchors, tspec.metric, tspec.direction)
        else:
            floor_raw = anchors.worst_for(tspec.metric, tspec.direction)
        tr = _score_term(tspec, raw_val, floor_raw, anchors)
        if not is_finite_real(weight) or float(weight) <= 0.0:
            tr.score = 0.0
            tr.valid = False
            tr.invalid_reason = "invalid_objective_weight"
            tr.params = {"reason": tr.invalid_reason}
            weight_f = 0.0
        else:
            weight_f = float(weight)
        term_results.append(tr)
        obj_scores.append(tr.score)
        obj_weights.append(weight_f)

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
    invalid_term = next((term for term in term_results if not term.valid), None)
    if invalid_term is not None:
        valid = False
        invalid_reason = (
            f"invalid_term:{invalid_term.name}:"
            f"{invalid_term.invalid_reason or 'unknown'}"
        )
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


def _gmean(values: list[float]) -> float:
    """Return the true geometric mean, failing closed on invalid inputs."""
    if not values:
        return 0.0
    numeric: list[float] = []
    for value in values:
        if not is_finite_real(value):
            return 0.0
        score = float(value)
        if not math.isfinite(score) or score <= 0.0:
            return 0.0
        numeric.append(score)
    return math.exp(math.fsum(math.log(score) for score in numeric) / len(numeric))


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

    A record is valid if all its settings are valid (no missing/non-finite
    objective or constraint, no partial seed=mean, no crash-default pattern). When invalid, the
    aggregated score is forced to 0 so that crash-defaulted vanilla rows
    don't outscore real agent runs."""
    spec_errors = validate_score_spec(spec, [str(key) for key in record])
    if spec_errors:
        reason = f"invalid_score_spec:{spec_errors[0]}"
        return 0.0, [
            SettingResult(
                name=sspec.name,
                objective_score=0.0,
                penalty=0.0,
                score=0.0,
                valid=False,
                invalid_reason=reason,
            )
            for sspec in spec.settings.values()
        ], False

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
        # A final row is authoritative even when it is incomplete. Never skip a
        # failed final verifier row and resurrect an older non-final score.
        final_any = [r for r in recs if str(r.get("is_final", "")).lower() == "true"]

        if warns:
            results.append(TaskResult(
                task=task_name,
                model=model_name,
                score=0.0,
                warnings=[
                    "Invalid score specification; score forced to 0: "
                    + "; ".join(warns)
                ],
            ))
            continue

        config_path = task_dir / "config.json"
        config_error: str | None = None
        try:
            task_config = json.loads(config_path.read_text()) if config_path.exists() else {}
        except (OSError, json.JSONDecodeError) as exc:
            task_config = {}
            config_error = f"invalid task config: {exc}"
        configured_seeds = task_config.get("seeds", [42])
        if isinstance(configured_seeds, int):
            configured_seeds = [configured_seeds]
        if (
            not isinstance(configured_seeds, list)
            or not configured_seeds
            or any(not isinstance(seed, int) or isinstance(seed, bool) for seed in configured_seeds)
            or len(set(configured_seeds)) != len(configured_seeds)
        ):
            config_error = "config declares an invalid seed matrix"
        if config_error is not None:
            results.append(TaskResult(
                task=task_name,
                model=model_name,
                score=0.0,
                warnings=[f"{config_error}; score forced to 0."],
            ))
            continue

        if not final_any:
            results.append(TaskResult(
                task=task_name,
                model=model_name,
                score=0.0,
                warnings=["No authoritative final submission row; score forced to 0."],
            ))
            continue

        # A successful multi-seed submit writes the mean row last. If a newer
        # retry leaves only empty per-seed finals, that latest failure must win
        # over any older positive mean row.
        record = max(final_any, key=lambda r: str(r.get("timestamp", "")))
        if len(configured_seeds) > 1 and record.get("seed") != "mean":
            results.append(TaskResult(
                task=task_name,
                model=model_name,
                score=0.0,
                warnings=[
                    "Latest final multi-seed submission is missing its authoritative "
                    "mean row; score forced to 0."
                ],
            ))
            continue

        from mlsbench.agent.leaderboard import Leaderboard

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
