"""Extract baseline anchors from leaderboard CSV.

Reads ``tasks/<task>/leaderboard.csv`` and identifies baseline rows
(``model=baseline:*``). Provides worst/best per-metric values for use
in normalization primitives.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

META_COLS = {"timestamp", "model", "is_final", "seed"}
INFORMATIONAL_PREFIXES = ("elapsed_", "n_samples", "n_prompts")


def _is_final(record: dict) -> bool:
    return str(record.get("is_final", "")).lower() == "true"


def _is_real_metric_value(value) -> bool:
    return isinstance(value, (int, float)) and not (
        isinstance(value, float) and value != value
    )


def _metric_count(record: dict) -> int:
    count = 0
    for key, value in record.items():
        if key in META_COLS:
            continue
        if str(key).startswith(INFORMATIONAL_PREFIXES) or str(key).endswith("_std"):
            continue
        if _is_real_metric_value(value):
            count += 1
    return count


def _baseline_row_priority(record: dict) -> tuple[int, bool, bool, str]:
    return (
        _metric_count(record),
        record.get("seed") == "mean",
        _is_final(record),
        str(record.get("timestamp", "")),
    )


@dataclass
class MetricAnchors:
    """Baseline anchor values for a single metric column."""
    worst: float
    best: float
    values: list[float] = field(default_factory=list)


class BaselineAnchors:
    """Read-only access to baseline anchor values from a task leaderboard."""

    def __init__(self, task_dir: Path):
        self._task_dir = Path(task_dir)
        self._anchors: dict[str, MetricAnchors] = {}
        self._baseline_names: list[str] = []
        self._metric_cols: list[str] = []
        self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def worst(self, metric: str) -> float | None:
        a = self._anchors.get(metric)
        return a.worst if a else None

    def best(self, metric: str) -> float | None:
        a = self._anchors.get(metric)
        return a.best if a else None

    def worst_for(self, metric: str, direction: str) -> float | None:
        """Return the raw baseline value that is worst for *direction*.

        ``worst()`` preserves the historical raw minimum API. Scoring code
        should use this helper so lower-is-better metrics floor at the raw
        maximum instead.
        """
        a = self._anchors.get(metric)
        if not a:
            return None
        if direction == "lower":
            return a.best
        return a.worst

    def best_for(self, metric: str, direction: str) -> float | None:
        """Return the raw baseline value that is best for *direction*."""
        a = self._anchors.get(metric)
        if not a:
            return None
        if direction == "lower":
            return a.worst
        return a.best

    def get(self, metric: str) -> MetricAnchors | None:
        return self._anchors.get(metric)

    def metric_columns(self) -> list[str]:
        """All scorable metric column names (excludes meta, elapsed_, *_std)."""
        return list(self._metric_cols)

    def baseline_names(self) -> list[str]:
        return list(self._baseline_names)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load(self) -> None:
        from mlsbench.agent.leaderboard import Leaderboard

        lb_path = self._task_dir / "leaderboard.csv"
        if not lb_path.exists():
            return

        lb = Leaderboard(lb_path)
        records = lb.all_records()
        if not records:
            return

        cfg_path = self._task_dir / "config.json"
        bl_keys: set[str] = set()
        if cfg_path.exists():
            cfg = json.loads(cfg_path.read_text())
            bl_keys = set(cfg.get("baselines", {}).keys())

        # Identify current baseline rows. When config.json declares the current
        # baseline set, ignore stale baseline rows left over from older task
        # versions. Some valid historical anchors were not marked final, so
        # ``is_final`` is a preference signal rather than a hard filter.
        baseline_groups: dict[str, list[dict]] = {}
        for r in records:
            model = r.get("model")
            if not isinstance(model, str):
                continue
            if model.startswith("baseline:"):
                name = model.removeprefix("baseline:")
            elif model in bl_keys:
                name = model
            else:
                continue
            if bl_keys and name not in bl_keys:
                continue
            baseline_groups.setdefault(name, []).append(r)

        baseline_rows: list[tuple[str, dict]] = []
        for name, rows_for_baseline in baseline_groups.items():
            rows_with_metrics = [r for r in rows_for_baseline if _metric_count(r) > 0]
            if rows_with_metrics:
                baseline_rows.append((name, max(rows_with_metrics, key=_baseline_row_priority)))

        if not baseline_rows:
            return

        self._baseline_names = [f"baseline:{name}" for name, _ in sorted(baseline_rows)]

        data_rows = [r for _, r in baseline_rows]

        # Discover metric columns. ``metric_columns()`` intentionally excludes
        # elapsed_* so autogen does not score wall time by default, but explicit
        # score_specs may still reference elapsed_*; keep anchors for those.
        anchor_cols: list[str] = []
        seen: set[str] = set()
        for r in records:
            for k in r:
                if k in seen:
                    continue
                seen.add(k)
                if k in META_COLS:
                    continue
                if k.endswith("_std"):
                    continue
                if _is_real_metric_value(r[k]):
                    anchor_cols.append(k)

        # Compute per-metric anchors from baseline data rows
        for col in anchor_cols:
            vals: list[float] = []
            for r in data_rows:
                v = r.get(col)
                if _is_real_metric_value(v):
                    vals.append(float(v))
            if vals:
                self._anchors[col] = MetricAnchors(
                    worst=min(vals),
                    best=max(vals),
                    values=vals,
                )

        self._metric_cols = [
            col for col in anchor_cols
            if col in self._anchors and not col.startswith(INFORMATIONAL_PREFIXES)
        ]
