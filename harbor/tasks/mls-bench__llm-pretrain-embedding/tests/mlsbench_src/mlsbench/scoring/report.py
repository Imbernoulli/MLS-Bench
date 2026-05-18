"""Output formatting for score results (terminal tables, JSON, CSV)."""

from __future__ import annotations

import csv
import io
import json
from typing import Any

from mlsbench.scoring.evaluate import SettingResult, TaskResult


# ---------------------------------------------------------------------------
# Terminal table formatting
# ---------------------------------------------------------------------------

def format_task_detail(results: list[TaskResult], verbose: bool = False) -> str:
    """Render detailed scoring results for a single task."""
    if not results:
        return "No results to display."

    parts: list[str] = []
    task_name = results[0].task

    for tr in results:
        lines: list[str] = []
        lines.append(f"Task: {task_name}")
        lines.append(f"Model: {tr.model}")
        lines.append("")

        if tr.settings:
            # Setting table
            max_sname = max(len(sr.name) for sr in tr.settings)
            max_sname = max(max_sname, len("Setting"))
            hdr = f"  {'Setting':<{max_sname}} | Score"
            sep = f"  {'-' * max_sname}-+------"
            lines.append(hdr)
            lines.append(sep)
            for sr in tr.settings:
                lines.append(f"  {sr.name:<{max_sname}} | {sr.score:.3f}")
            lines.append(sep)
            lines.append(f"  {'Task Score (gmean)':<{max_sname}} | {tr.score:.3f}")
        else:
            lines.append(f"  Task Score: {tr.score:.3f}")

        if verbose and tr.settings:
            lines.append("")
            lines.append("  Term Details:")
            for sr in tr.settings:
                for t in sr.terms:
                    p = t.params
                    if t.score == 0.0 and p.get("reason") == "missing_value":
                        detail = "(missing)"
                    elif "gamma" in p:
                        detail = f"(bounded_power, gamma={p['gamma']:.2f})"
                    elif "scale" in p:
                        detail = f"(sigmoid, scale={p['scale']:.1f})"
                    elif "target" in p:
                        detail = f"(penalty, target={p['target']})"
                    else:
                        detail = ""
                    raw_str = f"{t.raw:.4f}" if not (t.raw != t.raw) else "NaN"
                    lines.append(
                        f"    [{sr.name}] {t.metric}: raw={raw_str} -> score={t.score:.3f} {detail}"
                    )

        if tr.warnings:
            lines.append("")
            for w in tr.warnings:
                lines.append(f"  WARNING: {w}")

        parts.append("\n".join(lines))

    return "\n\n".join(parts)


def format_summary_table(
    all_results: dict[str, list[TaskResult]],
    summary_only: bool = False,
) -> str:
    """Render a cross-task summary table.

    Columns: Task | model1 | model2 | ...
    """
    if not all_results:
        return "No scored tasks found."

    # Collect all models
    model_set: set[str] = set()
    for task_results in all_results.values():
        for tr in task_results:
            model_set.add(tr.model)
    models = sorted(model_set)

    if not models:
        return "No agent results found."

    # Build score lookup: {(task, model): score}
    scores: dict[tuple[str, str], float | None] = {}
    for task_name, task_results in all_results.items():
        for tr in task_results:
            scores[(task_name, tr.model)] = tr.score

    tasks = sorted(all_results.keys())

    # Format model names (truncate for display)
    def _short_model(m: str) -> str:
        # Remove common prefixes
        for prefix in ("claude-", "deepseek-", "gpt-", "qwen-"):
            if m.startswith(prefix):
                return m
        return m[:20]

    model_headers = [_short_model(m) for m in models]
    col_width = max(13, max(len(h) for h in model_headers) + 2)
    task_width = max(30, max(len(t) for t in tasks) + 2)

    lines: list[str] = []

    # Header
    hdr = f"{'Task':<{task_width}}"
    for mh in model_headers:
        hdr += f" | {mh:>{col_width}}"
    lines.append(hdr)

    # Separator
    sep = "-" * task_width
    for _ in models:
        sep += "-+-" + "-" * col_width
    lines.append(sep)

    # Rows
    model_task_scores: dict[str, list[float]] = {m: [] for m in models}
    for task_name in tasks:
        row = f"{task_name:<{task_width}}"
        for m in models:
            s = scores.get((task_name, m))
            if s is not None:
                row += f" | {s:>{col_width}.3f}"
                model_task_scores[m].append(s)
            else:
                row += f" | {'—':>{col_width}}"
        lines.append(row)

    # Overall row (gmean across tasks)
    lines.append(sep)
    from mlsbench.scoring.evaluate import _gmean
    overall_row = f"{'Overall (gmean)':<{task_width}}"
    for m in models:
        ts = model_task_scores[m]
        if ts:
            overall = _gmean(ts)
            overall_row += f" | {overall:>{col_width}.3f}"
        else:
            overall_row += f" | {'—':>{col_width}}"
    lines.append(overall_row)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# JSON / CSV export
# ---------------------------------------------------------------------------

def results_to_json(all_results: dict[str, list[TaskResult]], indent: int = 2) -> str:
    """Serialize all results to JSON."""
    data: dict[str, Any] = {}
    for task_name, task_results in all_results.items():
        data[task_name] = []
        for tr in task_results:
            entry: dict[str, Any] = {
                "model": tr.model,
                "task_score": tr.score,
                "settings": [],
            }
            for sr in tr.settings:
                s_entry: dict[str, Any] = {
                    "name": sr.name,
                    "score": sr.score,
                    "objective_score": sr.objective_score,
                    "penalty": sr.penalty,
                    "terms": [],
                }
                for t in sr.terms:
                    s_entry["terms"].append({
                        "name": t.name,
                        "metric": t.metric,
                        "raw": None if (isinstance(t.raw, float) and t.raw != t.raw) else t.raw,
                        "score": t.score,
                    })
                entry["settings"].append(s_entry)
            if tr.warnings:
                entry["warnings"] = tr.warnings
            data[task_name].append(entry)
    return json.dumps(data, indent=indent, ensure_ascii=False)


def results_to_csv(all_results: dict[str, list[TaskResult]]) -> str:
    """Serialize all results to CSV (one row per task-model pair)."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["task", "model", "task_score"])
    for task_name in sorted(all_results.keys()):
        for tr in all_results[task_name]:
            writer.writerow([task_name, tr.model, f"{tr.score:.4f}"])
    return buf.getvalue()
