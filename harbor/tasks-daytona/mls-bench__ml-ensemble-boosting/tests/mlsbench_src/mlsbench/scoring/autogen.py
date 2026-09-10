"""Auto-generate score_spec.py templates from task metadata and leaderboard data.

Heuristic-based: infers metric direction, norm type, bounds, and settings
from metric names and test_cmd labels. Baseline calibration anchors are left
implicit so rerunning baselines does not require regenerating score specs.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from mlsbench.scoring.anchors import BaselineAnchors

# ---------------------------------------------------------------------------
# Metric name heuristics
# ---------------------------------------------------------------------------

# Patterns for lower-is-better metrics
LOWER_PATTERNS = [
    r"loss", r"val_loss", r"fid", r"ece", r"brier", r"nll", r"mse", r"rmse",
    r"mae", r"mape", r"smape", r"nmae", r"nmse", r"igd", r"shd", r"mmd",
    r"lpips", r"latency", r"rel_err", r"relative.l2", r"cost", r"epsilon",
    r"chi2", r"perplexity", r"ppl", r"regret", r"drawdown", r"eer",
    r"mel_loss", r"mpjpe", r"degradation", r"forget_acc", r"asr",
    r"poison_fit", r"mia_auc", r"privacy_gap", r"give_up_rate",
    r"avg_queries", r"convergence_steps", r"median_steps", r"final_residual",
    r"runtime", r"num_runs", r"total_evals", r"total_grad_comps",
    r"mean_steps", r"risk_certificate", r"empirical.*risk",
    r"kl_divergence", r"ce_bound",
]

# Patterns for higher-is-better metrics
HIGHER_PATTERNS = [
    r"acc", r"accuracy", r"auc", r"auroc", r"rocauc", r"f1", r"f_score",
    r"recall", r"precision", r"return", r"reward", r"score", r"hv",
    r"hypervolume", r"clip_score", r"success", r"pass_rate", r"pass_at",
    r"psnr", r"ssim", r"stoi", r"pesq", r"si_snr", r"recovery",
    r"spearman", r"pearson", r"r2", r"ic$", r"icir", r"rank_ic",
    r"annualized_return", r"spread", r"trustworthiness", r"continuity",
    r"knn_acc", r"valid", r"unique", r"mol_stable", r"atom_stable",
    r"silhouette", r"ari", r"nmi", r"coverage", r"sopr",
    r"convergence_auc", r"best_fitness", r"downstream_score",
    r"subgroup_auroc", r"clean_acc", r"robust_acc", r"defense_score",
    r"retain_acc", r"unlearn_score", r"poison_recall", r"privacy_score",
    r"robust_score", r"test_acc", r"best_acc", r"nonzero_rate",
    r"auc_", r"balance$", r"fps", r"speedup",
    r"rp_",  # Pearson correlation coefficient
]

# Patterns for bounded metrics (metric_pattern -> bound_value)
BOUNDED_MAP = {
    r"acc": 100.0,      # some are 0-100
    r"accuracy": 100.0,
    r"test_acc": 100.0,
    r"best_acc": 100.0,
    r"clean_acc": 100.0,
    r"robust_acc": 100.0,
    r"retain_acc": 100.0,
    r"forget_acc": 100.0,
    r"auc": 1.0,
    r"auroc": 1.0,
    r"rocauc": 1.0,
    r"f1": 1.0,
    r"f_score": 1.0,
    r"recall": 1.0,
    r"precision": 1.0,
    r"ssim": 1.0,
    r"stoi": 1.0,
    r"r2": 1.0,
    r"pass_rate": 1.0,
    r"pass_at": 1.0,
    r"success_rate": 1.0,
    r"trustworthiness": 1.0,
    r"continuity": 1.0,
    r"recovery": 1.0,
    r"valid": 1.0,
    r"unique": 1.0,
    r"mol_stable": 1.0,
    r"atom_stable": 1.0,
    r"silhouette": 1.0,
    r"ari": 1.0,
    r"nmi": 1.0,
    r"loss": 0.0,
    r"val_loss": 0.0,
    r"fid": 0.0,
    r"ece": 0.0,
    r"brier": 0.0,
    r"nll": 0.0,
    r"mse": 0.0,
    r"rmse": 0.0,
    r"mae": 0.0,
    r"mape": 0.0,
    r"smape": 0.0,
    r"nmae": 0.0,
    r"nmse": 0.0,
    r"igd": 0.0,
    r"shd": 0.0,
    r"rel_err": 0.0,
    r"relative.l2": 0.0,
    r"chi2": 0.0,
    r"lpips": 0.0,
    r"eer": 0.0,
    r"mel_loss": 0.0,
    r"mpjpe": 0.0,
    r"pehe": 0.0,
    r"ate_error": 0.0,
    r"risk_certificate": 0.0,
    r"kl_divergence": 0.0,
    r"ce_bound": 0.0,
    r"degradation": 0.0,
    r"asr": 1.0,       # attack success rate: lower-is-better but bounded [0,1]
    r"mia_auc": 1.0,   # membership inference AUC
    r"privacy_gap": 1.0,
    r"poison_fit": 1.0,
    r"defense_score": 1.0,
    r"unlearn_score": 1.0,
    r"privacy_score": 1.0,
    r"robust_score": 1.0,
    r"perplexity": 1.0,  # bounded at 1 (lower)
    r"ppl": 1.0,
    r"coverage": 1.0,
    r"epsilon": 0.0,     # DP epsilon, lower = better, bounded at 0
}


def _infer_direction(metric: str, values: list[float] | None = None, task_name: str = "") -> str | None:
    """Infer higher/lower from metric name. Returns None if unsure."""
    m_lower = metric.lower()
    task_lower = task_name.lower()
    if "max_drawdown" in m_lower:
        if values and max(values) <= 0.0:
            return "higher"
        return "lower"
    if "asr" in m_lower:
        return "higher" if "attack" in task_lower else "lower"
    if "best_fitness" in m_lower:
        return "lower"
    if "spread" in m_lower and "multi-objective" in task_lower:
        return "lower"
    # Check lower patterns first (more specific)
    for pat in LOWER_PATTERNS:
        if re.search(pat, m_lower):
            # But check it's not a false positive against higher patterns
            # e.g. "convergence_auc" should be higher
            for hpat in HIGHER_PATTERNS:
                if re.search(hpat, m_lower):
                    return "higher"
            return "lower"
    for pat in HIGHER_PATTERNS:
        if re.search(pat, m_lower):
            return "higher"
    return None


def _infer_bound(
    metric: str,
    direction: str | None,
    values: list[float] | None = None,
) -> float | None:
    """Infer theoretical bound from metric name."""
    m_lower = metric.lower()
    for pat, bound in BOUNDED_MAP.items():
        if re.search(pat, m_lower):
            if bound == 100.0 and values and max(values) <= 1.5:
                return 1.0
            if bound == 1.0 and values and max(values) > 1.5 and (
                "auc" in m_lower or "mrr" in m_lower or "hits" in m_lower
            ):
                return 100.0
            return bound
    return None


def _infer_setting(metric: str, labels: list[str]) -> str:
    """Try to match metric suffix to a test_cmd label."""
    for label in labels:
        # Normalize label for matching
        label_norm = label.replace("-", "_").replace(".", "_").lower()
        metric_norm = metric.replace("-", "_").replace(".", "_").lower()
        if label_norm in metric_norm:
            return label
    # Fallback: use everything after the first underscore as setting
    return "default"


def _safe_term_name(metric: str) -> str:
    """Convert metric column name to a valid Python identifier."""
    return re.sub(r"[^a-zA-Z0-9_]", "_", metric).strip("_")


# ---------------------------------------------------------------------------
# Main autogen function
# ---------------------------------------------------------------------------

def autogen_score_spec(
    task_name: str,
    tasks_dir: Path | None = None,
    dry_run: bool = False,
) -> str:
    """Generate a score_spec.py template for a task.

    Returns the generated Python source code.
    If not dry_run, writes it to tasks/<task>/score_spec.py.
    """
    from mlsbench import PROJECT_ROOT
    if tasks_dir is None:
        tasks_dir = PROJECT_ROOT / "tasks"

    task_dir = tasks_dir / task_name
    anchors = BaselineAnchors(task_dir)

    # Load test_cmd labels
    labels: list[str] = []
    cfg_path = task_dir / "config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
        for tc in cfg.get("test_cmds", []):
            lbl = tc.get("label", "")
            if lbl:
                labels.append(lbl)

    metrics = anchors.metric_columns()
    if not metrics:
        return f"# No metrics found in leaderboard for {task_name}\n"

    # Classify metrics
    terms_by_setting: dict[str, list[dict]] = {}
    all_terms: list[dict] = []

    for metric in metrics:
        anchors_for_metric = anchors.get(metric)
        values = anchors_for_metric.values if anchors_for_metric else []
        direction = _infer_direction(metric, values, task_name)
        bound = _infer_bound(metric, direction, values)
        setting_name = _infer_setting(metric, labels)
        term_name = _safe_term_name(metric)

        info = {
            "metric": metric,
            "term_name": term_name,
            "direction": direction or "higher",  # default higher if unknown
            "bound": bound,
            "setting": setting_name,
        }
        all_terms.append(info)
        terms_by_setting.setdefault(setting_name, []).append(info)

    # Generate code
    lines: list[str] = []
    lines.append(f'"""Score spec for {task_name} (auto-generated)."""')
    lines.append("from mlsbench.scoring.dsl import *")
    lines.append("")

    # Terms
    for info in all_terms:
        if info["bound"] is not None:
            lines.append(f'term("{info["term_name"]}",')
            lines.append(f'    col("{info["metric"]}").{info["direction"]}().id()')
            lines.append(f'    .bounded_power(bound={info["bound"]}))')
        else:
            lines.append(f'term("{info["term_name"]}",')
            lines.append(f'    col("{info["metric"]}").{info["direction"]}().id()')
            lines.append("    .sigmoid())")
        lines.append("")

    # Settings
    for setting_name, terms in terms_by_setting.items():
        objective_terms = [(t["term_name"], 1.0) for t in terms]
        terms_str = ", ".join(f'("{t}", {w})' for t, w in objective_terms)
        lines.append(f'setting("{setting_name}", weighted_mean({terms_str}))')

    lines.append("")

    # Task
    setting_names = list(terms_by_setting.keys())
    settings_str = ", ".join(f'"{s}"' for s in setting_names)
    lines.append(f"task(gmean({settings_str}))")
    lines.append("")

    source = "\n".join(lines)

    if not dry_run:
        out_path = task_dir / "score_spec.py"
        out_path.write_text(source)

    return source
