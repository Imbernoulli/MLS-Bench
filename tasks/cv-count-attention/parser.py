"""Output parser for the cv-count-* density-map crowd/object-counting tasks.

The harness emits one metric line per run:
    COUNT_METRICS surface=<S> setting=<L> mae=<..> rmse=<..> nae=<..> gt_mean=<..> pred_mean=<..>

Leaderboard metrics: mae_{label} (counting MAE, LOWER better — primary scored),
rmse_{label} (LOWER better), nae_{label} (normalised MAE, LOWER better). It also
records CONST_MEAN_MAE (the MAE of a degenerate mean-count predictor) as context.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mlsbench.agent.parsers import OutputParser, ParseResult

_PAT = re.compile(
    r"COUNT_METRICS\s+surface=(\S+)\s+setting=(\S+)\s+"
    r"mae=([\d.eE+-]+)\s+rmse=([\d.eE+-]+)\s+nae=([\d.eE+-]+)\s+"
    r"gt_mean=([\d.eE+-]+)\s+pred_mean=([\d.eE+-]+)"
)
_CMM = re.compile(r"CONST_MEAN_MAE=([\d.eE+-]+)")


class Parser(OutputParser):
    """Parser for the density-map counting tasks (counting MAE / RMSE / NAE)."""

    def parse(self, cmd_label: str, raw_output: str) -> ParseResult:
        metrics: dict = {}
        feedback = ""
        const_mae = None

        for line in raw_output.splitlines():
            cm = _CMM.search(line)
            if cm:
                const_mae = float(cm.group(1))
            m = _PAT.search(line)
            if m:
                mae = float(m.group(3)); rmse = float(m.group(4))
                nae = float(m.group(5)); gt_mean = float(m.group(6))
                pred_mean = float(m.group(7))
                metrics[f"mae_{cmd_label}"] = mae
                metrics[f"rmse_{cmd_label}"] = rmse
                metrics[f"nae_{cmd_label}"] = nae
                cm_str = f"   (constant-mean predictor MAE={const_mae:.2f})" if const_mae else ""
                feedback = (
                    f"Counting results ({cmd_label}):\n"
                    f"  surface: {m.group(1)}  setting: {m.group(2)}\n"
                    f"  MAE:  {mae:.4f}   (counting mean-absolute-error, LOWER is better){cm_str}\n"
                    f"  RMSE: {rmse:.4f}   (LOWER is better)\n"
                    f"  NAE:  {nae:.4f}   (MAE / gt_mean, LOWER is better)\n"
                    f"  gt_mean_count: {gt_mean:.2f}   pred_mean_count: {pred_mean:.2f}"
                )

        trace_keys = ("HEAD_APPLIED", "HEAD_FALLBACK", "HEAD_MODE",
                      "OUTPUT_APPLIED", "OUTPUT_FALLBACK", "DATA ", "train it=")
        trace = [ln.strip() for ln in raw_output.splitlines()
                 if any(k in ln for k in trace_keys)]
        if trace:
            tail = "\n".join(trace[-6:])
            feedback = (feedback + "\n" + tail) if feedback else tail

        if not feedback:
            feedback = raw_output[-3000:]

        return ParseResult(feedback=feedback, metrics=metrics)
