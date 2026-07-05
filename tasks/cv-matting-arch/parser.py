"""Output parser for the cv-matting-* trimap-guided image-matting tasks.

The harness emits one metric line per run:
    MATTING_METRICS surface=<S> setting=<L> sad=<..> mse=<..> grad=<..> unk_frac=<..>

Leaderboard metrics: sad_{label} (alpha SAD in the trimap unknown band, /1000,
LOWER better — primary scored), mse_{label} (alpha MSE *1e3, LOWER better),
grad_{label} (alpha gradient error /1000, LOWER better). It also records
CONST_HALF_SAD / MEAN_ALPHA_SAD (SAD of degenerate constant-0.5 / mean-alpha
predictors) as context — a real matting net must beat them.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mlsbench.agent.parsers import OutputParser, ParseResult

_PAT = re.compile(
    r"MATTING_METRICS\s+surface=(\S+)\s+setting=(\S+)\s+"
    r"sad=([\d.eE+-]+)\s+mse=([\d.eE+-]+)\s+grad=([\d.eE+-]+)\s+"
    r"unk_frac=([\d.eE+-]+)"
)
_CH = re.compile(r"CONST_HALF_SAD=([\d.eE+-]+)")
_MA = re.compile(r"MEAN_ALPHA_SAD=([\d.eE+-]+)")


class Parser(OutputParser):
    """Parser for the trimap-guided image-matting tasks (alpha SAD / MSE / Grad)."""

    def parse(self, cmd_label: str, raw_output: str) -> ParseResult:
        metrics: dict = {}
        feedback = ""
        const_half = None
        mean_alpha = None

        for line in raw_output.splitlines():
            ch = _CH.search(line)
            if ch:
                const_half = float(ch.group(1))
            ma = _MA.search(line)
            if ma:
                mean_alpha = float(ma.group(1))
            m = _PAT.search(line)
            if m:
                sad = float(m.group(3)); mse = float(m.group(4))
                grad = float(m.group(5)); unk = float(m.group(6))
                metrics[f"sad_{cmd_label}"] = sad
                metrics[f"mse_{cmd_label}"] = mse
                metrics[f"grad_{cmd_label}"] = grad
                degen = ""
                if const_half is not None:
                    degen = (f"   (constant-0.5 SAD={const_half:.3f}"
                             f", mean-alpha SAD={mean_alpha:.3f})"
                             if mean_alpha is not None
                             else f"   (constant-0.5 SAD={const_half:.3f})")
                feedback = (
                    f"Matting results ({cmd_label}):\n"
                    f"  surface: {m.group(1)}  setting: {m.group(2)}\n"
                    f"  SAD:  {sad:.4f}   (alpha SAD in the unknown band /1000, LOWER is better){degen}\n"
                    f"  MSE:  {mse:.4f}   (alpha MSE *1e3, LOWER is better)\n"
                    f"  Grad: {grad:.4f}   (alpha gradient error /1000, LOWER is better)\n"
                    f"  unknown_frac: {unk:.3f}"
                )

        trace_keys = ("LOSS_APPLIED", "LOSS_FALLBACK", "TRIMAP_APPLIED", "TRIMAP_FALLBACK",
                      "DECODER_APPLIED", "DECODER_FALLBACK", "INPUT_CHANNELS", "DATA ",
                      "train it=")
        trace = [ln.strip() for ln in raw_output.splitlines()
                 if any(k in ln for k in trace_keys)]
        if trace:
            tail = "\n".join(trace[-6:])
            feedback = (feedback + "\n" + tail) if feedback else tail

        if not feedback:
            feedback = raw_output[-3000:]

        return ParseResult(feedback=feedback, metrics=metrics)
