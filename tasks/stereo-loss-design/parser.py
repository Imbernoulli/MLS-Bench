"""Output parser for the stereo-* disparity tasks.

Harness emits a single metric line per run:
    STEREO_METRICS task=<T> setting=<L> epe=<E> d1=<D1> acc1=<a1> steps=<n>

epe  = mean disparity end-point error in px (LOWER better) -> column epe_{label}
d1   = bad-pixel fraction, EPE>3px and >5% GT (LOWER better) -> d1_{label}
acc1 = fraction of pixels with EPE<1px (HIGHER better) -> acc1_{label}
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mlsbench.agent.parsers import OutputParser, ParseResult


class Parser(OutputParser):
    """Parser for the stereo-matching disparity tasks (EPE-based)."""

    def parse(self, cmd_label: str, raw_output: str) -> ParseResult:
        metrics: dict = {}
        feedback = ""

        for line in raw_output.splitlines():
            m = re.search(
                r"STEREO_METRICS\s+task=(\S+)\s+setting=(\S+)\s+epe=([\d.eE+-]+)\s+"
                r"d1=([\d.eE+-]+)\s+acc1=([\d.eE+-]+)\s+steps=(\S+)",
                line,
            )
            if m:
                epe = float(m.group(3))
                d1 = float(m.group(4))
                acc1 = float(m.group(5))
                metrics[f"epe_{cmd_label}"] = epe
                metrics[f"d1_{cmd_label}"] = d1
                metrics[f"acc1_{cmd_label}"] = acc1
                feedback = (
                    f"Stereo disparity results ({cmd_label}):\n"
                    f"  task:      {m.group(1)}\n"
                    f"  steps:     {m.group(6)}\n"
                    f"  EPE:       {epe:.4f} px   (disparity end-point error, LOWER is better)\n"
                    f"  D1:        {d1:.4f}      (bad-pixel fraction, LOWER is better)\n"
                    f"  acc@1px:   {acc1:.4f}      (fraction of pixels within 1px)"
                )

        trace = [ln.strip() for ln in raw_output.splitlines()
                 if ln.strip().startswith(("DATA_LOADED", "MODEL_BUILT",
                                           "SOLUTION_FALLBACK"))]
        if trace:
            extra = "\n".join(trace[-3:])
            feedback = (feedback + "\n" + extra) if feedback else extra

        if not feedback:
            feedback = raw_output[-3000:]

        return ParseResult(feedback=feedback, metrics=metrics)
