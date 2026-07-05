"""Output parser for the text-simplification (simp-*) tasks.

The harness emits one metric line PER SETTING (asset / turk / wiki):
    SIMP_METRICS setting=<S> sari=<V> bleu=<B> n_sents=<N> plen=<W> lenratio=<R>

Leaderboard metric (higher is better): sari_{setting}  (corpus SARI, 0-100, the
standard reference-based simplification metric, Xu et al. 2016). bleu_{setting}
(adequacy sanity BLEU) is reported as a secondary metric.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mlsbench.agent.parsers import OutputParser, ParseResult


class Parser(OutputParser):
    def parse(self, cmd_label: str, raw_output: str) -> ParseResult:
        metrics: dict = {}
        lines_fb = []

        for line in raw_output.splitlines():
            m = re.search(
                r"SIMP_METRICS\s+setting=(\w+)\s+sari=([\d.eE+-]+)\s+"
                r"bleu=([\d.eE+-]+)\s+n_sents=(\S+)\s+plen=(\S+)\s+lenratio=(\S+)",
                line,
            )
            if m:
                setting = m.group(1)
                sari = float(m.group(2))
                bleu = float(m.group(3))
                metrics[f"sari_{setting}"] = sari
                metrics[f"bleu_{setting}"] = bleu
                lines_fb.append(
                    f"  [{setting}] SARI={sari:.4f}  BLEU={bleu:.4f}  "
                    f"n={m.group(4)}  plen={m.group(5)}  lenratio={m.group(6)}"
                )

        feedback = ""
        if lines_fb:
            feedback = "Results (SARI higher is better):\n" + "\n".join(lines_fb)

        trace = [ln.strip() for ln in raw_output.splitlines()
                 if ln.strip().startswith(
                     ("SIMP_POLICY", "SIMP_BEAM", "SIMP_LENGTH", "SIMP_DONE"))]
        if trace:
            feedback = (feedback + "\n" + "\n".join(trace[-4:])) if feedback else "\n".join(trace[-4:])

        if not feedback:
            feedback = raw_output[-3000:]

        return ParseResult(feedback=feedback, metrics=metrics)
