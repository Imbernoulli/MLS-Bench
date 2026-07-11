"""Output parser for the machine-translation (mt-*) tasks.

Harness emits one metric line followed by a completion line per run:
    MT_METRICS bleu=<B> chrf=<C> n_pairs=<N> plen=<W> elapsed=<T>
    MT_SETTING_COMPLETE direction=<LABEL>

Leaderboard metric (higher is better): bleu_{label}  (corpus sacreBLEU, 0-100).
chrf_{label} is reported as a secondary metric.
"""
from __future__ import annotations

import math
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mlsbench.agent.parsers import OutputParser, ParseResult


class Parser(OutputParser):
    def parse(self, cmd_label: str, raw_output: str) -> ParseResult:
        matches = []
        completions = []
        for line in raw_output.splitlines():
            m = re.fullmatch(
                r"MT_METRICS\s+bleu=([\d.eE+-]+)\s+chrf=([\d.eE+-]+)\s+"
                r"n_pairs=(\S+)\s+plen=(\S+)\s+elapsed=(\S+)",
                line.strip(),
            )
            if m:
                matches.append(m)
            complete = re.fullmatch(
                r"MT_SETTING_COMPLETE\s+direction=(\S+)", line.strip()
            )
            if complete:
                completions.append(complete.group(1))

        metrics: dict = {}
        feedback = ""
        if len(matches) == 1:
            match = matches[0]
            bleu = float(match.group(1))
            chrf = float(match.group(2))
            n_pairs = int(match.group(3)) if match.group(3).isdigit() else -1
            try:
                pred_len = float(match.group(4))
            except ValueError:
                pred_len = float("nan")
            try:
                elapsed = float(match.group(5))
            except ValueError:
                elapsed = float("nan")
            valid = (
                cmd_label in {"de_en", "fr_en", "ru_en"}
                and completions == [cmd_label]
                and n_pairs == 2000
                and math.isfinite(bleu)
                and math.isfinite(chrf)
                and math.isfinite(pred_len)
                and math.isfinite(elapsed)
                and 0.0 <= bleu <= 100.0
                and 0.0 <= chrf <= 100.0
                and pred_len >= 0.0
                and elapsed > 0.0
            )
            if valid:
                metrics[f"bleu_{cmd_label}"] = bleu
                metrics[f"chrf_{cmd_label}"] = chrf
                feedback = (
                    f"Results ({cmd_label}):\n"
                    f"  corpus sacreBLEU: {bleu:.4f}   (higher is better)\n"
                    f"  chrF:             {chrf:.4f}\n"
                    f"  n_pairs: {n_pairs}   mean_pred_len_words: {pred_len:g}\n"
                    f"  elapsed_seconds: {elapsed:g}"
                )
            else:
                feedback = "Rejected incomplete or malformed translation evaluation output."
        elif matches or completions:
            feedback = "Rejected ambiguous translation evaluation output."

        trace = [ln.strip() for ln in raw_output.splitlines()
                 if ln.strip().startswith(
                     ("MT_DATA", "MT_BEAM", "MT_LENGTH", "MT_STRATEGY"))]
        if trace:
            feedback = (feedback + "\n" + "\n".join(trace[-6:])) if feedback else "\n".join(trace[-6:])

        if not feedback:
            feedback = raw_output[-3000:]

        return ParseResult(feedback=feedback, metrics=metrics)
