"""Output parser for mlsys-sparse-attention-inference.

Parses TRAIN_METRICS / TEST_METRICS / DENSITY_STATS lines emitted by
run_llm.py, run_vit.py, run_dit.py.
"""

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mlsbench.agent.parsers import OutputParser, ParseResult


_NUM = r"[\d.eE+-]+|nan|inf|-inf"


class Parser(OutputParser):
    """Parser for sparse-attention inference task."""

    def parse(self, cmd_label: str, raw_output: str) -> ParseResult:
        feedback = []
        metrics: dict = {}

        # ── TRAIN_METRICS (running progress) ──────────────────────────
        train_lines = [l.strip() for l in raw_output.splitlines()
                       if l.strip().startswith("TRAIN_METRICS")]
        if train_lines:
            feedback.append("Progress (" + cmd_label + "):\n" +
                            "\n".join(train_lines[-5:]))

        # ── DENSITY_STATS (info only, also parse density into metrics) ─
        for line in raw_output.splitlines():
            line = line.strip()
            if line.startswith("DENSITY_STATS"):
                feedback.append(line)

        # ── TEST_METRICS (final, leaderboard) ─────────────────────────
        for line in raw_output.splitlines():
            line = line.strip()
            if not line.startswith("TEST_METRICS"):
                continue
            # match key=value pairs (allow scientific + nan/inf)
            for m in re.finditer(rf"(\w+)=({_NUM})", line):
                key, raw = m.group(1), m.group(2)
                try:
                    val = float(raw)
                except ValueError:
                    continue
                # Keys in run_*.py are already namespaced (llm_ppl, vit_acc,
                # dit_lpips, etc) — keep as-is so columns align across labels.
                metrics[key] = val
            if metrics:
                summary = ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
                feedback.append(f"Final ({cmd_label}): {summary}")

        if not feedback:
            feedback.append(raw_output[-2000:])

        return ParseResult(feedback="\n".join(feedback), metrics=metrics)
