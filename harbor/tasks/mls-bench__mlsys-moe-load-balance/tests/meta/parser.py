"""Output parser for mlsys-moe-load-balance."""

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mlsbench.agent.parsers import OutputParser, ParseResult


class Parser(OutputParser):
    """Parse MoE EPLB benchmark output."""

    def parse(self, cmd_label: str, raw_output: str) -> ParseResult:
        feedback_parts = []
        metrics = {}

        # --- TRAIN_METRICS (progress feedback) ---
        train_lines = [
            l.strip()
            for l in raw_output.splitlines()
            if l.strip().startswith("TRAIN_METRICS")
        ]
        if train_lines:
            feedback_parts.append(
                f"Progress ({cmd_label}):\n" + "\n".join(train_lines[-5:])
            )

        # --- TEST_METRICS (final results) ---
        for line in raw_output.splitlines():
            line = line.strip()
            if line.startswith("TEST_METRICS"):
                for match in re.finditer(r"(\w+)=([\d.eE+-]+)", line):
                    key, val = match.group(1), float(match.group(2))
                    metric_key = f"{key}_{cmd_label}"
                    metrics[metric_key] = val
                    feedback_parts.append(f"{metric_key}: {val:.6f}")

        # --- Fallback ---
        if not feedback_parts:
            feedback_parts.append(raw_output[-3000:])

        return ParseResult(
            feedback="\n".join(feedback_parts),
            metrics=metrics,
        )
