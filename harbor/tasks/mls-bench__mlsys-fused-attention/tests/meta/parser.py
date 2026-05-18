"""Output parser for mlsys-fused-attention."""

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mlsbench.agent.parsers import OutputParser, ParseResult


class Parser(OutputParser):
    """Parse fused attention kernel benchmark output."""

    def parse(self, cmd_label: str, raw_output: str) -> ParseResult:
        feedback_parts = []
        metrics = {}

        # --- TRAIN_METRICS (feedback only) ---
        train_lines = [
            l.strip() for l in raw_output.splitlines()
            if l.strip().startswith("TRAIN_METRICS:")
        ]
        if train_lines:
            feedback_parts.append(
                f"Kernel diagnostics ({cmd_label}):\n" +
                "\n".join(train_lines[-5:])
            )

        # --- TEST_METRICS (leaderboard metrics) ---
        for line in raw_output.splitlines():
            line = line.strip()
            if line.startswith("TEST_METRICS:"):
                for match in re.finditer(
                    r"(\w+)=([\d.eE+-]+|nan|inf|-inf)", line
                ):
                    key, val_str = match.group(1), match.group(2)
                    try:
                        val = float(val_str)
                    except ValueError:
                        continue
                    metric_key = f"{key}_{cmd_label}"
                    metrics[metric_key] = val
                    feedback_parts.append(f"{metric_key}: {val}")

        # Highlight correctness status
        correct_key = f"correct_{cmd_label}"
        if correct_key in metrics:
            if metrics[correct_key] >= 1.0:
                feedback_parts.append(f"Correctness: PASSED")
            else:
                feedback_parts.append(
                    f"Correctness: FAILED — kernel output diverges from "
                    f"reference (max_diff > 1e-2). Fix numerical issues "
                    f"before optimizing throughput."
                )

        # Fallback
        if not feedback_parts:
            feedback_parts.append(raw_output[-3000:])

        return ParseResult(
            feedback="\n".join(feedback_parts),
            metrics=metrics,
        )
