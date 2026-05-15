"""Parser for llm-scaling-law-discovery."""

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mlsbench.agent.parsers import OutputParser, ParseResult


class Parser(OutputParser):
    """Parse scaling-law benchmark output."""

    def parse(self, cmd_label: str, raw_output: str) -> ParseResult:
        feedback_parts = []
        metrics = {}

        train_lines = [
            line.strip()
            for line in raw_output.splitlines()
            if line.strip().startswith("TRAIN_METRICS")
        ]
        if train_lines:
            feedback_parts.append(
                f"Training progress ({cmd_label}):\n" + "\n".join(train_lines[-5:])
            )

        for line in raw_output.splitlines():
            line = line.strip()
            if not line.startswith("TEST_METRICS"):
                continue
            for key, raw_val in re.findall(r"(\w+)=([\d.eE+-]+)", line):
                metric_key = f"{key}_{cmd_label.replace('-', '_')}"
                metrics[metric_key] = float(raw_val)
            if metrics:
                pretty = ", ".join(f"{k}={v:.6f}" for k, v in metrics.items())
                feedback_parts.append(f"Final metrics ({cmd_label}): {pretty}")

        if not feedback_parts:
            feedback_parts.append(raw_output[-3000:])

        return ParseResult(feedback="\n".join(feedback_parts), metrics=metrics)
