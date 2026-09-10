"""Output parser for ml-selective-deferral.

Parses TRAIN_METRICS and TEST_METRICS lines from the selective prediction
benchmark.
"""

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mlsbench.agent.parsers import OutputParser, ParseResult


class Parser(OutputParser):
    """Parser for selective deferral metrics."""

    def parse(self, cmd_label: str, raw_output: str) -> ParseResult:
        feedback_parts = []
        metrics: dict = {}

        train_feedback = self._parse_prefixed_metrics(raw_output, "TRAIN_METRICS:")
        if train_feedback:
            feedback_parts.append(train_feedback)

        test_feedback, test_metrics = self._parse_test_metrics(raw_output, cmd_label)
        if test_feedback:
            feedback_parts.append(test_feedback)
        metrics.update(test_metrics)

        feedback = "\n".join(feedback_parts) if feedback_parts else raw_output[-3000:]
        return ParseResult(feedback=feedback, metrics=metrics)

    def _parse_prefixed_metrics(self, output: str, prefix: str) -> str:
        lines = [line.strip() for line in output.splitlines() if line.strip().startswith(prefix)]
        if not lines:
            return ""
        return "Training progress:\n" + "\n".join(lines[-5:])

    def _parse_test_metrics(self, output: str, cmd_label: str) -> tuple[str, dict]:
        metrics: dict = {}
        feedback = ""

        for line in output.splitlines():
            if "TEST_METRICS:" not in line:
                continue
            pairs = re.findall(r"([A-Za-z0-9_@-]+)=([\d.]+(?:e[+-]?\d+)?|nan|inf|-inf)", line, re.IGNORECASE)
            if not pairs:
                continue
            for key, raw in pairs:
                metrics[f"{key}_{cmd_label}"] = float(raw.lower())
            parts = [f"{k}={v:.6f}" for k, v in metrics.items()]
            feedback = f"Final metrics ({cmd_label}): " + ", ".join(parts)

        return feedback, metrics
