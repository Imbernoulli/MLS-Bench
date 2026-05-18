"""Task-specific output parser for causal-treatment-effect.

Handles CATE estimation output:

Training feedback: lines matching
    TRAIN_METRICS rep=N PEHE=X.XXXXXX ATE_error=X.XXXXXX

Final metrics: lines matching
    TEST_METRICS PEHE=X.XXXXXX ATE_error=X.XXXXXX
"""

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mlsbench.agent.parsers import OutputParser, ParseResult


class Parser(OutputParser):
    """Parser for the causal-treatment-effect task."""

    def parse(self, cmd_label: str, raw_output: str) -> ParseResult:
        feedback_parts = []
        metrics: dict = {}

        # Parse training metrics (per-repetition)
        train_feedback = self._parse_train_metrics(raw_output)
        if train_feedback:
            feedback_parts.append(train_feedback)

        # Parse final test metrics
        test_feedback, test_metrics = self._parse_test_metrics(raw_output, cmd_label)
        if test_feedback:
            feedback_parts.append(test_feedback)
        metrics.update(test_metrics)

        if feedback_parts:
            feedback = "\n".join(feedback_parts)
        else:
            feedback = raw_output[-3000:]

        return ParseResult(feedback=feedback, metrics=metrics)

    def _parse_train_metrics(self, output: str) -> str:
        """Extract TRAIN_METRICS lines and return a summary."""
        lines = []
        for line in output.splitlines():
            if line.strip().startswith("TRAIN_METRICS "):
                lines.append(line.strip())

        if not lines:
            return ""

        summary_lines = lines[-5:]
        return "Per-repetition metrics (last 5):\n" + "\n".join(summary_lines)

    def _parse_test_metrics(self, output: str, cmd_label: str) -> tuple:
        """Extract TEST_METRICS line and return feedback + metrics dict."""
        metrics = {}
        feedback = ""

        for line in output.splitlines():
            line = line.strip()
            if line.startswith("TEST_METRICS"):
                for match in re.finditer(r"(\w+)=([\d.eE+-]+)", line):
                    key, val = match.group(1), float(match.group(2))
                    metric_key = f"{key}_{cmd_label}"
                    metrics[metric_key] = val

        if metrics:
            parts = [f"{k}: {v:.6f}" for k, v in metrics.items()]
            feedback = f"Final metrics ({cmd_label}):\n" + "\n".join(parts)

        return feedback, metrics
