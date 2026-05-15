"""Task-specific output parser for ml-missing-data-imputation.

Handles combined train+eval output from missing data imputation:
- Training feedback: TRAIN_METRICS stage=... / rmse=... / downstream_score=...
- Test feedback: TEST_METRICS rmse=... downstream_score=...

Metrics are keyed by dataset label, e.g. rmse_breast_cancer, downstream_score_wine.
"""

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mlsbench.agent.parsers import OutputParser, ParseResult


class Parser(OutputParser):
    """Parser for the ml-missing-data-imputation task."""

    def parse(self, cmd_label: str, raw_output: str) -> ParseResult:
        feedback_parts = []
        metrics: dict = {}

        train_feedback = self._parse_train_metrics(raw_output)
        if train_feedback:
            feedback_parts.append(train_feedback)

        eval_feedback, eval_metrics = self._parse_test_metrics(raw_output, cmd_label)
        if eval_feedback:
            feedback_parts.append(eval_feedback)
        metrics.update(eval_metrics)

        feedback = "\n".join(feedback_parts) if feedback_parts else raw_output
        return ParseResult(feedback=feedback, metrics=metrics)

    def _parse_train_metrics(self, output: str) -> str:
        lines = [l.strip() for l in output.splitlines() if l.strip().startswith("TRAIN_METRICS ")]
        if not lines:
            return ""
        return "Training metrics (last lines):\n" + "\n".join(lines[-5:])

    def _parse_test_metrics(self, output: str, cmd_label: str) -> tuple[str, dict]:
        metrics: dict = {}
        feedback_lines = []

        for line in output.splitlines():
            line = line.strip()
            if not line.startswith("TEST_METRICS"):
                continue
            # Parse key=value pairs
            for match in re.finditer(r"(\w+)=([\d.eE+-]+)", line):
                key, val = match.group(1), float(match.group(2))
                metric_key = f"{key}_{cmd_label}"
                metrics[metric_key] = val
                feedback_lines.append(f"  {key}: {val:.6f}")

        feedback = ""
        if feedback_lines:
            feedback = f"Test results ({cmd_label}):\n" + "\n".join(feedback_lines)

        return feedback, metrics
