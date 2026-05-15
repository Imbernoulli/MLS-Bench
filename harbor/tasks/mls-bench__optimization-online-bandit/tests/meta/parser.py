"""Task-specific output parser for opt-online-bandit.

Handles combined train+eval output from bandit algorithms:

Training feedback: lines matching
    TRAIN_METRICS step=N cumulative_regret=X normalized_regret=Y

Final metrics: lines matching
    TEST_METRICS cumulative_regret=X normalized_regret=Y

Metrics are keyed by environment name, e.g. normalized_regret_stochastic_mab.
Lower regret is better.
"""

import re
import sys
from pathlib import Path

# Allow importing from mlsbench package when run standalone
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mlsbench.agent.parsers import OutputParser, ParseResult


class Parser(OutputParser):
    """Parser for the opt-online-bandit task."""

    def parse(self, cmd_label: str, raw_output: str) -> ParseResult:
        feedback_parts = []
        metrics: dict = {}

        # Parse training metrics
        train_feedback = self._parse_train_metrics(raw_output)
        if train_feedback:
            feedback_parts.append(train_feedback)

        # Parse test metrics
        test_feedback, test_metrics = self._parse_test_metrics(raw_output, cmd_label)
        if test_feedback:
            feedback_parts.append(test_feedback)
        metrics.update(test_metrics)

        if feedback_parts:
            feedback = "\n".join(feedback_parts)
        else:
            feedback = raw_output

        return ParseResult(feedback=feedback, metrics=metrics)

    def _parse_train_metrics(self, output: str) -> str:
        """Extract TRAIN_METRICS lines and return a summary of the last few."""
        lines = []
        for line in output.splitlines():
            if line.strip().startswith("TRAIN_METRICS "):
                lines.append(line.strip())

        if not lines:
            return ""

        summary_lines = lines[-5:]
        return "Training metrics (last steps):\n" + "\n".join(summary_lines)

    def _parse_test_metrics(self, output: str, cmd_label: str) -> tuple[str, dict]:
        """Extract TEST_METRICS lines and return feedback + metrics.

        Expected format: TEST_METRICS cumulative_regret=X normalized_regret=Y
        """
        metrics: dict = {}
        feedback_parts: list[str] = []

        for line in output.splitlines():
            line = line.strip()
            if line.startswith("TEST_METRICS"):
                for match in re.finditer(r"(\w+)=([\d.eE+-]+)", line):
                    key, val = match.group(1), float(match.group(2))
                    metric_key = f"{key}_{cmd_label.replace('-', '_')}"
                    metrics[metric_key] = val
                    feedback_parts.append(f"{metric_key}: {val:.6f}")

        feedback = ""
        if feedback_parts:
            feedback = f"Final metrics ({cmd_label}):\n" + "\n".join(feedback_parts)

        return feedback, metrics
