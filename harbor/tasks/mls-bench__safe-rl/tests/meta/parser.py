"""Task-specific output parser for safe-rl.

Handles training output from OmniSafe CustomLag algorithm:

Training feedback: lines matching
    TRAIN_METRICS epoch=N ep_ret=X.XXXX ep_cost=Y.YYYY ep_len=Z.Z

Final metrics: lines matching
    TEST_METRICS ep_ret=X.XXXX ep_cost=Y.YYYY ep_len=Z.Z
"""

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mlsbench.agent.parsers import OutputParser, ParseResult


class Parser(OutputParser):
    """Parser for the safe-rl task."""

    def parse(self, cmd_label: str, raw_output: str) -> ParseResult:
        feedback_parts = []
        metrics: dict = {}

        # Parse training metrics
        train_feedback = self._parse_train_metrics(raw_output)
        if train_feedback:
            feedback_parts.append(train_feedback)

        # Parse test metrics (final evaluation)
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
        return "Training metrics (last epochs):\n" + "\n".join(summary_lines)

    def _parse_test_metrics(self, output: str, cmd_label: str = "") -> tuple[str, dict]:
        """Extract TEST_METRICS line and return feedback + metrics.

        Expected format: TEST_METRICS ep_ret=X.XXXX ep_cost=Y.YYYY ep_len=Z.Z
        """
        metrics: dict = {}
        feedback = ""

        for line in output.splitlines():
            if line.strip().startswith("TEST_METRICS "):
                # Parse ep_ret
                ret_match = re.search(
                    r"ep_ret=(-?[\d.]+(?:e[+-]?\d+)?)", line, re.IGNORECASE,
                )
                if ret_match:
                    ret_key = "ep_ret_" + cmd_label.replace("-", "_")
                    metrics[ret_key] = float(ret_match.group(1))

                # Parse ep_cost
                cost_match = re.search(
                    r"ep_cost=(-?[\d.]+(?:e[+-]?\d+)?)", line, re.IGNORECASE,
                )
                if cost_match:
                    cost_key = "ep_cost_" + cmd_label.replace("-", "_")
                    metrics[cost_key] = float(cost_match.group(1))

                feedback = f"Final evaluation:\n  {line.strip()}"

        return feedback, metrics
