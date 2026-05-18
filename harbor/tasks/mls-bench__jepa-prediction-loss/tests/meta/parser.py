"""Task-specific output parser for jepa-prediction-loss.

Training feedback: lines matching
    TRAIN_METRICS epoch=E loss=L vc_loss=V pred_loss=P

Evaluation feedback: lines matching
    TEST_METRICS: mean_detection_ap=X.XXXX

Leaderboard metric: mean_detection_ap_moving-mnist
"""

import re
import sys
from pathlib import Path

# Allow importing from mlsbench package when run standalone
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mlsbench.agent.parsers import OutputParser, ParseResult


class Parser(OutputParser):
    """Parser for the jepa-prediction-loss task."""

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

        # Return last 5 training metric lines as feedback
        summary_lines = lines[-5:]
        return "Training metrics (last epochs):\n" + "\n".join(summary_lines)

    def _parse_test_metrics(self, output: str, cmd_label: str) -> tuple[str, dict]:
        """Extract TEST_METRICS lines and return feedback + metrics.

        Expected format: TEST_METRICS: mean_detection_ap=X.XXXX
        """
        mean_ap_values: list[float] = []
        test_lines: list[str] = []

        for line in output.splitlines():
            match = re.search(
                r"TEST_METRICS:\s+mean_detection_ap=([\d.]+)", line
            )
            if match:
                test_lines.append(line.strip())
                mean_ap_values.append(float(match.group(1)))

        metrics: dict = {}
        feedback = ""

        if mean_ap_values:
            final_ap = mean_ap_values[-1]
            metric_key = f"mean_detection_ap_{cmd_label}"
            metrics[metric_key] = final_ap

            feedback = "Test evaluation:\n" + "\n".join(test_lines[-3:])
            feedback += f"\nFinal mean detection AP: {final_ap:.4f}"

        return feedback, metrics
