"""Task-specific output parser for meta-inner-loop-optimizer.

Handles combined train+eval output from gradient-based meta-learning:
- Training feedback: TRAIN_METRICS iter=I train_loss=L train_acc=A val_acc=V val_ci95=C
- Test feedback: TEST_METRICS accuracy=A ci95=C

Metrics are keyed by setting label, e.g. accuracy_mini_imagenet_1shot.
"""

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mlsbench.agent.parsers import OutputParser, ParseResult


class Parser(OutputParser):
    """Parser for the meta-inner-loop-optimizer task."""

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

        feedback = "\n".join(feedback_parts) if feedback_parts else raw_output[-3000:]
        return ParseResult(feedback=feedback, metrics=metrics)

    def _parse_train_metrics(self, output: str) -> str:
        lines = [l.strip() for l in output.splitlines() if l.strip().startswith("TRAIN_METRICS ")]
        if not lines:
            return ""
        return "Training metrics (last steps):\n" + "\n".join(lines[-5:])

    def _parse_test_metrics(self, output: str, cmd_label: str) -> tuple[str, dict]:
        metrics: dict = {}
        feedback = ""

        for line in output.splitlines():
            match = re.search(r"TEST_METRICS\s+accuracy=([\d.]+)\s+ci95=([\d.]+)", line)
            if match:
                accuracy = float(match.group(1))
                ci95 = float(match.group(2))
                metric_key = "accuracy_" + cmd_label.replace("-", "_")
                metrics[metric_key] = accuracy
                feedback = (
                    f"Test results ({cmd_label}):\n"
                    f"  Accuracy: {accuracy:.4f} ({100 * accuracy:.2f}%) "
                    f"+/- {100 * ci95:.2f}%"
                )

        return feedback, metrics
