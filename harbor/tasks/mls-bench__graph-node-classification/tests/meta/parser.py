"""Task-specific output parser for graph-node-classification.

Handles combined train+eval output from GNN node classification:
- Training feedback: TRAIN_METRICS epoch=E loss=L train_acc=A val_acc=A test_acc=A
- Test feedback: TEST_METRICS accuracy=A macro_f1=F

Metrics are keyed by dataset label, e.g. accuracy_Cora, macro_f1_Cora.
"""

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mlsbench.agent.parsers import OutputParser, ParseResult


class Parser(OutputParser):
    """Parser for the graph-node-classification task."""

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
        lines = [l.strip() for l in output.splitlines()
                 if l.strip().startswith("TRAIN_METRICS ")]
        if not lines:
            return ""
        return "Training metrics (last epochs):\n" + "\n".join(lines[-5:])

    def _parse_test_metrics(self, output: str, cmd_label: str) -> tuple[str, dict]:
        metrics: dict = {}
        feedback_parts = []

        for line in output.splitlines():
            line = line.strip()
            if not line.startswith("TEST_METRICS"):
                continue

            acc_match = re.search(r"accuracy=([\d.]+)", line)
            f1_match = re.search(r"macro_f1=([\d.]+)", line)

            if acc_match:
                accuracy = float(acc_match.group(1))
                key = f"accuracy_{cmd_label}"
                metrics[key] = accuracy
                feedback_parts.append(f"  Accuracy ({cmd_label}): {accuracy:.4f} ({100 * accuracy:.2f}%)")

            if f1_match:
                f1 = float(f1_match.group(1))
                key = f"macro_f1_{cmd_label}"
                metrics[key] = f1
                feedback_parts.append(f"  Macro F1 ({cmd_label}): {f1:.4f} ({100 * f1:.2f}%)")

        feedback = ""
        if feedback_parts:
            feedback = f"Test results ({cmd_label}):\n" + "\n".join(feedback_parts)

        return feedback, metrics
