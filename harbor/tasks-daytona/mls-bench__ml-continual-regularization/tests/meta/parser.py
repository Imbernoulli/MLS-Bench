"""Task-specific output parser for cl-regularization.

Training feedback: lines matching
    TRAIN_METRICS seed=S experiment=E scenario=S contexts=C

Evaluation feedback: lines matching
    TEST_METRICS average_accuracy=A
    TEST_METRICS context_N_accuracy=A

Also parses the native codebase output:
    - Context N: X.XXXX
    => average accuracy over all N contexts: X.XXXX

Leaderboard metric: average_accuracy_<env> (higher is better).
Metrics are prefixed with cmd_label to avoid collisions across envs.
"""

import re
import sys
from pathlib import Path

# Allow importing from mlsbench package when run standalone
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mlsbench.agent.parsers import OutputParser, ParseResult


class Parser(OutputParser):
    """Parser for the cl-regularization task."""

    def parse(self, cmd_label: str, raw_output: str) -> ParseResult:
        feedback_parts = []
        metrics: dict = {}

        # Parse native per-context accuracy lines
        context_feedback, context_metrics = self._parse_context_accs(raw_output)
        if context_feedback:
            feedback_parts.append(context_feedback)
        metrics.update(context_metrics)

        # Parse average accuracy
        avg_feedback, avg_metrics = self._parse_average_acc(raw_output)
        if avg_feedback:
            feedback_parts.append(avg_feedback)
        metrics.update(avg_metrics)

        # Parse TEST_METRICS lines (from our train.sh extraction)
        test_feedback, test_metrics = self._parse_test_metrics(raw_output)
        if test_feedback:
            feedback_parts.append(test_feedback)
        # Only update if not already set from native output
        for k, v in test_metrics.items():
            if k not in metrics:
                metrics[k] = v

        # Prefix all metric keys with cmd_label to avoid collisions across envs
        label = cmd_label.replace("-", "_")
        prefixed = {f"{k}_{label}": v for k, v in metrics.items()}

        if feedback_parts:
            feedback = "\n".join(feedback_parts)
        else:
            # Fallback: return last 30 lines of output
            lines = raw_output.strip().splitlines()
            feedback = "\n".join(lines[-30:])

        return ParseResult(feedback=feedback, metrics=prefixed)

    def _parse_context_accs(self, output: str) -> tuple[str, dict]:
        """Parse lines like '- Context 1: 0.9876'."""
        lines = []
        metrics = {}
        for line in output.splitlines():
            match = re.search(r"Context\s+(\d+):\s+([\d.]+)", line)
            if match:
                ctx = int(match.group(1))
                acc = float(match.group(2))
                lines.append(f"  Context {ctx}: {acc:.4f}")
                metrics[f"context_{ctx}_accuracy"] = acc

        feedback = ""
        if lines:
            feedback = "Per-context accuracy:\n" + "\n".join(lines)
        return feedback, metrics

    def _parse_average_acc(self, output: str) -> tuple[str, dict]:
        """Parse '=> average accuracy over all N contexts: X.XXXX'."""
        metrics = {}
        feedback = ""
        match = re.search(
            r"average accuracy over all \d+ contexts:\s+([\d.]+)", output
        )
        if match:
            avg = float(match.group(1))
            metrics["average_accuracy"] = avg
            feedback = f"Average accuracy: {avg:.4f}"
        return feedback, metrics

    def _parse_test_metrics(self, output: str) -> tuple[str, dict]:
        """Parse TEST_METRICS lines from train.sh."""
        metrics = {}
        lines = []
        for line in output.splitlines():
            if line.strip().startswith("TEST_METRICS"):
                lines.append(line.strip())
                # Parse key=value pairs
                for kv in re.findall(r"(\w+)=([\d.]+)", line):
                    key, val = kv
                    try:
                        metrics[key] = float(val)
                    except ValueError:
                        pass
        feedback = ""
        if lines:
            feedback = "Extracted metrics:\n" + "\n".join(lines[-5:])
        return feedback, metrics
