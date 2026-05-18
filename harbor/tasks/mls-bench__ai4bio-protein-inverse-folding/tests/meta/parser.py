"""Task-specific output parser for ai4bio-protein-inverse-folding.
Handles output from custom_invfold.py:
- Training feedback: TRAIN_METRICS epoch=N loss=val recovery=val ...
- Test feedback: TEST_METRICS recovery=value perplexity=value
Metrics are keyed by benchmark label, e.g. recovery_CATH4.2, perplexity_TS50.
"""

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mlsbench.agent.parsers import OutputParser, ParseResult


class Parser(OutputParser):
    """Parser for the ai4bio-protein-inverse-folding task."""

    def parse(self, cmd_label: str, raw_output: str) -> ParseResult:
        feedback_parts = []
        metrics: dict = {}

        train_feedback = self._parse_train_metrics(raw_output)
        if train_feedback:
            feedback_parts.append(train_feedback)

        eval_feedback, eval_metrics = self._parse_eval_metrics(raw_output, cmd_label)
        if eval_feedback:
            feedback_parts.append(eval_feedback)
        metrics.update(eval_metrics)

        feedback = "\n".join(feedback_parts) if feedback_parts else raw_output[-3000:]
        return ParseResult(feedback=feedback, metrics=metrics)

    def _parse_train_metrics(self, output: str) -> str:
        lines = [l.strip() for l in output.splitlines() if l.strip().startswith("TRAIN_METRICS ")]
        if not lines:
            return ""
        return "Training progress (last 5 reports):\n" + "\n".join(lines[-5:])

    def _parse_eval_metrics(self, output: str, cmd_label: str) -> tuple:
        metrics: dict = {}
        feedback_parts = []

        for line in output.splitlines():
            line = line.strip()
            if not line.startswith("TEST_METRICS "):
                continue
            parts = line[len("TEST_METRICS "):].strip()
            for match in re.finditer(r"(\w+)=([\d.eE+-]+)", parts):
                metric_name = match.group(1).strip()
                value = float(match.group(2))
                key = f"{metric_name}_{cmd_label}"
                metrics[key] = value
                feedback_parts.append(f"  {metric_name}: {value:.6f}")

        feedback = ""
        if feedback_parts:
            feedback = f"Test results ({cmd_label}):\n" + "\n".join(feedback_parts)

        return feedback, metrics
