"""Task-specific output parser for opt-pac-bayes-bound.

Handles output from PAC-Bayes bound optimization training:
- Training feedback: TRAIN_METRICS prior_epoch=N loss=X accuracy=Y
                     TRAIN_METRICS posterior_epoch=N train_obj=X kl=Y
- Test feedback: TEST_METRICS risk_certificate=X
                 TEST_METRICS test_error=X
                 TEST_METRICS kl_divergence=X
                 TEST_METRICS ce_bound=X
                 TEST_METRICS empirical_01_risk=X

Primary metric: risk_certificate (lower = tighter bound = better).
"""

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mlsbench.agent.parsers import OutputParser, ParseResult


class Parser(OutputParser):
    """Parser for the opt-pac-bayes-bound task."""

    def parse(self, cmd_label: str, raw_output: str) -> ParseResult:
        feedback_parts = []
        metrics: dict = {}

        # Parse training metrics
        train_feedback = self._parse_train_metrics(raw_output)
        if train_feedback:
            feedback_parts.append(train_feedback)

        # Parse test metrics
        eval_feedback, eval_metrics = self._parse_test_metrics(raw_output, cmd_label)
        if eval_feedback:
            feedback_parts.append(eval_feedback)
        metrics.update(eval_metrics)

        feedback = "\n".join(feedback_parts) if feedback_parts else raw_output[-3000:]
        return ParseResult(feedback=feedback, metrics=metrics)

    def _parse_train_metrics(self, output: str) -> str:
        """Extract TRAIN_METRICS lines and return a summary."""
        lines = [l.strip() for l in output.splitlines()
                 if l.strip().startswith("TRAIN_METRICS ")]
        if not lines:
            return ""
        return "Training progress (last 5 steps):\n" + "\n".join(lines[-5:])

    def _parse_test_metrics(self, output: str, cmd_label: str) -> tuple[str, dict]:
        """Extract TEST_METRICS and return feedback + metrics dict."""
        metrics: dict = {}
        feedback_parts = []

        for line in output.splitlines():
            line = line.strip()
            if not line.startswith("TEST_METRICS "):
                continue
            for match in re.finditer(r"(\w+)=([\d.eE+-]+)", line):
                key, val = match.group(1), float(match.group(2))
                metric_key = f"{key}_{cmd_label}"
                metrics[metric_key] = val
                feedback_parts.append(f"  {key}: {val:.6f}")

        feedback = ""
        if feedback_parts:
            feedback = f"Results ({cmd_label}):\n" + "\n".join(feedback_parts)

        return feedback, metrics
