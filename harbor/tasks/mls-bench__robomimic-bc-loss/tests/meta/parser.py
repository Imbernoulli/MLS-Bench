"""Task-specific output parser for robomimic-bc-loss.

Parses robomimic's native training output:

Training feedback: lines matching
    TRAIN_METRICS epoch=E train_loss=L

Rollout output (robomimic format):
    Epoch N Rollouts took Xs (avg) with results:
    Env: env_name
    { "Success_Rate": 0.95, ... }

Final metric: line matching
    TEST_METRICS success_rate=X.XXXXXX

Leaderboard metric: success_rate (higher is better).
"""

import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mlsbench.agent.parsers import OutputParser, ParseResult


class Parser(OutputParser):
    """Parser for the robomimic-bc-loss task."""

    def parse(self, cmd_label: str, raw_output: str) -> ParseResult:
        feedback_parts = []
        metrics: dict = {}

        train_feedback = self._parse_train_metrics(raw_output)
        if train_feedback:
            feedback_parts.append(train_feedback)

        rollout_feedback, rollout_metrics = self._parse_rollout_metrics(raw_output, cmd_label)
        if rollout_feedback:
            feedback_parts.append(rollout_feedback)
        metrics.update(rollout_metrics)

        test_feedback, test_metrics = self._parse_test_metrics(raw_output, cmd_label)
        if test_feedback:
            feedback_parts.append(test_feedback)
        metrics.update(test_metrics)

        feedback = "\n".join(feedback_parts) if feedback_parts else raw_output[-3000:]
        return ParseResult(feedback=feedback, metrics=metrics)

    def _parse_train_metrics(self, output: str) -> str:
        lines = [
            l.strip()
            for l in output.splitlines()
            if l.strip().startswith("TRAIN_METRICS ")
        ]
        if not lines:
            return ""
        return "Training metrics (last 5 epochs):\n" + "\n".join(lines[-5:])

    def _parse_rollout_metrics(self, output: str, cmd_label: str = "") -> tuple[str, dict]:
        """Parse robomimic's native rollout JSON output for Success_Rate."""
        metrics: dict = {}
        feedback = ""
        best_sr = -1.0

        # Look for Success_Rate in JSON rollout logs
        for match in re.finditer(r'"Success_Rate"\s*:\s*([\d.]+)', output):
            sr = float(match.group(1))
            if sr > best_sr:
                best_sr = sr

        if best_sr >= 0:
            sr_key = f"success_rate_{cmd_label}" if cmd_label else "success_rate"
            metrics[sr_key] = best_sr
            feedback = f"Best rollout success rate: {best_sr:.2%}"

        return feedback, metrics

    def _parse_test_metrics(self, output: str, cmd_label: str = "") -> tuple[str, dict]:
        """Parse the TEST_METRICS line injected by pre_edit."""
        metrics: dict = {}
        feedback = ""

        for line in output.splitlines():
            match = re.search(
                r"TEST_METRICS\s+success_rate=([\d.]+)",
                line,
            )
            if match:
                success_rate = float(match.group(1))
                sr_key = f"success_rate_{cmd_label}" if cmd_label else "success_rate"
                metrics[sr_key] = success_rate
                feedback = f"Final success rate: {success_rate:.2%}"

        return feedback, metrics
