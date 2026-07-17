"""Task-specific output parser for ts-exogenous-forecast.
Handles combined train+eval output from TSLib:
- Training feedback: TRAIN_METRICS epoch=E train_loss=L vali_loss=V test_loss=T
- Test feedback: mse:{value}, mae:{value}, dtw:{value}
Metrics keyed by dataset label, e.g. mse_ETTh1, mae_ETTh1.
"""

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mlsbench.agent.parsers import OutputParser, ParseResult


class Parser(OutputParser):
    """Parser for the ts-exogenous-forecast task."""

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

        feedback = "\n".join(feedback_parts) if feedback_parts else raw_output
        return ParseResult(feedback=feedback, metrics=metrics)

    def _parse_train_metrics(self, output: str) -> str:
        lines = [l.strip() for l in output.splitlines() if l.strip().startswith("TRAIN_METRICS ")]
        if not lines:
            return ""
        return "Training metrics (last epochs):\n" + "\n".join(lines[-5:])

    def _parse_eval_metrics(self, output: str, cmd_label: str) -> tuple[str, dict]:
        metrics: dict = {}
        feedback = ""

        for line in output.splitlines():
            match = re.search(r"mse:([\d.eE+-]+),\s*mae:([\d.eE+-]+)", line)
            if match:
                mse = float(match.group(1))
                mae = float(match.group(2))
                metrics[f"mse_{cmd_label}"] = mse
                metrics[f"mae_{cmd_label}"] = mae
                feedback = f"Test results ({cmd_label}):\n  MSE: {mse:.6f}, MAE: {mae:.6f}"

        return feedback, metrics
