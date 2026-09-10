"""Task-specific output parser for stf-traffic-forecast.
Handles combined train+eval output from BasicTS:
- Training feedback: TRAIN_METRICS epoch=E train_loss=L
- Test feedback: mae:{value},rmse:{value},mape:{value}
Metrics are keyed by dataset label, e.g. mae_METR-LA, rmse_METR-LA.
"""

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mlsbench.agent.parsers import OutputParser, ParseResult


class Parser(OutputParser):
    """Parser for the stf-traffic-forecast task."""

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
            match = re.search(r"mae:([\d.eE+-]+),\s*rmse:([\d.eE+-]+),\s*mape:([\d.eE+-]+)", line)
            if match:
                mae = float(match.group(1))
                rmse = float(match.group(2))
                mape = float(match.group(3))
                metrics[f"mae_{cmd_label}"] = mae
                metrics[f"rmse_{cmd_label}"] = rmse
                metrics[f"mape_{cmd_label}"] = mape
                feedback = (
                    f"Test results ({cmd_label}):\n"
                    f"  MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.4f}"
                )

        return feedback, metrics
