"""Task-specific output parser for ts-short-term-forecast.
Handles combined train+eval output from TSLib short-term forecasting (M4):
- Training feedback: TRAIN_METRICS epoch=E train_loss=L vali_loss=V
- Test feedback: smape:{value}, mape:{value} (injected per-pattern by pre_edit)
Metrics are keyed by seasonal pattern label, e.g. smape_m4_monthly.
"""

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mlsbench.agent.parsers import OutputParser, ParseResult


class Parser(OutputParser):
    """Parser for the ts-short-term-forecast task."""

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
        feedback_lines = []

        for line in output.splitlines():
            # Per-pattern SMAPE (injected by pre_edit)
            smape_match = re.search(r"smape:([\d.eE+-]+)", line)
            if smape_match:
                smape = float(smape_match.group(1))
                metrics[f"smape_{cmd_label}"] = smape
                feedback_lines.append(f"SMAPE: {smape:.4f}")

            # Anchor to avoid matching the "mape" substring inside "smape:".
            mape_match = re.search(r"(?:^|[^a-zA-Z])mape:([\d.eE+-]+)", line)
            if mape_match:
                mape = float(mape_match.group(1))
                metrics[f"mape_{cmd_label}"] = mape
                feedback_lines.append(f"MAPE: {mape:.4f}")

        feedback = ""
        if feedback_lines:
            feedback = f"Test results ({cmd_label}):\n  " + ", ".join(feedback_lines)

        return feedback, metrics
