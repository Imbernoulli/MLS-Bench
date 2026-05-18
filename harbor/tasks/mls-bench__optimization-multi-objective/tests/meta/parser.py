"""Output parser for opt-multi-objective.

Handles output from the multi-objective optimization benchmark:
- Training feedback: TRAIN_METRICS gen=G hv=H igd=I spread=S front_size=N
- Test feedback: TEST_METRICS hv=H igd=I spread=S
Metrics are keyed by problem label, e.g. hv_zdt1, igd_dtlz2.
"""

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mlsbench.agent.parsers import OutputParser, ParseResult


class Parser(OutputParser):
    """Parser for the opt-multi-objective task."""

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
        lines = [l.strip() for l in output.splitlines() if l.strip().startswith("TRAIN_METRICS")]
        if not lines:
            return ""
        return "Training progress (last generations):\n" + "\n".join(lines[-5:])

    def _parse_eval_metrics(self, output: str, cmd_label: str) -> tuple[str, dict]:
        metrics: dict = {}
        feedback_parts = []

        for line in output.splitlines():
            line = line.strip()
            if line.startswith("TEST_METRICS"):
                for key in ("hv", "igd", "spread"):
                    match = re.search(rf"{key}=([\d.eE+-]+)", line)
                    if match:
                        val = float(match.group(1))
                        metrics[f"{key}_{cmd_label}"] = val
                        feedback_parts.append(f"  {key}: {val:.6f}")

        feedback = ""
        if feedback_parts:
            feedback = f"Test results ({cmd_label}):\n" + "\n".join(feedback_parts)

        return feedback, metrics
