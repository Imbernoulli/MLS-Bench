"""Task-specific output parser for nanoGPT KV-structural reduction tasks."""

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mlsbench.agent.parsers import OutputParser, ParseResult


class Parser(OutputParser):
    """Parser for KV-structural reduction nanoGPT tasks."""

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

        feedback = "\n".join(feedback_parts) if feedback_parts else raw_output
        return ParseResult(feedback=feedback, metrics=metrics)

    def _parse_train_metrics(self, output: str) -> str:
        lines = [l.strip() for l in output.splitlines() if l.strip().startswith("TRAIN_METRICS:")]
        if not lines:
            return ""
        return "Training metrics (last steps):\n" + "\n".join(lines[-5:])

    def _parse_test_metrics(self, output: str, cmd_label: str) -> tuple[str, dict]:
        metrics: dict = {}
        feedback = ""

        for line in output.splitlines():
            if "TEST_METRICS:" not in line:
                continue
            pairs = re.findall(r"(\w+)=([\d.]+(?:e[+-]?\d+)?|nan|inf|-inf)", line, re.IGNORECASE)
            for key, raw in pairs:
                metrics[f"{key}_{cmd_label}"] = float(raw.lower())
            if metrics:
                parts = [f"{k}={v:.4f}" for k, v in metrics.items()]
                feedback = f"Final metrics ({cmd_label}): " + ", ".join(parts)

        return feedback, metrics
