"""Output parser for ml-subgroup-calibration-shift."""

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mlsbench.agent.parsers import OutputParser, ParseResult


class Parser(OutputParser):
    """Parse subgroup calibration metrics from stdout."""

    def parse(self, cmd_label: str, raw_output: str) -> ParseResult:
        feedback_parts = []
        metrics: dict = {}

        train_feedback = self._parse_train_metrics(raw_output)
        if train_feedback:
            feedback_parts.append(train_feedback)

        test_feedback, test_metrics = self._parse_test_metrics(raw_output, cmd_label)
        if test_feedback:
            feedback_parts.append(test_feedback)
        metrics.update(test_metrics)

        feedback = "\n".join(feedback_parts) if feedback_parts else raw_output[-3000:]
        return ParseResult(feedback=feedback, metrics=metrics)

    def _parse_train_metrics(self, output: str) -> str:
        lines = [line.strip() for line in output.splitlines() if line.strip().startswith("TRAIN_METRICS:")]
        if not lines:
            return ""
        return "Training progress:\n" + "\n".join(lines[-5:])

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
                feedback = "Final metrics (%s): %s" % (
                    cmd_label,
                    ", ".join(f"{k}={v:.6f}" for k, v in metrics.items()),
                )

        return feedback, metrics
