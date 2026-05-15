"""Parser for security-backdoor-defense."""

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mlsbench.agent.parsers import OutputParser, ParseResult


class Parser(OutputParser):
    """Extract backdoor-defense metrics from TEST_METRICS lines."""

    _TEST_PATTERN = re.compile(
        r"TEST_METRICS\s+"
        r"clean_acc=([\d.eE+\-]+)\s+"
        r"asr=([\d.eE+\-]+)\s+"
        r"poison_recall=([\d.eE+\-]+)\s+"
        r"defense_score=([\d.eE+\-]+)"
    )

    def parse(self, cmd_label: str, raw_output: str) -> ParseResult:
        suffix = cmd_label.replace("-", "_")
        metrics = {}
        feedback_parts = []

        for line in raw_output.splitlines():
            line = line.strip()
            if line.startswith("TRAIN_METRICS"):
                feedback_parts.append(line)
                continue
            match = self._TEST_PATTERN.search(line)
            if not match:
                continue
            clean_acc, asr, poison_recall, defense_score = map(float, match.groups())
            metrics[f"clean_acc_{suffix}"] = clean_acc
            metrics[f"asr_{suffix}"] = asr
            metrics[f"poison_recall_{suffix}"] = poison_recall
            metrics[f"defense_score_{suffix}"] = defense_score
            feedback_parts.append(
                f"{cmd_label}: clean_acc={clean_acc:.4f}, asr={asr:.4f}, "
                f"poison_recall={poison_recall:.4f}, defense_score={defense_score:.4f}"
            )

        feedback = "\n".join(feedback_parts) if feedback_parts else raw_output[-3000:]
        return ParseResult(feedback=feedback, metrics=metrics)
