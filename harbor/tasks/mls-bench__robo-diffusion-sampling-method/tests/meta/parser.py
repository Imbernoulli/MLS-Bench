"""Task-specific output parser."""
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mlsbench.agent.parsers import OutputParser, ParseResult

class Parser(OutputParser):
    def parse(self, cmd_label: str, raw_output: str) -> ParseResult:
        feedback_parts = []
        metrics: dict = {}

        # Parse training metrics
        for line in raw_output.splitlines():
            if "TRAIN_METRICS" in line:
                feedback_parts.append(line.strip())

        # Parse eval metrics
        env_name = cmd_label.replace("train_", "")
        for line in raw_output.splitlines():
            if "EVAL_METRICS" in line:
                feedback_parts.append(line.strip())
                match = re.search(r"normalized_score=(-?[\d.]+(?:e[+-]?\d+)?)", line, re.IGNORECASE)
                if match:
                    metrics[f"{env_name}_normalized_score"] = float(match.group(1))
            elif "NFE_METRICS" in line:
                feedback_parts.append(line.strip())
                match = re.search(r"sampling_steps=(\d+)", line)
                if match:
                    metrics[f"{env_name}_sampling_steps"] = int(match.group(1))

        feedback = "\n".join(feedback_parts[-10:]) if feedback_parts else raw_output
        return ParseResult(feedback=feedback, metrics=metrics)
