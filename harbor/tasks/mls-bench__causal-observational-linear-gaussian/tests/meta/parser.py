"""Task-specific parser for causal-observational-linear-gaussian."""
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mlsbench.agent.parsers import OutputParser, ParseResult


class Parser(OutputParser):
    """Parse CPDAG metrics emitted by bench/run_eval.py."""

    _PATTERN = re.compile(
        r"CAUSAL_METRICS\s+"
        r"shd=(\d+)\s+"
        r"adj_precision=([\d.eE+\-]+)\s+"
        r"adj_recall=([\d.eE+\-]+)\s+"
        r"arrow_precision=([\d.eE+\-]+)\s+"
        r"arrow_recall=([\d.eE+\-]+)"
    )

    def parse(self, cmd_label: str, raw_output: str) -> ParseResult:
        metrics = {}
        feedback_parts = []

        for line in raw_output.splitlines():
            match = self._PATTERN.search(line)
            if not match:
                continue

            shd = int(match.group(1))
            adj_precision = float(match.group(2))
            adj_recall = float(match.group(3))
            arrow_precision = float(match.group(4))
            arrow_recall = float(match.group(5))

            metrics[f"shd_{cmd_label}"] = shd
            metrics[f"adj_precision_{cmd_label}"] = adj_precision
            metrics[f"adj_recall_{cmd_label}"] = adj_recall
            metrics[f"arrow_precision_{cmd_label}"] = arrow_precision
            metrics[f"arrow_recall_{cmd_label}"] = arrow_recall

            feedback_parts.append(
                f"Results ({cmd_label}):\n"
                f"  SHD={shd}  "
                f"AdjP={adj_precision:.4f} AdjR={adj_recall:.4f}  "
                f"ArrowP={arrow_precision:.4f} ArrowR={arrow_recall:.4f}"
            )

        feedback = "\n".join(feedback_parts) if feedback_parts else raw_output
        return ParseResult(feedback=feedback, metrics=metrics)
