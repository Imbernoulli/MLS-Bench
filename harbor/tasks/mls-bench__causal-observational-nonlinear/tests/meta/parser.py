"""Task-specific output parser for causal-observational-nonlinear.

Parses lines of the form:
    CAUSAL_METRICS shd=X f1=X.XXXX precision=X.XXXX recall=X.XXXX

Metrics are keyed by evaluation scenario label, e.g.:
    shd_ER8-MLP, f1_ER8-MLP, precision_ER8-MLP, recall_ER8-MLP
"""

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mlsbench.agent.parsers import OutputParser, ParseResult


class Parser(OutputParser):
    """Parser for the causal-observational-nonlinear task."""

    _PATTERN = re.compile(
        r"CAUSAL_METRICS\s+"
        r"shd=(\d+)\s+"
        r"f1=([\d.eE+\-]+)\s+"
        r"precision=([\d.eE+\-]+)\s+"
        r"recall=([\d.eE+\-]+)"
    )

    def parse(self, cmd_label: str, raw_output: str) -> ParseResult:
        metrics: dict = {}
        feedback_parts = []

        for line in raw_output.splitlines():
            m = self._PATTERN.search(line)
            if m:
                shd       = int(m.group(1))
                f1        = float(m.group(2))
                precision = float(m.group(3))
                recall    = float(m.group(4))

                metrics[f"shd_{cmd_label}"]       = shd
                metrics[f"f1_{cmd_label}"]        = f1
                metrics[f"precision_{cmd_label}"] = precision
                metrics[f"recall_{cmd_label}"]    = recall

                feedback_parts.append(
                    f"Results ({cmd_label}):\n"
                    f"  SHD={shd}  F1={f1:.4f}  "
                    f"Precision={precision:.4f}  Recall={recall:.4f}"
                )

        feedback = "\n".join(feedback_parts) if feedback_parts else raw_output
        return ParseResult(feedback=feedback, metrics=metrics)
