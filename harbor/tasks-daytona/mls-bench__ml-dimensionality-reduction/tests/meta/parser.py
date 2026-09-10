"""Task-specific output parser for ml-dimensionality-reduction.

Parses lines of the form:
    TRAIN_METRICS dataset=mnist elapsed=12.34s
    DIMRED_METRICS knn_acc=0.912345 trustworthiness=0.987654 continuity=0.976543 time=12.34

Metrics keyed by dataset label, e.g.:
    knn_acc_mnist, trustworthiness_mnist, continuity_mnist
"""

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mlsbench.agent.parsers import OutputParser, ParseResult


class Parser(OutputParser):
    """Parser for the ml-dimensionality-reduction task."""

    _METRIC_PATTERN = re.compile(
        r"DIMRED_METRICS\s+"
        r"knn_acc=([\d.eE+\-]+)\s+"
        r"trustworthiness=([\d.eE+\-]+)\s+"
        r"continuity=([\d.eE+\-]+)\s+"
        r"time=([\d.eE+\-]+)"
    )

    def parse(self, cmd_label: str, raw_output: str) -> ParseResult:
        metrics: dict = {}
        feedback_parts = []

        # Show training progress
        train_lines = [
            l.strip()
            for l in raw_output.splitlines()
            if l.strip().startswith("TRAIN_METRICS")
        ]
        if train_lines:
            feedback_parts.append(
                f"Reduction timing ({cmd_label}):\n" + "\n".join(train_lines)
            )

        # Final metrics
        for line in raw_output.splitlines():
            m = self._METRIC_PATTERN.search(line)
            if m:
                knn_acc = float(m.group(1))
                trust = float(m.group(2))
                cont = float(m.group(3))
                elapsed = float(m.group(4))

                metrics[f"knn_acc_{cmd_label}"] = knn_acc
                metrics[f"trustworthiness_{cmd_label}"] = trust
                metrics[f"continuity_{cmd_label}"] = cont

                feedback_parts.append(
                    f"Results ({cmd_label}):\n"
                    f"  kNN accuracy={knn_acc:.6f}\n"
                    f"  Trustworthiness={trust:.6f}\n"
                    f"  Continuity={cont:.6f}\n"
                    f"  Time={elapsed:.2f}s"
                )

        feedback = "\n".join(feedback_parts) if feedback_parts else raw_output[-3000:]
        return ParseResult(feedback=feedback, metrics=metrics)
