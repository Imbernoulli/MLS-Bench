"""Task-specific output parser for quant-graph-stock.

Handles combined training + evaluation output from qlib stock prediction:

Signal metrics: lines matching
    SIGNAL_METRIC IC=X.XXXXXX
    SIGNAL_METRIC ICIR=X.XXXXXX
    SIGNAL_METRIC Rank_IC=X.XXXXXX
    SIGNAL_METRIC Rank_ICIR=X.XXXXXX

Portfolio metrics: lines matching
    PORTFOLIO_METRIC annualized_return=X.XXXXXX
    PORTFOLIO_METRIC max_drawdown=X.XXXXXX
    PORTFOLIO_METRIC information_ratio=X.XXXXXX

Metrics keyed for leaderboard: ic_csi300, icir_csi300, rank_ic_csi300,
    annualized_return_csi300, max_drawdown_csi300, information_ratio_csi300
"""

import re
import sys
from pathlib import Path

# Allow importing from mlsbench package when run standalone
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mlsbench.agent.parsers import OutputParser, ParseResult


class Parser(OutputParser):
    """Parser for the quant-graph-stock task."""

    def parse(self, cmd_label: str, raw_output: str) -> ParseResult:
        feedback_parts = []
        metrics: dict = {}

        # Parse signal metrics
        signal_feedback, signal_metrics = self._parse_signal_metrics(
            raw_output, cmd_label
        )
        if signal_feedback:
            feedback_parts.append(signal_feedback)
        metrics.update(signal_metrics)

        # Parse portfolio metrics
        port_feedback, port_metrics = self._parse_portfolio_metrics(
            raw_output, cmd_label
        )
        if port_feedback:
            feedback_parts.append(port_feedback)
        metrics.update(port_metrics)

        if feedback_parts:
            feedback = "\n".join(feedback_parts)
        else:
            feedback = raw_output

        return ParseResult(feedback=feedback, metrics=metrics)

    def _parse_signal_metrics(
        self, output: str, cmd_label: str
    ) -> tuple[str, dict]:
        """Extract SIGNAL_METRIC lines and return feedback + metrics."""
        metrics: dict = {}
        feedback_lines: list[str] = []

        for line in output.splitlines():
            match = re.match(
                r"SIGNAL_METRIC\s+(\w+)=(-?[\d.]+(?:e[+-]?\d+)?|nan|inf|-inf)",
                line.strip(),
                re.IGNORECASE,
            )
            if match:
                key = match.group(1).lower()
                raw = match.group(2).lower()
                value = float(raw)
                feedback_lines.append(line.strip())
                if not (value != value or abs(value) == float("inf")):
                    metric_key = f"{key}_{cmd_label}"
                    metrics[metric_key] = value

        feedback = ""
        if feedback_lines:
            feedback = (
                f"Signal analysis ({cmd_label}):\n"
                + "\n".join(feedback_lines)
            )

        return feedback, metrics

    def _parse_portfolio_metrics(
        self, output: str, cmd_label: str
    ) -> tuple[str, dict]:
        """Extract PORTFOLIO_METRIC lines and return feedback + metrics."""
        metrics: dict = {}
        feedback_lines: list[str] = []

        for line in output.splitlines():
            match = re.match(
                r"PORTFOLIO_METRIC\s+(\w+)=(-?[\d.]+(?:e[+-]?\d+)?|nan|inf|-inf)",
                line.strip(),
                re.IGNORECASE,
            )
            if match:
                key = match.group(1).lower()
                raw = match.group(2).lower()
                value = float(raw)
                feedback_lines.append(line.strip())
                if not (value != value or abs(value) == float("inf")):
                    metric_key = f"{key}_{cmd_label}"
                    metrics[metric_key] = value

        feedback = ""
        if feedback_lines:
            feedback = (
                f"Portfolio analysis ({cmd_label}):\n"
                + "\n".join(feedback_lines)
            )

        return feedback, metrics
