"""Task-specific output parser for cv-meanflow-perceptual-loss.

Extracts FID from TEST_METRICS output line.

Expected format:
    TEST_METRICS: fid=12.34, best_fid=11.50
"""

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mlsbench.agent.parsers import OutputParser, ParseResult


class Parser(OutputParser):
    """Parser for the cv-meanflow-perceptual-loss task."""

    def parse(self, cmd_label: str, raw_output: str) -> ParseResult:
        feedback_parts = []
        metrics: dict = {}

        for line in raw_output.splitlines():
            if "TEST_METRICS:" not in line:
                continue

            # Match nan/inf too (mse-only training can diverge): \b avoids
            # matching the "fid=" inside "best_fid=". Without this a divergent
            # run prints fid=nan, the regex misses, and no metric is recorded.
            fid_match = re.search(r"\bfid=([\d.]+|nan|[-+]?inf)", line, re.IGNORECASE)
            best_match = re.search(r"best_fid=([\d.]+|nan|[-+]?inf)", line, re.IGNORECASE)

            if fid_match:
                fid = float(fid_match.group(1))
                metrics["fid"] = fid

                best_fid = float(best_match.group(1)) if best_match else fid
                metrics["best_fid"] = best_fid

                size = None
                for s in ("small", "medium", "large"):
                    if s in cmd_label:
                        size = s
                        break
                if size:
                    metrics[f"fid_{size}"] = fid
                    metrics[f"best_fid_{size}"] = best_fid

                feedback_parts.append(
                    f"FID: {fid:.2f}, Best FID: {best_fid:.2f}"
                )

        if feedback_parts:
            feedback = "Training results:\n" + "\n".join(feedback_parts)
        else:
            feedback = raw_output

        return ParseResult(feedback=feedback, metrics=metrics)
