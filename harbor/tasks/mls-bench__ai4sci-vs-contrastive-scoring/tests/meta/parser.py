"""Output parser for vs-contrastive-scoring."""

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mlsbench.agent.parsers import OutputParser, ParseResult


class Parser(OutputParser):
    """Parse virtual screening training and evaluation output."""

    def parse(self, cmd_label: str, raw_output: str) -> ParseResult:
        feedback_parts = []
        metrics = {}

        if cmd_label == "train":
            # --- TRAIN_METRICS ---
            train_lines = [
                l.strip()
                for l in raw_output.splitlines()
                if l.strip().startswith("TRAIN_METRICS")
            ]
            if train_lines:
                feedback_parts.append(
                    f"Training progress:\n" + "\n".join(train_lines[-10:])
                )
            # Look for validation BEDROC
            for line in raw_output.splitlines():
                m = re.search(r"valid_bedroc\s*[=:]\s*([\d.]+)", line)
                if m:
                    val = float(m.group(1))
                    feedback_parts.append(f"valid_bedroc: {val:.4f}")

        else:
            # --- Evaluation metrics (DUD-E / LIT-PCBA / DEKOIS) ---
            # Parse "TEST_METRICS key=value" lines
            for line in raw_output.splitlines():
                line = line.strip()
                if line.startswith("TEST_METRICS"):
                    for match in re.finditer(r"(\w+)=([\d.eE+-]+)", line):
                        key, val = match.group(1), float(match.group(2))
                        metric_key = f"{key}_{cmd_label}"
                        metrics[metric_key] = val
                        feedback_parts.append(f"{metric_key}: {val:.6f}")

            # Parse printed summary lines like "auc mean 0.9435"
            for line in raw_output.splitlines():
                line = line.strip()
                # "auc mean 0.9435"
                m = re.match(r"(auc|bedroc)\s+mean\s+([\d.]+)", line)
                if m:
                    key = f"{m.group(1)}_mean_{cmd_label}"
                    val = float(m.group(2))
                    if key not in metrics:
                        metrics[key] = val
                        feedback_parts.append(f"{key}: {val:.6f}")
                # "ef 0.005 mean 55.19"
                m = re.match(r"ef\s+([\d.]+)\s+mean\s+([\d.]+)", line)
                if m:
                    pct = m.group(1).replace(".", "")
                    key = f"ef{pct}_mean_{cmd_label}"
                    val = float(m.group(2))
                    if key not in metrics:
                        metrics[key] = val
                        feedback_parts.append(f"{key}: {val:.4f}")

        if not feedback_parts:
            feedback_parts.append(raw_output[-3000:])

        return ParseResult(
            feedback="\n".join(feedback_parts),
            metrics=metrics,
        )
