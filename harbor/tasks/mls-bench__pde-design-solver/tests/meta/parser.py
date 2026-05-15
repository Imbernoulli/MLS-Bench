"""Task-specific output parser for pde-design-solver.
Handles output from Neural-Solver-Library exp_steady_design:
- Training feedback: TRAIN_METRICS epoch=E train_loss=L rel_err=R
- Test feedback: rho_d, c_d, relative l2 error press/velo
"""

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mlsbench.agent.parsers import OutputParser, ParseResult


class Parser(OutputParser):
    """Parser for the pde-design-solver task."""

    def parse(self, cmd_label: str, raw_output: str) -> ParseResult:
        feedback_parts = []
        metrics: dict = {}

        train_feedback = self._parse_train_metrics(raw_output)
        if train_feedback:
            feedback_parts.append(train_feedback)

        eval_feedback, eval_metrics = self._parse_eval_metrics(raw_output, cmd_label)
        if eval_feedback:
            feedback_parts.append(eval_feedback)
        metrics.update(eval_metrics)

        feedback = "\n".join(feedback_parts) if feedback_parts else raw_output
        return ParseResult(feedback=feedback, metrics=metrics)

    def _parse_train_metrics(self, output: str) -> str:
        lines = [l.strip() for l in output.splitlines() if l.strip().startswith("TRAIN_METRICS ")]
        if not lines:
            return ""
        return "Training metrics (last epochs):\n" + "\n".join(lines[-5:])

    def _parse_eval_metrics(self, output: str, cmd_label: str) -> tuple[str, dict]:
        metrics: dict = {}
        feedback_lines = []

        for line in output.splitlines():
            # rho_d (Spearman correlation of drag coefficient, higher is better)
            match = re.search(r"rho_d:\s*([\d.eE+-]+)", line)
            if match:
                val = float(match.group(1))
                metrics[f"rho_d_{cmd_label}"] = val
                feedback_lines.append(f"  rho_d (drag correlation): {val:.6f}")

            # c_d (relative drag coefficient error, lower is better)
            match = re.search(r"c_d:\s*([\d.eE+-]+)", line)
            if match:
                val = float(match.group(1))
                metrics[f"c_d_{cmd_label}"] = val
                feedback_lines.append(f"  c_d (drag error): {val:.6f}")

            # relative l2 error press
            match = re.search(r"relative l2 error press:\s*([\d.eE+-]+)", line)
            if match:
                val = float(match.group(1))
                metrics[f"l2_press_{cmd_label}"] = val
                feedback_lines.append(f"  L2 error press: {val:.6f}")

            # relative l2 error velo
            match = re.search(r"relative l2 error velo:\s*([\d.eE+-]+)", line)
            if match:
                val = float(match.group(1))
                metrics[f"l2_velo_{cmd_label}"] = val
                feedback_lines.append(f"  L2 error velo: {val:.6f}")

        feedback = ""
        if feedback_lines:
            feedback = f"Test results ({cmd_label}):\n" + "\n".join(feedback_lines)

        return feedback, metrics
