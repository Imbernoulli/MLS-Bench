"""Output parser for opt-evolution-strategy.

Training feedback: TRAIN_METRICS gen=G best_fitness=F avg_fitness=A
Test metrics: TEST_METRICS best_fitness=F convergence_gen=G

Leaderboard metrics: best_fitness_<label>, convergence_gen_<label>
Lower best_fitness is better (minimization). Lower convergence_gen is better.
"""

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mlsbench.agent.parsers import OutputParser, ParseResult


class Parser(OutputParser):
    """Parser for evolutionary optimization benchmark output."""

    def parse(self, cmd_label: str, raw_output: str) -> ParseResult:
        feedback_parts = []
        metrics = {}

        # --- TRAIN_METRICS (feedback only) ---
        train_lines = [
            l.strip() for l in raw_output.splitlines()
            if l.strip().startswith("TRAIN_METRICS")
        ]
        if train_lines:
            feedback_parts.append(
                f"Training progress ({cmd_label}):\n" +
                "\n".join(train_lines[-5:])
            )

        # --- TEST_METRICS (feedback + leaderboard metrics) ---
        for line in raw_output.splitlines():
            line = line.strip()
            if line.startswith("TEST_METRICS"):
                # Parse best_fitness
                bf_match = re.search(r"best_fitness=([\d.eE+-]+)", line)
                if bf_match:
                    val = float(bf_match.group(1))
                    key = f"best_fitness_{cmd_label}"
                    metrics[key] = val
                    feedback_parts.append(f"{key}: {val:.6e}")

                # Parse convergence_gen
                cg_match = re.search(r"convergence_gen=(\d+)", line)
                if cg_match:
                    val = int(cg_match.group(1))
                    key = f"convergence_gen_{cmd_label}"
                    metrics[key] = val
                    feedback_parts.append(f"{key}: {val}")

        # --- Fallback ---
        if not feedback_parts:
            feedback_parts.append(raw_output[-3000:])

        return ParseResult(
            feedback="\n".join(feedback_parts),
            metrics=metrics,
        )
