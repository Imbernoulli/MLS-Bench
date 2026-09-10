"""Task-specific output parser for llm-rl-reward-normalization.

Handles combined train+eval output from verl PPO training with multiple
validation benchmarks (GSM8K, MATH-500, AMC).

Training feedback: lines matching
    TRAIN_METRICS step=N key=val key=val ...

Validation feedback: lines matching
    VAL_METRICS step=N val-core/<data_source>/acc/mean@1=X.XX ...

Metrics extracted:
    score_mean           (from critic/score/mean in TRAIN_METRICS)
    gsm8k_accuracy       (from val-core/openai/gsm8k/acc/mean@1)
    math500_accuracy     (from val-core/math500/acc/mean@1)
    amc_accuracy    (from val-core/amc23/acc/mean@N)
    gsm8k_reward_mean    (from val-core/openai/gsm8k/reward/mean@1)
"""

import re
import sys
from pathlib import Path

# Allow importing from mlsbench package when run standalone
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mlsbench.agent.parsers import OutputParser, ParseResult

# Regex to strip ANSI escape codes
_ANSI_RE = re.compile(r'\x1b\[[0-9;]*[a-zA-Z]|\x1b\(B')
# Regex to strip Ray worker prefixes like "(TaskRunner pid=1234)"
_RAY_PREFIX_RE = re.compile(r'^\([\w]+ pid=\d+\)\s*')

# (data_source, metric_name, display_name)
_BENCHMARKS = [
    ("openai/gsm8k", "gsm8k_accuracy", "GSM8K"),
    ("HuggingFaceH4/MATH-500", "math500_accuracy", "MATH-500"),
    ("amc23", "amc_accuracy", "AMC"),
]


def _clean_line(line: str) -> str:
    """Strip ANSI codes and Ray worker prefixes from a line."""
    cleaned = _ANSI_RE.sub('', line).strip()
    cleaned = _RAY_PREFIX_RE.sub('', cleaned)
    return cleaned


class Parser(OutputParser):
    """Parser for the llm-rl-reward-normalization task."""

    def parse(self, cmd_label: str, raw_output: str) -> ParseResult:
        feedback_parts = []
        metrics: dict = {}

        clean_lines = [_clean_line(line) for line in raw_output.splitlines()]

        # Parse training metrics
        train_feedback, train_metrics = self._parse_train_metrics(clean_lines)
        if train_feedback:
            feedback_parts.append(train_feedback)
        metrics.update(train_metrics)

        # Parse validation metrics
        val_feedback, val_metrics = self._parse_val_metrics(clean_lines)
        if val_feedback:
            feedback_parts.append(val_feedback)
        metrics.update(val_metrics)

        if feedback_parts:
            feedback = "\n\n".join(feedback_parts)
        else:
            feedback = raw_output

        return ParseResult(feedback=feedback, metrics=metrics)

    def _parse_train_metrics(self, clean_lines: list[str]) -> tuple[str, dict]:
        """Extract TRAIN_METRICS lines and return a summary + score_mean metric."""
        lines = []
        last_score_mean = None

        for line in clean_lines:
            if not line.startswith("TRAIN_METRICS "):
                continue
            lines.append(line)

            # Extract critic/score/mean
            match = re.search(r"critic/score/mean=(-?[\d.]+(?:e[+-]?\d+)?)", line)
            if match:
                try:
                    last_score_mean = float(match.group(1))
                except ValueError:
                    pass

        metrics: dict = {}
        if last_score_mean is not None:
            metrics["score_mean"] = last_score_mean

        if not lines:
            return "", metrics

        # Return last 5 training metric lines as feedback
        summary_lines = lines[-5:]
        feedback = "Training metrics (last steps):\n" + "\n".join(summary_lines)
        if last_score_mean is not None:
            feedback += f"\nFinal score mean: {last_score_mean:.4f}"
        return feedback, metrics

    def _parse_val_metrics(self, clean_lines: list[str]) -> tuple[str, dict]:
        """Extract VAL_METRICS lines and return feedback + metrics for all benchmarks."""
        val_lines: list[str] = []
        last_acc: dict[str, float] = {}
        last_reward_mean = None

        for line in clean_lines:
            if not line.startswith("VAL_METRICS "):
                continue
            val_lines.append(line)

            # Extract accuracy for each benchmark
            for data_source, metric_name, _ in _BENCHMARKS:
                pattern = re.escape(f"val-core/{data_source}/acc/mean@") + r"\d+=(-?[\d.]+(?:e[+-]?\d+)?)"
                match = re.search(pattern, line)
                if match:
                    try:
                        last_acc[metric_name] = float(match.group(1))
                    except ValueError:
                        pass

            # Extract GSM8K reward/mean (primary training signal)
            match = re.search(
                r"val-aux/openai/gsm8k/reward/mean@\d+=(-?[\d.]+(?:e[+-]?\d+)?)", line
            )
            if match:
                try:
                    last_reward_mean = float(match.group(1))
                except ValueError:
                    pass

        metrics: dict = {}
        feedback = ""

        if val_lines:
            feedback = "Validation metrics:\n" + "\n".join(val_lines[-3:])

        if last_reward_mean is not None:
            metrics["gsm8k_reward_mean"] = last_reward_mean
            feedback += f"\nFinal reward mean: {last_reward_mean:.4f}"

        if last_acc:
            summary_parts = []
            for _, metric_name, display_name in _BENCHMARKS:
                if metric_name in last_acc:
                    metrics[metric_name] = last_acc[metric_name]
                    summary_parts.append(f"{display_name}: {last_acc[metric_name]:.4f}")
            if summary_parts:
                feedback += f"\nFinal accuracy: {' | '.join(summary_parts)}"

        return feedback, metrics
