"""Task-specific output parser for inverse-diffusion-algo.
Handles output from InverseBench main.py:
- Training feedback: TRAIN_METRICS sample=ID metric1=val metric2=val ...
- Test feedback: TEST_METRICS metric=value
Metrics are keyed by problem label, e.g. psnr_inv-scatter, psnr_blackhole.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mlsbench.agent.parsers import OutputParser, ParseResult


class Parser(OutputParser):
    """Parser for the inverse-diffusion-algo task."""

    # For inv-scatter: higher PSNR and SSIM are better
    # For blackhole: higher PSNR and blur_psnr are better; lower chi2 is better
    # We report all metrics keyed by problem label.

    def parse(self, cmd_label: str, raw_output: str) -> ParseResult:
        feedback_parts = []
        metrics: dict = {}

        # Preserve framework-emitted failure markers (`[COMMAND FAILED ...]`,
        # `[TIMEOUT ...]`) so they aren't silently dropped when partial
        # TRAIN_METRICS exist but no aggregate TEST_METRICS line was reached.
        for line in raw_output.splitlines()[:5]:
            if line.startswith("[COMMAND FAILED") or line.startswith("[TIMEOUT") or line.startswith("[exit code"):
                feedback_parts.append(line)
                break

        train_feedback = self._parse_train_metrics(raw_output)
        if train_feedback:
            feedback_parts.append(train_feedback)

        eval_feedback, eval_metrics = self._parse_eval_metrics(raw_output, cmd_label)
        if eval_feedback:
            feedback_parts.append(eval_feedback)
        else:
            # No aggregate TEST_METRICS line — run almost certainly didn't
            # finish (timeout / crash / killed). Surface this to the agent.
            feedback_parts.append(
                f"[NOTE] No aggregate `Test results ({cmd_label}):` was produced — "
                "the run likely did not complete (timeout or error). Per-sample "
                "TRAIN_METRICS above are partial; metrics will NOT be recorded."
            )
        metrics.update(eval_metrics)

        feedback = "\n".join(feedback_parts) if feedback_parts else raw_output
        return ParseResult(feedback=feedback, metrics=metrics)

    def _parse_train_metrics(self, output: str) -> str:
        lines = [l.strip() for l in output.splitlines() if l.strip().startswith("TRAIN_METRICS ")]
        if not lines:
            return ""
        return "Per-sample metrics (last 5 samples):\n" + "\n".join(lines[-5:])

    def _parse_eval_metrics(self, output: str, cmd_label: str) -> tuple[str, dict]:
        metrics: dict = {}
        feedback_parts = []

        for line in output.splitlines():
            line = line.strip()
            if not line.startswith("TEST_METRICS "):
                continue
            # Parse: TEST_METRICS metric_name=value
            parts = line[len("TEST_METRICS "):].strip()
            parsed = self.parse_metric_assignment(parts)
            if parsed is None:
                continue
            metric_name, value = parsed
            # Skip std metrics for leaderboard
            if metric_name.endswith("_std"):
                continue
            key = f"{metric_name}_{cmd_label}"
            metrics[key] = value
            feedback_parts.append(f"  {metric_name}: {value:.6f}")

        feedback = ""
        if feedback_parts:
            feedback = f"Test results ({cmd_label}):\n" + "\n".join(feedback_parts)

        return feedback, metrics
