"""Task-specific output parser for cv-diffusion-efficiency.

Extracts per-model FID from generation output. CLIP score is intentionally
discarded because task scoring uses FID only.

Expected format:
    GENERATION_METRICS model=sd15 method=ddim_cfg++ cfg_guidance=0.6 NFE=50 seed=42 fid=25.1234 clip_score=0.3245
"""

import re
import sys
from pathlib import Path

# Allow importing from mlsbench package when run standalone
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mlsbench.agent.parsers import OutputParser, ParseResult


class Parser(OutputParser):
    """Parser for the cv-diffusion-efficiency task."""

    def parse(self, cmd_label: str, raw_output: str) -> ParseResult:
        feedback_parts = []
        metrics: dict = {}

        # Parse generation metrics
        gen_feedback, gen_metrics = self._parse_generation_metrics(raw_output)
        if gen_feedback:
            feedback_parts.append(gen_feedback)
        metrics.update(gen_metrics)

        if feedback_parts:
            feedback = "\n".join(feedback_parts)
        else:
            feedback = raw_output

        return ParseResult(feedback=feedback, metrics=metrics)

    def _parse_generation_metrics(self, output: str) -> tuple[str, dict]:
        """Extract GENERATION_METRICS lines and return FID feedback + metrics."""
        model_fid: dict[str, float] = {}
        gen_lines: list[str] = []

        for line in output.splitlines():
            if "GENERATION_METRICS" not in line:
                continue
            gen_lines.append(line.strip())

            model_match = re.search(r"model=(\w+)", line)
            fid_match = re.search(r"fid=([\d.\-]+)", line)

            model = model_match.group(1) if model_match else "unknown"

            if fid_match:
                model_fid[model] = float(fid_match.group(1))

        metrics: dict = {}
        feedback = ""

        if model_fid:
            # Per-model metrics
            for m, fid in model_fid.items():
                metrics[f"fid_{m}"] = fid

            # Average metrics
            avg_fid = sum(model_fid.values()) / len(model_fid)
            metrics["fid"] = avg_fid

            # Feedback
            cleaned_lines = [re.sub(r"\s*clip_score=[\d.\-]+", "", ln) for ln in gen_lines]
            feedback = "Generation results:\n" + "\n".join(cleaned_lines)
            for m in sorted(model_fid.keys()):
                feedback += f"\n  {m}: FID={model_fid[m]:.4f}"
            feedback += f"\nAverage FID: {avg_fid:.4f}"

        return feedback, metrics
