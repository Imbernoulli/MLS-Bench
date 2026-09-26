"""Task-specific output parser for cv-diffusion-cfg.

Extracts per-model FID and CLIP score from generation output. FID is the
scored metric; CLIP score is recorded, shown for prompt-following, and used
by score_spec.py as a per-model floor (penalty_lower on clip_<model>).

Expected format:
    GENERATION_METRICS model=sd15 method=ddim_cfg++ cfg_guidance=7.5 NFE=50 nfe_used=50 seed=42 fid=25.1234 clip_score=0.3245
"""

import re
import sys
from pathlib import Path

# Allow importing from mlsbench package when run standalone
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mlsbench.agent.parsers import OutputParser, ParseResult


class Parser(OutputParser):
    """Parser for the cv-diffusion-cfg task."""

    def parse(self, cmd_label: str, raw_output: str) -> ParseResult:
        feedback_parts = []
        metrics: dict = {}

        # Parse generation metrics
        gen_feedback, gen_metrics = self._parse_generation_metrics(raw_output)
        if gen_feedback:
            feedback_parts.append(gen_feedback)
        metrics.update(gen_metrics)

        # batch_eval.py counts UNet evaluations and aborts before printing
        # GENERATION_METRICS when a sampler exceeds the NFE budget, so such a
        # run records no metric; say why instead of dumping the traceback.
        budget = re.search(r"(NFE_BUDGET_EXCEEDED|NFE accounting):[^\n]*", raw_output)
        if budget and not metrics:
            feedback_parts.append(f"[{cmd_label}] REJECTED — {budget.group(0).strip()}")

        if feedback_parts:
            feedback = "\n".join(feedback_parts)
        else:
            feedback = raw_output

        return ParseResult(feedback=feedback, metrics=metrics)

    def _parse_generation_metrics(self, output: str) -> tuple[str, dict]:
        """Extract GENERATION_METRICS lines and return FID/CLIP feedback + metrics."""
        model_fid: dict[str, float] = {}
        model_clip: dict[str, float] = {}
        gen_lines: list[str] = []

        for line in output.splitlines():
            if "GENERATION_METRICS" not in line:
                continue
            gen_lines.append(line.strip())

            model_match = re.search(r"model=(\w+)", line)
            fid_match = re.search(r"fid=([\d.\-]+)", line)
            clip_match = re.search(r"clip_score=([\d.\-]+)", line)

            model = model_match.group(1) if model_match else "unknown"

            if fid_match:
                model_fid[model] = float(fid_match.group(1))
            if clip_match:
                model_clip[model] = float(clip_match.group(1))

        metrics: dict = {}
        feedback = ""

        if model_fid:
            # Per-model metrics
            for m, fid in model_fid.items():
                metrics[f"fid_{m}"] = fid
            for m, cs in model_clip.items():
                metrics[f"clip_{m}"] = cs

            # Average metrics
            avg_fid = sum(model_fid.values()) / len(model_fid)
            metrics["fid"] = avg_fid

            # Feedback
            feedback = "Generation results:\n" + "\n".join(gen_lines)
            for m in sorted(model_fid.keys()):
                feedback += f"\n  {m}: FID={model_fid[m]:.4f}"
                if m in model_clip:
                    feedback += f"  CLIP={model_clip[m]:.4f}"
            feedback += f"\nAverage FID: {avg_fid:.4f}"

        return feedback, metrics
