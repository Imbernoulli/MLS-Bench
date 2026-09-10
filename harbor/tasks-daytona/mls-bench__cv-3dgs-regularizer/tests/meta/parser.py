"""Task-specific output parser for cv-3dgs-densification.

Extracts PSNR, SSIM, LPIPS from TEST_METRICS output line.

Expected format:
    TEST_METRICS: psnr=29.648, ssim=0.9211, lpips=0.033, num_gs=2512579, best_psnr=29.648
"""

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mlsbench.agent.parsers import OutputParser, ParseResult


class Parser(OutputParser):
    """Parser for the cv-3dgs-densification task."""

    def parse(self, cmd_label: str, raw_output: str) -> ParseResult:
        feedback_parts = []
        metrics: dict = {}

        for line in raw_output.splitlines():
            if ("TRAIN_METRICS:" in line or "EVAL " in line or
                "Initialized" in line or "Loaded" in line or
                "Traceback" in line or "Error" in line or
                line.strip().startswith("File ")):
                feedback_parts.append(line.strip())

            if "TEST_METRICS:" not in line:
                continue

            pattern = (r"psnr=([\d.]+),\s*ssim=([\d.]+),\s*lpips=([\d.]+),"
                       r"\s*num_gs=(\d+),\s*best_psnr=([\d.]+)")
            m = re.search(pattern, line)
            if m:
                metrics["psnr"] = float(m.group(1))
                metrics["ssim"] = float(m.group(2))
                metrics["lpips"] = float(m.group(3))
                metrics["num_gs"] = int(m.group(4))
                metrics["best_psnr"] = float(m.group(5))

                # Per-scene suffixed key so leaderboard fills best_psnr_<scene>
                scene = None
                for s in ("garden", "bicycle", "bonsai", "stump"):
                    if s in cmd_label:
                        scene = s
                        break
                if scene:
                    metrics[f"best_psnr_{scene}"] = float(m.group(5))

                feedback_parts.append(
                    f"PSNR: {metrics['psnr']:.3f}, SSIM: {metrics['ssim']:.4f}, "
                    f"LPIPS: {metrics['lpips']:.3f}, #GS: {metrics['num_gs']}"
                )

        if feedback_parts:
            feedback = "Training results:\n" + "\n".join(feedback_parts[-20:])
        else:
            feedback = raw_output[-3000:]

        return ParseResult(feedback=feedback, metrics=metrics)
