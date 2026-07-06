"""Output parser for the cv-harmonization-* image-harmonization tasks.

The harness emits one metric line per run:
    HARMONY_METRICS surface=<S> setting=<L> fg_psnr=<..> fg_psnr_gain=<..> \
        comp_fg_psnr=<..> fg_mse=<..> fg_ssim=<..>

Leaderboard metrics: fg_psnr_{label} (PRIMARY, FOREGROUND-region PSNR of the HARMONIZED
output vs the real GT, measured ONLY inside the foreground mask, higher better), plus
fg_psnr_gain_{label} (fg_psnr - comp_fg_psnr, must be > 0), comp_fg_psnr_{label} (the
composite-INPUT identity / do-nothing floor), fg_mse_{label} and fg_ssim_{label}
(diagnostics). Because comp_fg_psnr is the foreground PSNR of copying the composite
through, the 'copy' identity scores fg_psnr==comp_fg_psnr (gain 0); a real harmonizer must
recolour the foreground to beat that floor.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mlsbench.agent.parsers import OutputParser, ParseResult

_PAT = re.compile(
    r"HARMONY_METRICS\s+surface=(\S+)\s+setting=(\S+)\s+"
    r"fg_psnr=([\d.eE+-]+)\s+fg_psnr_gain=([\d.eE+-]+)\s+comp_fg_psnr=([\d.eE+-]+)\s+"
    r"fg_mse=([\d.eE+-]+)\s+fg_ssim=([\d.eE+-]+)"
)


class Parser(OutputParser):
    """Parser for the image-harmonization tasks (foreground-region PSNR vs the real GT)."""

    def parse(self, cmd_label: str, raw_output: str) -> ParseResult:
        metrics: dict = {}
        feedback = ""

        for line in raw_output.splitlines():
            m = _PAT.search(line)
            if m:
                fg_psnr = float(m.group(3)); gain = float(m.group(4))
                comp = float(m.group(5)); mse = float(m.group(6))
                ssim = float(m.group(7))
                metrics[f"fg_psnr_{cmd_label}"] = fg_psnr
                metrics[f"fg_psnr_gain_{cmd_label}"] = gain
                metrics[f"comp_fg_psnr_{cmd_label}"] = comp
                metrics[f"fg_mse_{cmd_label}"] = mse
                metrics[f"fg_ssim_{cmd_label}"] = ssim
                feedback = (
                    f"Image harmonization results ({cmd_label}):\n"
                    f"  surface: {m.group(1)}  setting: {m.group(2)}\n"
                    f"  fg_psnr:      {fg_psnr:.4f} dB   (PRIMARY = foreground-region PSNR harmonized-vs-GT, higher is better)\n"
                    f"  fg_psnr_gain: {gain:.4f} dB   (fg_psnr - comp_fg_psnr; MUST be > 0 to beat the do-nothing floor)\n"
                    f"  comp_fg_psnr: {comp:.4f} dB   (composite INPUT vs GT in the foreground; identity floor, diagnostic)\n"
                    f"  fg_mse:       {mse:.6f}   (diagnostic)\n"
                    f"  fg_ssim:      {ssim:.4f}   (diagnostic)"
                )

        trace_keys = ("NETWORK_APPLIED", "NETWORK_FALLBACK", "NETWORK_FIXED",
                      "NETWORK_COPY", "DATA ", "train it=")
        trace = [ln.strip() for ln in raw_output.splitlines()
                 if any(k in ln for k in trace_keys)]
        if trace:
            tail = "\n".join(trace[-6:])
            feedback = (feedback + "\n" + tail) if feedback else tail

        if not feedback:
            feedback = raw_output[-3000:]

        return ParseResult(feedback=feedback, metrics=metrics)
