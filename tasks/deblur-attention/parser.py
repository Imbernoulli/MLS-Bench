"""Output parser for the deblur-* image-deblurring tasks.

The harness emits one metric line per run:
    DEBLUR_METRICS surface=<S> setting=<L> psnr=<..> psnr_gain=<..> blurry_psnr=<..> \
        ssim=<..> mse=<..>

Leaderboard metrics: psnr_{label} (PRIMARY, deblur PSNR of the RESTORED output vs the
sharp GT, higher better), plus psnr_gain_{label} (psnr - blurry_psnr, must be > 0),
blurry_psnr_{label} (the identity / do-nothing floor), ssim_{label} and mse_{label}
(diagnostics). Because blurry_psnr is the PSNR of passing the input straight through,
a net that copies its input scores psnr==blurry_psnr (gain 0) and a constant output
scores far below it -- a real deblur net must clearly beat the floor.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mlsbench.agent.parsers import OutputParser, ParseResult

_PAT = re.compile(
    r"DEBLUR_METRICS\s+surface=(\S+)\s+setting=(\S+)\s+"
    r"psnr=([\d.eE+-]+)\s+psnr_gain=([\d.eE+-]+)\s+blurry_psnr=([\d.eE+-]+)\s+"
    r"ssim=([\d.eE+-]+)\s+mse=([\d.eE+-]+)"
)


class Parser(OutputParser):
    """Parser for the image-deblur tasks (deblur PSNR vs sharp GT)."""

    def parse(self, cmd_label: str, raw_output: str) -> ParseResult:
        metrics: dict = {}
        feedback = ""

        for line in raw_output.splitlines():
            m = _PAT.search(line)
            if m:
                psnr = float(m.group(3)); gain = float(m.group(4))
                blurry = float(m.group(5)); ssim = float(m.group(6))
                mse = float(m.group(7))
                metrics[f"psnr_{cmd_label}"] = psnr
                metrics[f"psnr_gain_{cmd_label}"] = gain
                metrics[f"blurry_psnr_{cmd_label}"] = blurry
                metrics[f"ssim_{cmd_label}"] = ssim
                metrics[f"mse_{cmd_label}"] = mse
                feedback = (
                    f"Deblur results ({cmd_label}):\n"
                    f"  surface: {m.group(1)}  setting: {m.group(2)}\n"
                    f"  psnr:        {psnr:.4f} dB   (PRIMARY = deblur PSNR restored-vs-sharp, higher is better)\n"
                    f"  psnr_gain:   {gain:.4f} dB   (psnr - blurry_psnr; MUST be > 0 to beat the do-nothing floor)\n"
                    f"  blurry_psnr: {blurry:.4f} dB   (blurry INPUT vs sharp GT; identity floor, diagnostic)\n"
                    f"  ssim:        {ssim:.4f}   (diagnostic)\n"
                    f"  mse:         {mse:.6f}   (diagnostic)"
                )

        trace_keys = ("RESIDUAL_APPLIED", "RESIDUAL_FALLBACK", "RESIDUAL_FIXED",
                      "LOSS_APPLIED", "LOSS_FALLBACK", "LOSS_FIXED",
                      "SCALE_APPLIED", "SCALE_FALLBACK", "SCALE_FIXED",
                      "DATA ", "train it=")
        trace = [ln.strip() for ln in raw_output.splitlines()
                 if any(k in ln for k in trace_keys)]
        if trace:
            tail = "\n".join(trace[-6:])
            feedback = (feedback + "\n" + tail) if feedback else tail

        if not feedback:
            feedback = raw_output[-3000:]

        return ParseResult(feedback=feedback, metrics=metrics)
