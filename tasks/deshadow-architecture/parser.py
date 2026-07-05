"""Output parser for the deshadow-* image shadow-removal tasks.

The harness emits one metric line per run:
    DESHADOW_METRICS surface=<S> setting=<L> psnr=<..> psnr_gain=<..> shadow_psnr=<..> \
        ssim=<..> mse=<..> full_psnr=<..>

Leaderboard metrics: psnr_{label} (PRIMARY, SHADOW-REGION PSNR of the DESHADOWED output vs the
clean GT computed ONLY over pixels the shadow touches, higher better), plus psnr_gain_{label}
(psnr - shadow_psnr, must be > 0), shadow_psnr_{label} (the shadowed-INPUT copy / do-nothing
floor over the shadow region), ssim_{label}, mse_{label} and full_psnr_{label} (diagnostics).
Because psnr is measured IN the shadow region, a method that only copies the LIT region gains
nothing -- it must actually brighten the shadow. A net that copies its input scores
psnr==shadow_psnr (gain 0); a constant / all-white / all-black output scores far below it.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mlsbench.agent.parsers import OutputParser, ParseResult

_PAT = re.compile(
    r"DESHADOW_METRICS\s+surface=(\S+)\s+setting=(\S+)\s+"
    r"psnr=([\d.eE+-]+)\s+psnr_gain=([\d.eE+-]+)\s+shadow_psnr=([\d.eE+-]+)\s+"
    r"ssim=([\d.eE+-]+)\s+mse=([\d.eE+-]+)\s+full_psnr=([\d.eE+-]+)"
)


class Parser(OutputParser):
    """Parser for the image-deshadow tasks (SHADOW-REGION PSNR vs clean GT)."""

    def parse(self, cmd_label: str, raw_output: str) -> ParseResult:
        metrics: dict = {}
        feedback = ""

        for line in raw_output.splitlines():
            m = _PAT.search(line)
            if m:
                psnr = float(m.group(3)); gain = float(m.group(4))
                floor = float(m.group(5)); ssim = float(m.group(6))
                mse = float(m.group(7)); full = float(m.group(8))
                metrics[f"psnr_{cmd_label}"] = psnr
                metrics[f"psnr_gain_{cmd_label}"] = gain
                metrics[f"shadow_psnr_{cmd_label}"] = floor
                metrics[f"ssim_{cmd_label}"] = ssim
                metrics[f"mse_{cmd_label}"] = mse
                metrics[f"full_psnr_{cmd_label}"] = full
                feedback = (
                    f"Shadow-removal results ({cmd_label}):\n"
                    f"  surface: {m.group(1)}  setting: {m.group(2)}\n"
                    f"  psnr:        {psnr:.4f} dB   (PRIMARY = SHADOW-REGION deshadow PSNR, higher is better)\n"
                    f"  psnr_gain:   {gain:.4f} dB   (psnr - shadow_psnr; MUST be > 0 to beat the do-nothing floor)\n"
                    f"  shadow_psnr: {floor:.4f} dB   (shadowed INPUT over the shadow region; copy / identity floor)\n"
                    f"  full_psnr:   {full:.4f} dB   (whole-image PSNR, diagnostic)\n"
                    f"  ssim:        {ssim:.4f}   (diagnostic)\n"
                    f"  mse:         {mse:.6f}   (diagnostic)"
                )

        trace_keys = ("NETWORK_APPLIED", "NETWORK_FALLBACK", "NETWORK_FIXED",
                      "DATA ", "train it=")
        trace = [ln.strip() for ln in raw_output.splitlines()
                 if any(k in ln for k in trace_keys)]
        if trace:
            tail = "\n".join(trace[-6:])
            feedback = (feedback + "\n" + tail) if feedback else tail

        if not feedback:
            feedback = raw_output[-3000:]

        return ParseResult(feedback=feedback, metrics=metrics)
