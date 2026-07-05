"""Output parser for the vfi-* video-frame-interpolation tasks.

The harness emits one metric line per run:
    VFI_METRICS surface=<S> setting=<L> psnr=<..> psnr_gain=<..> blend_psnr=<..> \
        ssim=<..> mse=<..>

Leaderboard metrics: psnr_{label} (PRIMARY, interpolation PSNR of the SYNTHESIZED middle
frame vs the true middle frame, higher better), plus psnr_gain_{label}
(psnr - blend_psnr, must be > 0 to beat the motion-agnostic blend), blend_psnr_{label}
(the naive-blend floor), ssim_{label} and mse_{label} (diagnostics).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mlsbench.agent.parsers import OutputParser, ParseResult

_PAT = re.compile(
    r"VFI_METRICS\s+surface=(\S+)\s+setting=(\S+)\s+"
    r"psnr=([\d.eE+-]+)\s+psnr_gain=([\d.eE+-]+)\s+blend_psnr=([\d.eE+-]+)\s+"
    r"ssim=([\d.eE+-]+)\s+mse=([\d.eE+-]+)"
)


class Parser(OutputParser):
    """Parser for the video-frame-interp tasks (interpolation PSNR vs true middle frame)."""

    def parse(self, cmd_label: str, raw_output: str) -> ParseResult:
        metrics: dict = {}
        feedback = ""

        for line in raw_output.splitlines():
            m = _PAT.search(line)
            if m:
                psnr = float(m.group(3)); gain = float(m.group(4))
                blend = float(m.group(5)); ssim = float(m.group(6))
                mse = float(m.group(7))
                metrics[f"psnr_{cmd_label}"] = psnr
                metrics[f"psnr_gain_{cmd_label}"] = gain
                metrics[f"blend_psnr_{cmd_label}"] = blend
                metrics[f"ssim_{cmd_label}"] = ssim
                metrics[f"mse_{cmd_label}"] = mse
                feedback = (
                    f"VFI results ({cmd_label}):\n"
                    f"  surface: {m.group(1)}  setting: {m.group(2)}\n"
                    f"  psnr:        {psnr:.4f} dB   (PRIMARY = interpolation PSNR synth-vs-true-middle, higher is better)\n"
                    f"  psnr_gain:   {gain:.4f} dB   (psnr - blend_psnr; MUST be > 0 to beat the naive blend)\n"
                    f"  blend_psnr:  {blend:.4f} dB   (naive 0.5*(f0+f2) vs GT; motion-agnostic floor, diagnostic)\n"
                    f"  ssim:        {ssim:.4f}   (diagnostic)\n"
                    f"  mse:         {mse:.6f}   (diagnostic)"
                )

        trace_keys = ("SYNTHESIS_APPLIED", "SYNTHESIS_FALLBACK", "SYNTHESIS_FIXED",
                      "MODEL ", "DATA ", "train it=")
        trace = [ln.strip() for ln in raw_output.splitlines()
                 if any(k in ln for k in trace_keys)]
        if trace:
            tail = "\n".join(trace[-6:])
            feedback = (feedback + "\n" + tail) if feedback else tail

        if not feedback:
            feedback = raw_output[-3000:]

        return ParseResult(feedback=feedback, metrics=metrics)
