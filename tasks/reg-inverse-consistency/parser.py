"""Output parser for the reg-* deformable-registration tasks.

The harness emits ONE metric line per deformation-magnitude setting
(small / medium / large):

    REG_METRICS task=<T> setting=<mag> method=<M> psnr=<P> tre=<E> \
        folding=<F> ncc=<N>

psnr  = warped-moving vs fixed PSNR in dB (HIGHER better) -> column psnr_<mag>
tre   = mean landmark target-registration-error in px (LOWER better) -> tre_<mag>
folding = fraction of pixels with non-positive Jacobian det (LOWER) -> folding_<mag>
ncc   = warped-moving vs fixed normalized cross-correlation (HIGHER) -> ncc_<mag>

The primary scored metric is per-setting PSNR (the standard post-registration
image-similarity objective); the partial order identity < affine < voxelmorph is
preserved on PSNR across all three magnitudes. tre / folding / ncc are recorded
for diagnostics.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mlsbench.agent.parsers import OutputParser, ParseResult


class Parser(OutputParser):
    """Parser for the deformable-registration tasks (PSNR / TRE based)."""

    def parse(self, cmd_label: str, raw_output: str) -> ParseResult:
        metrics: dict = {}
        rows = []

        for line in raw_output.splitlines():
            m = re.search(
                r"REG_METRICS\s+task=(\S+)\s+setting=(\S+)\s+method=(\S+)\s+"
                r"psnr=([\d.eE+-]+)\s+tre=([\d.eE+-]+)\s+"
                r"folding=([\d.eE+-]+)\s+ncc=([\d.eE+-]+)",
                line,
            )
            if m:
                mag = m.group(2)
                psnr = float(m.group(4))
                tre = float(m.group(5))
                folding = float(m.group(6))
                ncc = float(m.group(7))
                metrics[f"psnr_{mag}"] = psnr
                metrics[f"tre_{mag}"] = tre
                metrics[f"folding_{mag}"] = folding
                metrics[f"ncc_{mag}"] = ncc
                rows.append(
                    f"  [{mag:6s}] method={m.group(3):11s} "
                    f"PSNR={psnr:6.3f} dB  TRE={tre:6.3f} px  "
                    f"folding={folding:.4f}  NCC={ncc:.4f}")

        if rows:
            feedback = ("Deformable-registration results "
                        f"({cmd_label}); per deformation-magnitude setting "
                        "(PSNR/NCC higher better, TRE/folding lower better):\n"
                        + "\n".join(rows))
        else:
            feedback = raw_output[-3000:]

        trace = [ln.strip() for ln in raw_output.splitlines()
                 if ln.strip().startswith(("METHOD_RESOLVED", "SOLUTION_FALLBACK"))]
        if trace:
            feedback = feedback + "\n" + "\n".join(trace[-3:])

        return ParseResult(feedback=feedback, metrics=metrics)
