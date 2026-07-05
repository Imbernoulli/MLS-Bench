"""Agent-editable surface: CONTEXT / FEATURE EXTRACTION warped alongside RGB.

A FIXED harness trains the SAME interpolator on tiny two-layer (occluding) triplets and scores
interpolation PSNR over three inter-frame MOTION MAGNITUDES. You design ONLY the context
extractor whose features are warped to t=0.5 and fused into the refinement net (contextual
synthesis, Niklaus & Liu 2018); everything else is fixed.

    def get_context_config():
        return {'kind': 'pyramid'}

`kind` chooses the context extractor:
  'none'    -> no context features; the refine net sees only warped RGB. (weak floor)
  'shallow' -> a single-conv feature map is warped alongside RGB -> a little extra context for
               synthesis. (mid)
  'pyramid' -> a multi-scale (residual) feature map is warped and fused -> rich context for
               resolving disocclusion. Highest PSNR. (strong / SOTA)

Known ordering (PSNR): none < shallow < pyramid, WIDENING with motion. The DEFAULT below
returns 'none'. A malformed / crashing return falls back to 'pyramid'.
"""
from __future__ import annotations


# ================================================================
# EDITABLE REGION -- design the context surface below
# ================================================================
def get_context_config():
    # DEFAULT: the WEAK baseline -- improve this toward the SOTA choice.
    return {'kind': 'none'}
# ================================================================
# END EDITABLE REGION
# ================================================================
