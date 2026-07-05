"""Agent-editable surface: OCCLUSION HANDLING (how the two warped candidates are COMBINED).

A FIXED harness trains the SAME interpolator (learned flow + refinement) on tiny two-layer
(occluding) triplets and scores interpolation PSNR over three inter-frame MOTION MAGNITUDES.
You design ONLY how the two motion-compensated candidates (warped frame0 and warped frame2)
are merged into the middle frame; everything else is fixed.

    def get_occlusion_config():
        return {'kind': 'mask'}

`kind` chooses the combination rule:
  'avg'  -> fixed 0.5 average of the two warped frames. At a DISOCCLUSION boundary (one frame
            has no correct content) a blind average still ghosts. (weak floor)
  'time' -> a fixed temporal weight (still 0.5 at t=0.5, but the intended lever for
            time-aware weighting) -- no learned per-pixel visibility. (mid, ~avg here)
  'mask' -> a LEARNED soft per-pixel VISIBILITY mask (Super-SloMo) that PICKS the visible
            frame at each (dis)occluded pixel + a residual. Resolves occlusion -> highest
            PSNR, and its margin over 'avg' WIDENS with motion. (strong / SOTA)

Known ordering (PSNR): avg <= time < mask, WIDENING with motion. The DEFAULT below returns
'avg'. A malformed / crashing return falls back to 'mask'.
"""
from __future__ import annotations


# ================================================================
# EDITABLE REGION -- design the occlusion surface below
# ================================================================
def get_occlusion_config():
    # DEFAULT: the WEAK baseline -- improve this toward the SOTA choice.
    return {'kind': 'avg'}
# ================================================================
# END EDITABLE REGION
# ================================================================
