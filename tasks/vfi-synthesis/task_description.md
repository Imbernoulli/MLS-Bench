# Video Frame Interpolation: Middle-Frame Synthesis Strategy

## Research Question
Given two frames (frame0 at t=0 and frame2 at t=1), synthesize the MIDDLE frame at
t=0.5. The core design choice in video frame interpolation (VFI) is HOW the middle
frame is built. The naive answer — a linear BLEND `0.5*(frame0+frame2)` — ignores
motion entirely and GHOSTS (double edges) the moment anything moves. The VFI
literature instead COMPENSATES motion: estimate the optical flow to the middle time,
BACKWARD-WARP each frame, and combine the two motion-compensated candidates
(Super-SloMo, Jiang et al. CVPR 2018; SepConv, Niklaus et al. ICCV 2017; RIFE, Huang
et al. ECCV 2022). The strongest methods go further and LEARN a per-pixel visibility
mask + a refinement residual so they can resolve OCCLUSION/DISOCCLUSION — the regions
where a blind average of the two warps still ghosts. Design the synthesis strategy
that maximizes the interpolated-frame PSNR under a FIXED backbone and training budget.

VFI is DISTINCT from all restoration (super-resolution / deblur / dehaze / derain /
inpaint / colorize / deshadow / HDR): nothing is degraded — the middle frame simply
does not exist and must be SYNTHESIZED from the inter-frame motion. It is also DISTINCT
from optical-flow estimation (RAFT predicts a flow FIELD; VFI predicts an IMAGE):
flow is a means here, the deliverable is the frame itself.

## Background
On a static scene any strategy is trivial; the difficulty is MOTION. A blend has no
notion of motion, so its error grows with displacement. A flow-warp fixes the bulk
motion but, at DISOCCLUSION boundaries (where the fast-moving foreground uncovers
background), a fixed 0.5-average of the two warped candidates still ghosts because one
of the two frames has no correct content there. A learned refinement net predicts a
soft visibility mask that PICKS the visible frame at each occluded pixel plus a residual
correction, so it is best — and its margin over pure flow-warp WIDENS with motion (more
disocclusion). The known ordering (interpolation PSNR) is: `blend < flow_warp < learned`.

## Implementation Contract
Modify `get_synthesis_config` in `video-frame-interp/solution/synthesis.py`:

```python
def get_synthesis_config():
    return {"method": "learned"}   # 'blend' | 'flow_warp' | 'learned'
```

- `blend`     — linear blend `0.5*(frame0+frame2)`, no motion (weak; ghosts).
- `flow_warp` — learnable flow net + backward-warp both frames to t=0.5 + fixed-0.5
                average of the two motion-compensated candidates (mid).
- `learned`   — flow_warp PLUS a refinement U-Net that predicts a soft per-pixel
                visibility/blend mask and a residual (Super-SloMo; strong / SOTA).

A malformed / crashing / unknown return falls back to `learned`.

## Fixed Pipeline & Evaluation
- Backbone: compact 3-level residual encoder-decoder (base width 32), FROZEN design; the
  flow net and the refinement net share this capacity where used.
- Training: 800 iters, Adam (lr 1e-3), Charbonnier loss, batch 32, on REAL Vimeo-90K
  triplet-interpolation test-set tiles (TOFlow, Xue et al. IJCV 2019 — genuine decoded
  video frames, non-overlapping 64x64 tiles, no synthesis/warping in the data itself; see
  `vendor/data_scripts/video-frame-interp/prepare_data.py` for full provenance). Train
  and val splits are disjoint by SOURCE VIDEO ID within each motion tercile.
- Settings (score aggregates over TWO real inter-frame motion-magnitude terciles,
  measured via per-tile Farneback flow magnitude): `medium` / `large` displacement.
  (A third tercile, `small`, was measured and DROPPED — see below.)
- Metric (higher is better): `psnr_<setting>` — interpolation PSNR of the SYNTHESIZED
  middle frame vs the true (real, camera-captured) middle frame. `blend_psnr` (the
  motion-agnostic floor) and `psnr_gain = psnr - blend_psnr` are reported so a real
  interpolator must beat the blend.
- Deterministic; runs on one GPU in a few minutes per setting.
- Scoring is anchored per setting between the `flow_warp` (weak real baseline -> score 0)
  and `learned` (strong reference) methods; `blend` sits below the floor. The
  blend < flow_warp < learned partial-order holds, CROSS-SEED (42, 123), on both
  settings, with the learned>flow_warp margin widening with motion (medium ~0.2-0.35 dB,
  large ~1.5-1.7 dB both seeds).

### Dropped setting: `small` (honest, cross-seed-confirmed real-data finding)
On real Vimeo-90K small-motion tiles the blend floor alone is already 41.16 dB — near
this compact net's effective ceiling — so there is essentially no disocclusion for the
learned refinement head to resolve. On BOTH seeds, `flow_warp` (41.69 / 41.77 dB) beats
`learned` (41.32 / 41.47 dB), inverting the intended order; this reproduces cross-seed,
so it is a genuine near-ceiling-saturation effect, not noise (see
`vendor/video-frame-interp/anchors/README.md` for full numbers). Per project mandate
(never HP-sweep to force monotonicity), `small` is dropped rather than shipped with a
forced ordering; `vfi-synthesis` ships as a 2-setting task (medium, large).

Measured anchors (B0 8xH200, cross-seed 42/123 averaged, 800 iters) — see
leaderboard.csv. Per setting (blend / flow_warp -> learned):
  medium: 28.59 / 32.43 -> 32.64
  large : 22.16 / 23.70 -> 25.35
