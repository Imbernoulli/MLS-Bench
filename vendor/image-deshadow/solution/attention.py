"""Agent-editable surface: CHANNEL ATTENTION in the mask-guided deshadower.

A FIXED mask-guided residual deshadower (same U-Net width/depth, loss, optimiser, data,
iters, seed, eval split) removes a cast shadow (SP+M-Net linear illumination model, Le et al.
ICCV 2019). You design ONLY whether a squeeze-and-excitation CHANNEL ATTENTION gate is
applied to the decoder features:

    def get_attention_config():
        # {'attention': True | False}
        return {'attention': True}

  * attention=False -> plain decoder features, every channel weighted equally. WEAK.
  * attention=True  -> a squeeze-excite (SE) channel gate re-weights feature channels so the
                       shadow / illumination-carrying channels are emphasised (RCAN-style),
                       sharpening the predicted relighting -> higher shadow-region PSNR.

The DEFAULT below returns attention=False. Enabling SE attention helps most on the deeper
HEAVY shadows (more illumination structure to disentangle). A malformed / crashing return
falls back to attention=False (weak).
"""
from __future__ import annotations


# ================================================================
# EDITABLE REGION — design the attention config below
# ================================================================
def get_attention_config():
    # Default: no channel attention.
    return {"attention": False}
# ================================================================
# END EDITABLE REGION
# ================================================================
