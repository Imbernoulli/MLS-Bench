"""Design surface: WHERE the reconstruction loss is applied.

The whole point of harmonization is to correct the appearance of the PASTED FOREGROUND;
the background is already correct. If the loss ignores the foreground, the net has no
signal to recolour it and degenerates to copying the composite through. Edit ONLY
get_loss_config() to return a dict {'mode': ...}:

    'bg'     -> supervise the (already-correct) BACKGROUND region ONLY. The net gets NO
                foreground signal, so it learns the trivial identity and the foreground stays
                mismatched (degenerate: scores ~ the do-nothing floor).
    'global' -> a whole-image L1 (background + foreground equally).
    'fg'     -> whole-image L1 PLUS a FOREGROUND emphasis term, so the region that actually
                needs correcting drives the optimisation -> best foreground PSNR.

Everything else is FIXED. A malformed / crashing return falls back to {'mode': 'fg'}.
"""


def get_loss_config():
    # Default: supervise the (already-correct) BACKGROUND only -> no foreground signal.
    return {"mode": "bg"}
