"""Editable ReAct clipping surface for ``ood-react``.

Return ``None`` for no clipping, or a finite ID-fit feature quantile in
``[0.5, 1.0]``. The logit reconstruction and energy formula are fixed.
"""


# ================================================================
# EDITABLE REGION - select the clipping quantile below
# ================================================================
def select_clip_quantile():
    return 0.90
# ================================================================
# END EDITABLE REGION
# ================================================================
