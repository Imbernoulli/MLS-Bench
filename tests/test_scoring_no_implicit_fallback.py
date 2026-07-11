from __future__ import annotations

from mlsbench.scoring.anchors import BaselineAnchors
from mlsbench.scoring.evaluate import _score_term
from mlsbench.scoring.spec import AnchorRef, TermSpec


def test_pathological_bounded_power_calibration_fails_instead_of_clamping(tmp_path):
    term = TermSpec(
        name="accuracy",
        metric="accuracy",
        norm_type="bounded_power",
        bound=1.0,
        ref=AnchorRef(kind="const", value=0.99),
        ref_score=0.5,
    )

    result = _score_term(
        term,
        raw_value=0.99,
        floor_raw=0.0,
        anchors=BaselineAnchors(tmp_path),
    )

    assert result.score == 0.0
    assert result.valid is False
    assert result.invalid_reason is not None
    assert "outside supported range" in result.invalid_reason
