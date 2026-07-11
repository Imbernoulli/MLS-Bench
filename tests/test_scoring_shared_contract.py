from __future__ import annotations

import math
from fractions import Fraction

import pytest

from mlsbench.scoring.anchors import BaselineAnchors, _is_real_metric_value
from mlsbench.scoring.dsl import bl_best, col, const
from mlsbench.scoring.evaluate import _gmean, score_record_details
from mlsbench.scoring.primitives import (
    apply_direction_and_transform,
    bounded_power,
    logistic_score,
    penalty_upper,
    sigmoid_score,
    solve_gamma,
    solve_scale,
)
from mlsbench.scoring.spec import (
    AnchorRef,
    SettingSpec,
    TaskScoreSpec,
    TermSpec,
    validate_score_spec,
)


def _single_term_spec(term_spec: TermSpec, *, weight: object = 1.0) -> TaskScoreSpec:
    term_spec.name = "metric"
    return TaskScoreSpec(
        terms={"metric": term_spec},
        settings={
            "eval": SettingSpec(
                name="eval",
                terms=[("metric", weight)],
            ),
        },
    )


@pytest.mark.parametrize(
    ("term_spec", "raw", "expected"),
    [
        (
            col("metric").higher().bounded_power(
                bound=1.0, floor=const(0.0)
            ),
            0.75,
            0.75,
        ),
        (
            col("metric").lower().bounded_power(
                bound=0.0, floor=const(1.0)
            ),
            0.25,
            0.75,
        ),
    ],
)
def test_explicit_floor_bounded_power_is_baseline_free_linear(
    tmp_path,
    term_spec,
    raw,
    expected,
):
    score, settings, valid = score_record_details(
        _single_term_spec(term_spec),
        {"metric": raw},
        BaselineAnchors(tmp_path),
    )

    assert valid is True
    assert score == pytest.approx(expected)
    assert settings[0].terms[0].params == {
        "floor": pytest.approx(
            0.0 if term_spec.direction == "higher" else -1.0
        ),
        "bound": pytest.approx(0.0 if term_spec.direction == "lower" else 1.0),
        "gamma": 1.0,
        "ref": None,
        "r_ref": None,
    }


def test_explicit_unresolved_bounded_power_ref_never_uses_default_baseline(
    tmp_path,
):
    (tmp_path / "leaderboard.csv").write_text(
        "timestamp,model,is_final,seed,metric\n"
        "2026-01-01,baseline:weak,true,42,0.25\n"
        "2026-01-02,baseline:strong,true,42,0.75\n"
    )
    term_spec = col("metric").higher().bounded_power(
        bound=1.0,
        floor=const(0.0),
        ref=bl_best("missing_metric"),
    )

    score, settings, valid = score_record_details(
        _single_term_spec(term_spec),
        {"metric": 0.75},
        BaselineAnchors(tmp_path),
    )

    assert valid is False
    assert score == 0.0
    assert "unresolved_bounded_power_ref" in settings[0].invalid_reason


def test_floor_sigmoid_pending_anchor_scores_exact_zero(tmp_path):
    term_spec = col("metric").sigmoid(floor=const(1.0), scale=1.0)
    spec = _single_term_spec(term_spec)

    for raw in (0.0, 0.75, 1.0):
        score, settings, valid = score_record_details(
            spec, {"metric": raw}, BaselineAnchors(tmp_path)
        )
        assert valid is True
        assert score == 0.0
        assert settings[0].score == 0.0


def test_baseline_free_logistic_remains_smooth_and_ordered(tmp_path):
    spec = _single_term_spec(
        col("metric").sigmoid(ref=const(0.5), scale=0.1)
    )

    scores = [
        score_record_details(spec, {"metric": raw}, BaselineAnchors(tmp_path))[0]
        for raw in (0.2, 0.5, 0.8)
    ]

    assert 0.0 < scores[0] < scores[1] < scores[2] < 1.0
    assert scores[1] == pytest.approx(0.5)


@pytest.mark.parametrize(
    "anchor",
    [
        AnchorRef(kind="unknown", value=1.0),
        AnchorRef(kind="const", value=float("nan")),
        AnchorRef(kind="const", value=float("inf")),
        AnchorRef(kind="const", value=True),
        AnchorRef(kind="const", value="1.0"),
        AnchorRef(kind="const", metric="unexpected", value=1.0),
        AnchorRef(kind="bl_worst", metric=""),
        AnchorRef(kind="bl_worst", metric=123),
        AnchorRef(kind="bl_worst", metric=["metric"]),
        AnchorRef(kind="bl_best", metric="metric", value=1.0),
        True,
        "1.0",
        float("nan"),
        float("inf"),
    ],
)
def test_malformed_anchors_are_rejected_without_scoring_crashes(
    tmp_path,
    anchor,
):
    spec = _single_term_spec(
        TermSpec(
            name="metric",
            metric="metric",
            norm_type="sigmoid",
            floor=anchor,
            scale=1.0,
        )
    )

    errors = validate_score_spec(spec, ["metric"])
    score, settings, valid = score_record_details(
        spec, {"metric": 0.5}, BaselineAnchors(tmp_path)
    )

    assert any("floor anchor" in error for error in errors)
    assert valid is False
    assert score == 0.0
    assert settings[0].score == 0.0


def _constraint_with_target(target: object) -> TaskScoreSpec:
    objective = col("metric").sigmoid(ref=const(0.5), scale=0.1)
    objective.name = "metric"
    constraint = TermSpec(
        name="limit",
        metric="limit",
        role="constraint",
        norm_type="penalty_upper",
        constraint_target=target,
    )
    return TaskScoreSpec(
        terms={"metric": objective, "limit": constraint},
        settings={
            "eval": SettingSpec(
                name="eval",
                terms=[("metric", 1.0)],
                constraints=["limit"],
            ),
        },
    )


def _bounded_with_ref_score(ref_score: object) -> TaskScoreSpec:
    return _single_term_spec(
        TermSpec(
            name="metric",
            metric="metric",
            norm_type="bounded_power",
            floor=0.0,
            bound=1.0,
            ref=0.5,
            ref_score=ref_score,
        )
    )


@pytest.mark.parametrize(
    ("spec", "record", "error_fragment"),
    [
        (
            _single_term_spec(
                col("metric").sigmoid(ref=const(0.5), scale=0.1),
                weight="1.0",
            ),
            {"metric": 0.8},
            "weight",
        ),
        (
            _constraint_with_target("1.0"),
            {"metric": 0.8, "limit": 0.5},
            "target",
        ),
        (
            _bounded_with_ref_score("0.5"),
            {"metric": 0.8},
            "ref_score",
        ),
    ],
)
def test_numeric_strings_in_specs_fail_closed_without_type_errors(
    tmp_path,
    spec,
    record,
    error_fragment,
):
    errors = validate_score_spec(spec, list(record))
    score, settings, valid = score_record_details(
        spec, record, BaselineAnchors(tmp_path)
    )

    assert any(error_fragment in error for error in errors)
    assert valid is False
    assert score == 0.0
    assert settings[0].score == 0.0


@pytest.mark.parametrize("raw", ["0.8", True, float("nan"), float("inf")])
def test_non_real_or_nonfinite_metric_values_fail_closed(tmp_path, raw):
    spec = _single_term_spec(
        col("metric").sigmoid(ref=const(0.5), scale=0.1)
    )

    score, settings, valid = score_record_details(
        spec, {"metric": raw}, BaselineAnchors(tmp_path)
    )

    assert valid is False
    assert score == 0.0
    assert settings[0].score == 0.0


def test_dsl_preserves_malformed_bound_for_validation():
    bool_bound = _single_term_spec(
        col("metric").bounded_power(bound=True, floor=const(0.0))
    )
    malformed_const = _single_term_spec(
        col("metric").bounded_power(
            bound=AnchorRef(
                kind="const", metric="unexpected", value=1.0
            ),
            floor=const(0.0),
        )
    )

    assert any(
        "bound must be finite" in error
        for error in validate_score_spec(bool_bound, ["metric"])
    )
    assert any(
        "bound must be finite" in error
        for error in validate_score_spec(malformed_const, ["metric"])
    )


def test_baseline_anchors_ignore_invalid_values_without_hiding_later_finite_value(
    tmp_path,
):
    (tmp_path / "leaderboard.csv").write_text(
        "timestamp,model,is_final,seed,metric,other\n"
        "2026-01-01,baseline:inf,true,42,inf,0.1\n"
        "2026-01-02,baseline:nan,true,42,nan,0.2\n"
        "2026-01-03,baseline:bool,true,42,True,0.3\n"
        "2026-01-04,baseline:finite,true,42,0.5,0.4\n"
    )

    anchors = BaselineAnchors(tmp_path)

    assert anchors.metric_columns() == ["other", "metric"]
    assert anchors.get("metric") is not None
    assert anchors.get("metric").values == [0.5]
    assert anchors.worst("metric") == 0.5
    assert anchors.best("metric") == 0.5


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0, True),
        (0.5, True),
        (Fraction(1, 2), True),
        (True, False),
        ("0.5", False),
        (float("nan"), False),
        (float("inf"), False),
    ],
)
def test_anchor_metric_value_predicate_is_strict(value, expected):
    assert _is_real_metric_value(value) is expected


def test_legacy_baseline_calibration_preserves_representative_score(tmp_path):
    (tmp_path / "leaderboard.csv").write_text(
        "timestamp,model,is_final,seed,reward,accuracy\n"
        "2026-01-01,baseline:weak,true,42,1.0,0.2\n"
        "2026-01-02,baseline:middle,true,42,2.0,0.4\n"
        "2026-01-03,baseline:strong,true,42,3.0,0.6\n"
    )
    reward = col("reward").sigmoid()
    reward.name = "reward"
    accuracy = col("accuracy").bounded_power(bound=1.0)
    accuracy.name = "accuracy"
    spec = TaskScoreSpec(
        terms={"reward": reward, "accuracy": accuracy},
        settings={
            "eval": SettingSpec(
                name="eval",
                terms=[("reward", 1.0), ("accuracy", 1.0)],
            ),
        },
    )

    score, settings, valid = score_record_details(
        spec,
        {"reward": 3.0, "accuracy": 0.6},
        BaselineAnchors(tmp_path),
    )

    assert valid is True
    assert score == pytest.approx(0.5)
    assert [result.score for result in settings[0].terms] == pytest.approx(
        [0.5, 0.5]
    )


def test_primitives_reject_malformed_numeric_types_without_type_errors():
    assert bounded_power("0.5", 0.0, 1.0, 1.0) == 0.0
    assert logistic_score(0.5, "0.5", 0.1) == 0.0
    assert sigmoid_score(True, 0.0, 1.0) == 0.0
    assert penalty_upper(0.5, "1.0") == 0.0
    with pytest.raises(ValueError, match="must be finite"):
        solve_gamma(0.0, 1.0, 0.5, "0.5")
    with pytest.raises(ValueError, match="must be finite"):
        solve_scale(0.0, "1.0", 0.5)
    with pytest.raises(ValueError, match="must be finite"):
        apply_direction_and_transform("0.5", "higher", "id")


def test_true_geometric_mean_keeps_zero_and_rejects_numeric_strings():
    assert _gmean([0.0, 0.5]) == 0.0
    assert _gmean(["0.5", 1.0]) == 0.0
    assert _gmean([True, 1.0]) == 0.0
    assert _gmean([0.25, 1.0]) == pytest.approx(0.5)


def test_finite_real_anchors_remain_valid():
    valid_anchors = [
        0.0,
        Fraction(1, 2),
        AnchorRef(kind="const", value=1.0),
        AnchorRef(kind="bl_worst", metric="metric"),
        AnchorRef(kind="bl_best", metric="metric"),
    ]

    for anchor in valid_anchors:
        spec = _single_term_spec(
            TermSpec(
                name="metric",
                metric="metric",
                norm_type="sigmoid",
                floor=anchor,
                scale=1.0,
            )
        )
        assert validate_score_spec(spec, ["metric"]) == []


def test_math_reference_values_remain_unchanged():
    assert bounded_power(0.5, 0.0, 1.0, 1.0) == 0.5
    assert logistic_score(0.5, 0.5, 0.1) == 0.5
    assert sigmoid_score(1.0, 1.0, 1.0) == 0.0
    assert solve_gamma(0.0, 1.0, 0.5, 0.5) == pytest.approx(1.0)
    assert solve_scale(0.0, 1.0, 0.5) == pytest.approx(
        0.9102392266268373
    )
    assert math.isfinite(penalty_upper(2.0, 1.0))
