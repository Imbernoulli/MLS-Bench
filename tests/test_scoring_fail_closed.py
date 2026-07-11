from __future__ import annotations

import math

import pytest

from mlsbench.scoring.anchors import BaselineAnchors
from mlsbench.scoring.evaluate import evaluate_task, score_record_details
from mlsbench.scoring.primitives import solve_gamma
from mlsbench.scoring.spec import (
    SettingSpec,
    TaskScoreSpec,
    TermSpec,
    validate_score_spec,
)


def _spec() -> TaskScoreSpec:
    return TaskScoreSpec(
        terms={
            "accuracy": TermSpec(
                name="accuracy",
                metric="accuracy",
                direction="higher",
                norm_type="sigmoid",
                ref=0.8,
                scale=0.1,
            ),
            "robustness": TermSpec(
                name="robustness",
                metric="robustness",
                direction="higher",
                norm_type="sigmoid",
                ref=0.7,
                scale=0.1,
            ),
            "latency": TermSpec(
                name="latency",
                metric="latency",
                role="constraint",
                direction="lower",
                norm_type="penalty_upper",
                constraint_target=10.0,
            ),
        },
        settings={
            "eval": SettingSpec(
                name="eval",
                terms=[("accuracy", 0.5), ("robustness", 0.5)],
                constraints=["latency"],
            ),
        },
    )


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), "not-a-number"])
@pytest.mark.parametrize("metric", ["accuracy", "robustness", "latency"])
def test_nonfinite_or_nonnumeric_score_inputs_force_zero(
    tmp_path,
    bad_value,
    metric,
):
    record = {"accuracy": 0.8, "robustness": 0.7, "latency": 5.0}
    record[metric] = bad_value

    score, settings, valid = score_record_details(
        _spec(),
        record,
        BaselineAnchors(tmp_path),
    )

    assert not valid
    assert score == 0.0
    assert not settings[0].valid


def test_partial_seeds_must_not_be_combined_into_apparent_success(tmp_path):
    spec = _spec()
    anchors = BaselineAnchors(tmp_path)
    seed_one = {"accuracy": 0.8, "latency": 5.0}
    seed_two = {"robustness": 0.7, "latency": 5.0}
    combined = {**seed_one, **seed_two}

    combined_score, _combined_settings, combined_valid = score_record_details(
        spec,
        combined,
        anchors,
    )
    seed_one_score, _seed_one_settings, seed_one_valid = score_record_details(
        spec,
        seed_one,
        anchors,
    )
    seed_two_score, _seed_two_settings, seed_two_valid = score_record_details(
        spec,
        seed_two,
        anchors,
    )

    assert combined_valid
    assert math.isfinite(combined_score) and combined_score > 0.0
    assert not seed_one_valid and seed_one_score == 0.0
    assert not seed_two_valid and seed_two_score == 0.0


@pytest.mark.parametrize(
    ("norm_type", "floor", "bound", "ref", "scale"),
    [
        ("bounded_power", 0.0, 1.0, 1.0, None),
        ("bounded_power", 1.0, 1.0, 0.5, None),
        ("sigmoid", 0.5, None, 0.5, None),
        ("sigmoid", 0.0, None, None, None),
        ("sigmoid", 0.0, None, 0.5, 0.0),
    ],
)
def test_invalid_calibration_forces_entire_record_to_zero(
    tmp_path,
    norm_type,
    floor,
    bound,
    ref,
    scale,
):
    term = TermSpec(
        name="metric",
        metric="metric",
        direction="higher",
        norm_type=norm_type,
        floor=floor,
        bound=bound,
        ref=ref,
        scale=scale,
    )
    spec = TaskScoreSpec(
        terms={"metric": term},
        settings={"eval": SettingSpec(name="eval", terms=[("metric", 1.0)])},
    )

    score, settings, valid = score_record_details(
        spec,
        {"metric": 0.75},
        BaselineAnchors(tmp_path),
    )

    assert not valid
    assert score == 0.0
    assert not settings[0].valid


def test_constraint_without_target_forces_entire_record_to_zero(tmp_path):
    spec = _spec()
    spec.terms["latency"].constraint_target = None

    score, settings, valid = score_record_details(
        spec,
        {"accuracy": 0.8, "robustness": 0.7, "latency": 5.0},
        BaselineAnchors(tmp_path),
    )

    assert not valid
    assert score == 0.0
    assert not settings[0].valid


@pytest.mark.parametrize("sharpness", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_constraint_sharpness_forces_zero(tmp_path, sharpness):
    spec = _spec()
    spec.terms["latency"].constraint_sharpness = sharpness

    errors = validate_score_spec(spec, ["accuracy", "robustness", "latency"])
    score, settings, valid = score_record_details(
        spec,
        {"accuracy": 0.8, "robustness": 0.7, "latency": 5.0},
        BaselineAnchors(tmp_path),
    )

    assert any("sharpness" in error for error in errors)
    assert not valid
    assert score == 0.0
    assert not settings[0].valid


@pytest.mark.parametrize("weight", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_objective_weight_forces_zero(tmp_path, weight):
    spec = _spec()
    spec.settings["eval"].terms[0] = ("accuracy", weight)

    errors = validate_score_spec(spec, ["accuracy", "robustness", "latency"])
    score, settings, valid = score_record_details(
        spec,
        {"accuracy": 0.8, "robustness": 0.7, "latency": 5.0},
        BaselineAnchors(tmp_path),
    )

    assert any("weight" in error for error in errors)
    assert not valid
    assert score == 0.0
    assert not settings[0].valid


def test_nonfinite_positive_weight_total_forces_zero(tmp_path):
    spec = _spec()
    spec.settings["eval"].terms = [
        ("accuracy", 1e308),
        ("robustness", 1e308),
    ]

    errors = validate_score_spec(spec, ["accuracy", "robustness", "latency"])
    score, settings, valid = score_record_details(
        spec,
        {"accuracy": 0.8, "robustness": 0.7, "latency": 5.0},
        BaselineAnchors(tmp_path),
    )

    assert any("finite positive total" in error for error in errors)
    assert not valid
    assert score == 0.0
    assert not settings[0].valid


def test_setting_without_objectives_forces_zero(tmp_path):
    spec = _spec()
    spec.settings["eval"].terms = []

    errors = validate_score_spec(spec, ["accuracy", "robustness", "latency"])
    score, settings, valid = score_record_details(
        spec,
        {"accuracy": 0.8, "robustness": 0.7, "latency": 5.0},
        BaselineAnchors(tmp_path),
    )

    assert any("at least one objective" in error for error in errors)
    assert not valid
    assert score == 0.0
    assert not settings[0].valid


@pytest.mark.parametrize(
    ("transform", "raw_value"),
    [("log", 0.0), ("log", -1.0), ("log1p", -1.0), ("log1p", -2.0)],
)
def test_invalid_transform_domain_forces_zero(tmp_path, transform, raw_value):
    term = TermSpec(
        name="metric",
        metric="metric",
        direction="higher",
        transform=transform,
        norm_type="sigmoid",
        ref=1.0,
        scale=0.1,
    )
    spec = TaskScoreSpec(
        terms={"metric": term},
        settings={"eval": SettingSpec(name="eval", terms=[("metric", 1.0)])},
    )

    score, settings, valid = score_record_details(
        spec,
        {"metric": raw_value},
        BaselineAnchors(tmp_path),
    )

    assert not valid
    assert score == 0.0
    assert settings[0].invalid_reason is not None
    assert "transform" in settings[0].invalid_reason


def test_failed_final_row_never_falls_back_to_old_nonfinal_score(tmp_path):
    task_dir = tmp_path / "authoritative-final"
    task_dir.mkdir()
    (task_dir / "score_spec.py").write_text(
        "from mlsbench.scoring.dsl import *\n"
        "term('metric', col('metric').sigmoid(ref=const(0.5), scale=0.1))\n"
        "setting('eval', weighted_mean(('metric', 1.0)))\n"
        "task(gmean('eval'))\n"
    )
    (task_dir / "leaderboard.csv").write_text(
        "timestamp,model,is_final,seed,metric\n"
        "2026-01-01T00:00:00,agent,false,mean,0.8\n"
        "2026-01-02T00:00:00,agent,true,mean,\n"
    )

    results = evaluate_task("authoritative-final", tasks_dir=tmp_path)

    assert len(results) == 1
    assert results[0].score == 0.0
    assert any("No metric values" in warning for warning in results[0].warnings)


def _write_logistic_task(task_dir, *, config: str | None = None, rows: str) -> None:
    task_dir.mkdir()
    (task_dir / "score_spec.py").write_text(
        "from mlsbench.scoring.dsl import *\n"
        "term('metric', col('metric').sigmoid(ref=const(0.5), scale=0.1))\n"
        "setting('eval', weighted_mean(('metric', 1.0)))\n"
        "task(gmean('eval'))\n"
    )
    (task_dir / "leaderboard.csv").write_text(
        "timestamp,model,is_final,seed,metric\n" + rows
    )
    if config is not None:
        (task_dir / "config.json").write_text(config)


def test_nonfinal_metrics_never_count_as_a_submission(tmp_path):
    _write_logistic_task(
        tmp_path / "missing-final",
        rows="2026-01-01T00:00:00,agent,false,mean,0.9\n",
    )

    result = evaluate_task("missing-final", tasks_dir=tmp_path)[0]

    assert result.score == 0.0
    assert any("No authoritative final" in warning for warning in result.warnings)


def test_latest_failed_multiseed_final_beats_older_positive_mean(tmp_path):
    _write_logistic_task(
        tmp_path / "latest-final-failed",
        config='{"seeds": [1, 2]}\n',
        rows=(
            "2026-01-01T00:00:01,agent,true,1,0.9\n"
            "2026-01-01T00:00:02,agent,true,2,0.9\n"
            "2026-01-01T00:00:03,agent,true,mean,0.9\n"
            "2026-01-02T00:00:00,agent,true,1,\n"
        ),
    )

    result = evaluate_task("latest-final-failed", tasks_dir=tmp_path)[0]

    assert result.score == 0.0
    assert any("Latest final multi-seed" in warning for warning in result.warnings)


def test_valid_multiseed_mean_is_authoritative(tmp_path):
    _write_logistic_task(
        tmp_path / "valid-multiseed",
        config='{"seeds": [1, 2]}\n',
        rows=(
            "2026-01-01T00:00:01,agent,true,1,0.9\n"
            "2026-01-01T00:00:02,agent,true,2,0.9\n"
            "2026-01-01T00:00:03,agent,true,mean,0.9\n"
        ),
    )

    result = evaluate_task("valid-multiseed", tasks_dir=tmp_path)[0]

    assert result.score > 0.0


def test_evaluator_forces_zero_on_score_spec_validation_error(tmp_path):
    task_dir = tmp_path / "invalid-score-spec"
    task_dir.mkdir()
    (task_dir / "score_spec.py").write_text(
        "from mlsbench.scoring.dsl import *\n"
        "term('metric', col('metric').sigmoid(ref=const(0.5), scale=0.1))\n"
        "setting('eval', weighted_mean(('metric', 0.0)))\n"
        "task(gmean('eval'))\n"
    )
    (task_dir / "leaderboard.csv").write_text(
        "timestamp,model,is_final,seed,metric\n"
        "2026-01-01T00:00:00,agent,true,42,0.9\n"
    )

    result = evaluate_task("invalid-score-spec", tasks_dir=tmp_path)[0]

    assert result.score == 0.0
    assert any("Invalid score specification" in warning for warning in result.warnings)


def test_invalid_seed_config_forces_zero(tmp_path):
    _write_logistic_task(
        tmp_path / "invalid-seeds",
        config='{"seeds": [1, 1]}\n',
        rows="2026-01-01T00:00:00,agent,true,mean,0.9\n",
    )

    result = evaluate_task("invalid-seeds", tasks_dir=tmp_path)[0]

    assert result.score == 0.0
    assert any("invalid seed matrix" in warning for warning in result.warnings)


@pytest.mark.parametrize("ref", [1e-12, 1.0 - 1e-12])
def test_out_of_range_bounded_power_calibration_raises(ref):
    with pytest.raises(ValueError, match="outside supported range"):
        solve_gamma(0.0, 1.0, ref, 0.5)
