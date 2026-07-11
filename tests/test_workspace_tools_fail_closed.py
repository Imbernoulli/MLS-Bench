from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from mlsbench.agent.parsers import OutputParser, ParseResult
from mlsbench.agent.tools import WorkspaceTools, _failure_marker
from mlsbench.scoring.evaluate import _validate_setting
from mlsbench.scoring.spec import SettingSpec, TermSpec


def _bare_tools() -> WorkspaceTools:
    return object.__new__(WorkspaceTools)


class _FakeLeaderboard:
    def __init__(self, records: list[dict] | None = None) -> None:
        self.records = list(records or [])
        self.added: list[dict] = []

    def all_records(self) -> list[dict]:
        return list(self.records)

    def add(self, record: dict) -> None:
        self.added.append(dict(record))


def test_seed_metrics_require_every_requested_seed() -> None:
    tools = _bare_tools()

    seeds, metrics = tools._filter_valid_seed_metrics(
        [42, 43],
        [{"acc": 0.9}, {}],
    )

    assert seeds == []
    assert metrics == []


def test_empty_or_metadata_only_parser_metrics_are_failures() -> None:
    tools = _bare_tools()

    assert tools._parser_metrics_error({}) == "no-real-metrics"
    assert tools._parser_metrics_error({"elapsed_eval": 1.0}) == "no-real-metrics"
    assert tools._parser_metrics_error({"metric": 0.0}) is None


@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_base_parser_rejects_nonfinite_metric_values(value: str) -> None:
    assert OutputParser.parse_metric_assignment(f"score={value}") is None


@pytest.mark.parametrize("parsed_metrics", [{}, {"elapsed_eval": 1.0}])
def test_rc_zero_with_no_real_parser_metrics_marks_command_failed(
    parsed_metrics: dict,
) -> None:
    tools = _bare_tools()
    tools.container_runtime = "local"
    tools._current_test_had_failures = False
    tools._run_budget_check = lambda *args, **kwargs: None
    tools._build_local_exec_spec = lambda *args, **kwargs: (
        ["true"],
        Path("."),
        {},
    )
    tools._run_local_command = lambda *args, **kwargs: (
        SimpleNamespace(returncode=0),
        "verification completed\n",
        False,
    )
    tools.parser = SimpleNamespace(
        parse=lambda label, output: ParseResult(
            feedback=f"parsed {label}",
            metrics=parsed_metrics,
        )
    )

    feedback, metrics, _elapsed = tools._run_single_cmd(
        {"label": "setting-b", "cmd": "verify.sh"},
        seed=42,
    )

    assert metrics == {}
    assert tools._current_test_had_failures is True
    assert "invalid-parser-metric=no-real-metrics" in feedback


@pytest.mark.parametrize(
    ("returncode", "raw_output", "expected_marker"),
    [
        (7, "acc=0.9\n", None),
        (
            0,
            "acc=0.9\nTraceback (most recent call last):\nRuntimeError: late\n",
            "Traceback (most recent call last):",
        ),
        (0, "training failed after metric emission\nacc=0.9\n", "training failed"),
    ],
)
def test_failed_command_never_keeps_parseable_metrics(
    returncode: int,
    raw_output: str,
    expected_marker: str | None,
) -> None:
    tools = _bare_tools()
    tools.container_runtime = "local"
    tools._current_test_had_failures = False
    tools._run_budget_check = lambda *args, **kwargs: None
    tools._build_local_exec_spec = lambda *args, **kwargs: (["true"], Path("."), {})
    tools._run_local_command = lambda *args, **kwargs: (
        SimpleNamespace(returncode=returncode),
        raw_output,
        False,
    )
    tools.parser = SimpleNamespace(
        parse=lambda label, output: ParseResult(
            feedback=f"parsed {label}",
            metrics={"acc": 0.9},
        )
    )

    feedback, metrics, _elapsed = tools._run_single_cmd(
        {"label": "eval", "cmd": "verify.sh"},
        seed=42,
    )

    assert metrics == {}
    assert tools._current_test_had_failures is True
    if returncode:
        assert f"FAILED exit={returncode}" in feedback
    if expected_marker:
        assert f"harness-marker={expected_marker}" in feedback


def test_timeout_never_parses_or_keeps_partial_metrics() -> None:
    tools = _bare_tools()
    tools.container_runtime = "local"
    tools._current_test_had_failures = False
    tools._run_budget_check = lambda *args, **kwargs: None
    tools._build_local_exec_spec = lambda *args, **kwargs: (["true"], Path("."), {})
    tools._run_local_command = lambda *args, **kwargs: (
        None,
        "score=0.9\n",
        True,
    )
    tools.parser = SimpleNamespace(
        parse=lambda *_args: pytest.fail("timeout output must not be parsed")
    )

    _feedback, metrics, _elapsed = tools._run_single_cmd(
        {"label": "eval", "cmd": "verify.sh"},
        seed=42,
    )

    assert metrics == {}
    assert tools._current_test_had_failures is True


def test_parser_exception_discards_success_exit_output() -> None:
    tools = _bare_tools()
    tools.container_runtime = "local"
    tools._current_test_had_failures = False
    tools._run_budget_check = lambda *args, **kwargs: None
    tools._build_local_exec_spec = lambda *args, **kwargs: (["true"], Path("."), {})
    tools._run_local_command = lambda *args, **kwargs: (
        SimpleNamespace(returncode=0),
        "score=0.9\n",
        False,
    )

    def fail_parse(*_args):
        raise ValueError("malformed proof")

    tools.parser = SimpleNamespace(parse=fail_parse)

    feedback, metrics, _elapsed = tools._run_single_cmd(
        {"label": "eval", "cmd": "verify.sh"},
        seed=42,
    )

    assert metrics == {}
    assert tools._current_test_had_failures is True
    assert "parser-error=ValueError" in feedback


def test_partial_setting_matrix_cannot_submit_successful_sibling_metrics() -> None:
    tools = _bare_tools()
    tools.container_runtime = "local"
    tools._current_test_had_failures = False
    tools._run_budget_check = lambda *args, **kwargs: None
    tools._build_local_exec_spec = lambda *args, **kwargs: (
        ["true"],
        Path("."),
        {},
    )
    tools._run_local_command = lambda *args, **kwargs: (
        SimpleNamespace(returncode=0),
        "verification completed\n",
        False,
    )
    tools.parser = SimpleNamespace(
        parse=lambda label, output: ParseResult(
            feedback=f"parsed {label}",
            metrics={"score_setting_a": 0.9} if label == "setting-a" else {},
        )
    )

    _feedback_a, metrics_a, _ = tools._run_single_cmd(
        {"label": "setting-a", "cmd": "verify-a.sh"},
        seed=42,
    )
    _feedback_b, metrics_b, _ = tools._run_single_cmd(
        {"label": "setting-b", "cmd": "verify-b.sh"},
        seed=42,
    )
    merged = {**metrics_a, **metrics_b}
    seeds, metrics, source = tools._resolve_submission_payload({
        "seeds": [42],
        "seed_metrics": [merged],
        "had_failures": tools._current_test_had_failures,
    })

    assert merged == {"score_setting_a": 0.9}
    assert tools._current_test_had_failures is True
    assert (seeds, metrics, source) == ([], [], "none")


def test_seed_metrics_reject_truncated_metric_list() -> None:
    tools = _bare_tools()

    seeds, metrics = tools._filter_valid_seed_metrics(
        [42, 43],
        [{"acc": 0.9}],
    )

    assert seeds == []
    assert metrics == []


def test_seed_metrics_reject_partial_key_union() -> None:
    tools = _bare_tools()

    seeds, metrics = tools._filter_valid_seed_metrics(
        [42, 43],
        [{"acc_visible": 0.9}, {"acc_hidden": 0.8}],
    )

    assert seeds == []
    assert metrics == []


def test_seed_metrics_reject_nonfinite_values() -> None:
    tools = _bare_tools()

    seeds, metrics = tools._filter_valid_seed_metrics(
        [42, 43],
        [{"acc": float("nan")}, {"acc": 0.8}],
    )

    assert seeds == []
    assert metrics == []


def test_seed_metrics_reject_invalid_values_before_metadata_filtering() -> None:
    tools = _bare_tools()

    for invalid_metrics in (
        {"acc": 0.9, "elapsed_eval": float("inf")},
        {"acc": 0.9, "acc_std": "not-a-number"},
        {"acc": 0.9, "ignored": None},
        {"acc": 0.9, "flag": True},
    ):
        seeds, metrics = tools._filter_valid_seed_metrics([42], [invalid_metrics])
        assert seeds == []
        assert metrics == []


def test_metric_aggregation_never_salvages_a_partially_invalid_matrix() -> None:
    tools = _bare_tools()

    assert tools._aggregate_metrics([
        {"acc": 0.8, "ignored_std": float("nan")},
        {"acc": 1.0, "ignored_std": 0.0},
    ]) == {}


def test_submission_does_not_fallback_to_old_leaderboard_result() -> None:
    tools = _bare_tools()
    tools.leaderboard = object()

    seeds, metrics, source = tools._resolve_submission_payload({
        "seeds": [42],
        "seed_metrics": [{}],
        "had_failures": False,
    })

    assert seeds == []
    assert metrics == []
    assert source == "none"


def test_submission_rejects_metrics_from_failed_test() -> None:
    tools = _bare_tools()

    seeds, metrics, source = tools._resolve_submission_payload({
        "seeds": [42],
        "seed_metrics": [{"acc": 0.9}],
        "had_failures": True,
    })

    assert seeds == []
    assert metrics == []
    assert source == "none"


def test_current_failed_run_overrides_historical_positive_finals() -> None:
    tools = _bare_tools()
    tools.leaderboard = _FakeLeaderboard([
        {
            "model": "agent",
            "is_final": "true",
            "seed": "42",
            "acc": 0.99,
        },
    ])
    tools.model_name = "agent"
    tools.extra_context = None
    tools.allow_web_search = False
    tools.seeds = [42]
    tools._current_run_final_seeds = set()

    tools.record_zero_if_no_finals()

    assert tools.leaderboard.added == [{
        "model": "agent",
        "is_final": True,
        "seed": "42",
    }]


def test_current_successful_final_is_not_replaced_by_empty_row() -> None:
    tools = _bare_tools()
    tools.leaderboard = _FakeLeaderboard()
    tools.model_name = "agent"
    tools.extra_context = None
    tools.allow_web_search = False
    tools.seeds = [42, 43]
    tools._current_run_final_seeds = {42}

    tools.record_zero_if_no_finals()

    assert tools.leaderboard.added == [{
        "model": "agent",
        "is_final": True,
        "seed": "43",
    }]


def test_harness_fallback_and_nonfinite_markers_are_failures() -> None:
    assert _failure_marker("AGENT_LOAD_FALLBACK broken\nacc=0.9\n") == "AGENT_LOAD_FALLBACK"
    assert _failure_marker("GAN2D_NONFINITE d_loss\nacc=0.9\n") == "GAN2D_NONFINITE"
    assert _failure_marker("SURFACE_ERROR: using random risk\nacc=0.9\n") == "SURFACE_ERROR"
    assert _failure_marker("TRAIN_ERROR -> reporting untrained model\nacc=0.9\n") == "TRAIN_ERROR"
    assert _failure_marker("EVAL_FAILED reason=RuntimeError\nacc=0.9\n") == "EVAL_FAILED"
    assert _failure_marker("PROMPT_CFG build_prompt failed (bad); using object-name fallback\n") == (
        "PROMPT_CFG build_prompt failed"
    )
    assert _failure_marker("TOKENSTRAT_CFG model_cfg_set_failed (bad)\n") == (
        "TOKENSTRAT_CFG model_cfg_set_failed"
    )
    assert _failure_marker("LAYER_CFG surgery_failed (bad); running full depth\n") == (
        "LAYER_CFG surgery_failed"
    )
    assert _failure_marker("BREAK_ON_ERROR: False\nacc=0.9\n") is None
    assert _failure_marker("RETURN_ON_ERROR: False\nacc=0.9\n") is None
    assert _failure_marker("ordinary output\nacc=0.9\n") is None


@pytest.mark.parametrize(
    "diagnostic",
    [
        "No training failure occurred",
        "training failure count: 0",
        "BREAK_ON_ERROR: False",
        "RETURN_ON_ERROR: False",
        "error_rate=0",
    ],
)
def test_failure_marker_does_not_match_diagnostic_text(diagnostic: str) -> None:
    assert _failure_marker(f"{diagnostic}\nacc=0.9\n") is None


def test_scoring_rejects_missing_or_nonfinite_constraint_metrics() -> None:
    setting = SettingSpec(name="eval", terms=[("accuracy", 1.0)], constraints=["latency"])
    terms = {
        "accuracy": TermSpec(name="accuracy", metric="acc"),
        "latency": TermSpec(
            name="latency",
            metric="latency_ms",
            role="constraint",
            norm_type="penalty_upper",
            constraint_target=10.0,
        ),
    }

    valid, reason = _validate_setting(setting, terms, {"acc": 0.9}, object())
    assert valid is False
    assert reason == "missing_constraint:latency_ms"

    valid, reason = _validate_setting(
        setting,
        terms,
        {"acc": 0.9, "latency_ms": float("inf")},
        object(),
    )
    assert valid is False
    assert reason == "missing_constraint:latency_ms"
