"""Strict terminal-proof parser for full OpenML GP evaluations."""
from __future__ import annotations

import math
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mlsbench.agent.parsers import OutputParser, ParseResult


EXPECTED_SPLITS = {
    "concrete": (
        927,
        103,
        "c6cb20776e4eebaee665fea5d9a5a688db19c6699219dfcac2a258e7977f1773",
    ),
    "kin8nm": (
        7373,
        819,
        "7b0680527b8b8835c300c183a81fa92c6b7f64ce047db6b574d9f63aa02d9fd0",
    ),
    "elevators": (
        14939,
        1660,
        "2d852b51f9424cf235786e4402aa7a80cc0676cee68b0eafcfaf8e9c71e46c2c",
    ),
}
EXPECTED_BUDGET_KIND = "iterations"
EXPECTED_BUDGET = 200
EXPECTED_SEED = 42
EXPECTED_TASK = "gp-exact-lr"
EXPECTED_SURFACE = "exact_lr"
_NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_COMPLETE = re.compile(
    r"GP_COMPLETE protocol=openml_full_v2 task=([a-z0-9-]+) "
    r"surface=([a-z0-9_]+) dataset=([a-z0-9_]+) seed=(\d+) "
    r"device=(cuda|cpu) "
    r"split_sha256=([0-9a-f]{64}) n_train=(\d+) n_test=(\d+) "
    r"budget_kind=(iterations|epochs) budget=(\d+) batch_size=(\d+) updates=(\d+)"
)
_METRIC = re.compile(
    rf"GP_METRICS protocol=openml_full_v2 task=([a-z0-9-]+) "
    rf"surface=([a-z0-9_]+) dataset=([a-z0-9_]+) seed=(\d+) "
    rf"device=(cuda|cpu) "
    rf"split_sha256=([0-9a-f]{{64}}) n_train=(\d+) n_test=(\d+) "
    rf"budget_kind=(iterations|epochs) budget=(\d+) batch_size=(\d+) updates=(\d+) "
    rf"nll=({_NUMBER}) rmse=({_NUMBER}) elapsed=({_NUMBER})"
)
_FAILURE = re.compile(
    r"(?mi)(?:^\s*Traceback\s+\(most recent call last\):"
    r"|^\s*(?:SURFACE_ERROR|TRAIN_ERROR|EVAL_FAILED|VERIFICATION_FAILED"
    r"|TIMEOUT|OUT_OF_MEMORY|CANCELLED|NODE_FAIL)\b"
    r"|^\s*\[(?:COMMAND FAILED|STATUS:\s*FAILED|BUDGET CHECK FAILED"
    r"|TIMEOUT|OOM|KILLED|CANCELLED)\b"
    r"|^\s*\[ERROR\]"
    r"|^\s*(?:RuntimeError|ValueError|TypeError|SystemExit):"
    r"|^\s*Killed(?:\s|$)"
    r"|\bsegmentation fault\b"
    r"|\bcommand exited with (?:code|status)\s*[1-9]\d*\b"
    r"|\b(?:exit (?:code|status)|status)\s*[=:]?\s*[1-9]\d*\b"
    r"|\b(?:training|evaluation|verification)\s+(?:has\s+)?failed\b)"
)


def _empty(message: str) -> ParseResult:
    return ParseResult(feedback=message, metrics={})


class Parser(OutputParser):
    """Accept one hash-bound completion immediately followed by one metric."""

    def parse(self, cmd_label: str, raw_output: str) -> ParseResult:
        lines = [line.strip() for line in raw_output.splitlines() if line.strip()]
        if not lines:
            return _empty("GP evaluation log is empty.")
        if _FAILURE.search("\n".join(lines)):
            return _empty("GP evaluation log contains a failure marker.")

        complete_positions = [
            index for index, line in enumerate(lines) if "GP_COMPLETE" in line
        ]
        metric_positions = [
            index for index, line in enumerate(lines) if "GP_METRICS" in line
        ]
        if complete_positions != [len(lines) - 2] or metric_positions != [len(lines) - 1]:
            return _empty(
                "Expected one GP_COMPLETE and one GP_METRICS as the final two records."
            )

        complete = _COMPLETE.fullmatch(lines[-2])
        metric = _METRIC.fullmatch(lines[-1])
        if complete is None or metric is None or cmd_label not in EXPECTED_SPLITS:
            return _empty("Malformed or unversioned GP terminal proof.")

        expected_train, expected_test, expected_sha = EXPECTED_SPLITS[cmd_label]
        complete_fields = complete.groups()
        metric_fields = metric.groups()
        expected_prefix = (
            EXPECTED_TASK,
            EXPECTED_SURFACE,
            cmd_label,
            str(EXPECTED_SEED),
            "cuda",
            expected_sha,
            str(expected_train),
            str(expected_test),
            EXPECTED_BUDGET_KIND,
            str(EXPECTED_BUDGET),
            str(expected_train),
            str(EXPECTED_BUDGET),
        )
        if complete_fields != expected_prefix or metric_fields[:12] != expected_prefix:
            return _empty(
                "GP proof does not match the required split, seed, or training budget."
            )

        nll, rmse, elapsed = map(float, metric_fields[12:])
        if not all(math.isfinite(value) for value in (nll, rmse, elapsed)):
            return _empty("GP metrics must be finite.")
        if rmse < 0 or elapsed <= 0:
            return _empty("GP RMSE must be non-negative and elapsed time positive.")

        trace = [
            line
            for line in lines[:-2]
            if line.startswith(("INDUCING", "FEATURES", "TRAIN"))
        ]
        feedback = (
            f"Results ({cmd_label}):\n"
            f"  test NLL:   {nll:.6f}   (lower is better)\n"
            f"  test RMSE:  {rmse:.6f}   (lower is better)"
        )
        if trace:
            feedback += "\n" + "\n".join(trace[-4:])
        return ParseResult(
            feedback=feedback,
            metrics={f"nll_{cmd_label}": nll, f"rmse_{cmd_label}": rmse},
        )
