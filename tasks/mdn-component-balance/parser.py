"""Strict hash-bound terminal-proof parser for MDN full evaluations."""
from __future__ import annotations

import math
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mlsbench.agent.parsers import OutputParser, ParseResult


EXPECTED_SEED = 42
EXPECTED_STEPS = 4000
EXPECTED_BATCH_SIZE = 512
EXPECTED_TRAIN = 20000
EXPECTED_TEST = 20000
EXPECTED_TASK = "mdn-component-balance"
EXPECTED_SURFACE = "component_balance"
EXPECTED_TARGETS = {"spiral"}
EXPECTED_TRAIN_STEPS = [0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 3999]
DATA_SHA256 = {
    "inverse_sine": (
        "9bc8821db766ac288db6cb1d5235d8288c59da1c217d885c66221659781b1c41",
        "0650c8fa5d6b8ee7b2632def88901f1b281ebc2c49ae147207bf7087b3e82d39",
    ),
    "two_branch": (
        "e48a4d98064ab407bc36e8a345572a2004414f1e60645724330c9e514ffc2aa9",
        "7ebd1e2943273ad9a8ea989bfc07305dff6722a2bf7b718b2666bc99bf19815f",
    ),
    "spiral": (
        "5ce99ae78507fb6d06fa6a7cf7c90863665a21c8ab8ab9e72a0bf4276a7ab3cb",
        "e365e41b2db7566bd62ed36c668e93d10f71ea5780db198fb1b7cb7e2b4993f0",
    ),
    "rot_bimodal": (
        "7fcd661720ea9c6cdaa308315fbdc4d72ab7c0ac588f9fe1d9306eb3d3db86a1",
        "902b60a9b4bd22129e5142e52f656b03b0fd72b33e93f9ae19affef44b129897",
    ),
}
_NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_PROTOCOL = re.compile(
    r"MDN_PROTOCOL protocol=mdn_full_v3 task=([a-z0-9-]+) "
    r"surface=([a-z0-9_]+) target=([a-z0-9_]+) seed=(\d+) "
    r"steps=(\d+) batch_size=(\d+) n_train=(\d+) n_test=(\d+) "
    r"train_sha256=([0-9a-f]{64}) test_sha256=([0-9a-f]{64}) device=(cuda|cpu)"
)
_TRAIN = re.compile(rf"MDN_TRAIN step=(\d+) train_nll=({_NUMBER})")
_COMPLETE = re.compile(
    r"MDN_COMPLETE protocol=mdn_full_v3 task=([a-z0-9-]+) "
    r"surface=([a-z0-9_]+) target=([a-z0-9_]+) seed=(\d+) "
    r"steps=(\d+) final_step=(\d+) batch_size=(\d+) n_train=(\d+) n_test=(\d+) "
    r"train_sha256=([0-9a-f]{64}) test_sha256=([0-9a-f]{64})"
)
_METRIC = re.compile(
    rf"MDN_METRICS protocol=mdn_full_v3 task=([a-z0-9-]+) "
    rf"surface=([a-z0-9_]+) target=([a-z0-9_]+) seed=(\d+) "
    rf"steps=(\d+) train_sha256=([0-9a-f]{{64}}) test_sha256=([0-9a-f]{{64}}) "
    rf"nll=({_NUMBER}) params=(\d+) elapsed=({_NUMBER})"
)
_FAILURE = re.compile(
    r"(?mi)(?:^\s*Traceback\s+\(most recent call last\):"
    r"|^\s*(?:SURFACE_ERROR|TRAIN_ERROR|EVAL_FAILED|MDN_NONFINITE"
    r"|VERIFICATION_FAILED|TIMEOUT|OUT_OF_MEMORY|CANCELLED|NODE_FAIL)\b"
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
    """Accept exactly one complete full-budget, checksum-bound MDN run."""

    def parse(self, cmd_label: str, raw_output: str) -> ParseResult:
        lines = [line.strip() for line in raw_output.splitlines() if line.strip()]
        if not lines:
            return _empty("MDN evaluation log is empty.")
        if _FAILURE.search("\n".join(lines)):
            return _empty("MDN evaluation log contains a failure marker.")
        if cmd_label not in EXPECTED_TARGETS or cmd_label not in DATA_SHA256:
            return _empty("Unknown MDN evaluation target.")

        protocol_positions = [
            index for index, line in enumerate(lines) if "MDN_PROTOCOL" in line
        ]
        complete_positions = [
            index for index, line in enumerate(lines) if "MDN_COMPLETE" in line
        ]
        metric_positions = [
            index for index, line in enumerate(lines) if "MDN_METRICS" in line
        ]
        if len(protocol_positions) != 1:
            return _empty("Expected exactly one MDN_PROTOCOL record.")
        if complete_positions != [len(lines) - 2] or metric_positions != [len(lines) - 1]:
            return _empty(
                "Expected one MDN_COMPLETE and one MDN_METRICS as the final two records."
            )

        protocol = _PROTOCOL.fullmatch(lines[protocol_positions[0]])
        complete = _COMPLETE.fullmatch(lines[-2])
        metric = _METRIC.fullmatch(lines[-1])
        if protocol is None or complete is None or metric is None:
            return _empty("Malformed or unversioned MDN terminal proof.")

        train_records: list[tuple[int, float, int]] = []
        for index, line in enumerate(lines):
            if "MDN_TRAIN" not in line:
                continue
            match = _TRAIN.fullmatch(line)
            if match is None:
                return _empty("Malformed MDN_TRAIN record.")
            train_records.append((int(match.group(1)), float(match.group(2)), index))
        if [step for step, _loss, _index in train_records] != EXPECTED_TRAIN_STEPS:
            return _empty("MDN training trace does not prove the complete 4,000-step budget.")
        if not all(math.isfinite(loss) for _step, loss, _index in train_records):
            return _empty("MDN training losses must be finite.")
        if not (
            protocol_positions[0] < train_records[0][2]
            and train_records[-1][2] < complete_positions[0]
        ):
            return _empty("MDN protocol, training, completion, and metric order is invalid.")

        train_sha, test_sha = DATA_SHA256[cmd_label]
        expected_protocol = (
            EXPECTED_TASK,
            EXPECTED_SURFACE,
            cmd_label,
            str(EXPECTED_SEED),
            str(EXPECTED_STEPS),
            str(EXPECTED_BATCH_SIZE),
            str(EXPECTED_TRAIN),
            str(EXPECTED_TEST),
            train_sha,
            test_sha,
            "cuda",
        )
        expected_complete = (
            EXPECTED_TASK,
            EXPECTED_SURFACE,
            cmd_label,
            str(EXPECTED_SEED),
            str(EXPECTED_STEPS),
            str(EXPECTED_STEPS - 1),
            str(EXPECTED_BATCH_SIZE),
            str(EXPECTED_TRAIN),
            str(EXPECTED_TEST),
            train_sha,
            test_sha,
        )
        expected_metric_prefix = (
            EXPECTED_TASK,
            EXPECTED_SURFACE,
            cmd_label,
            str(EXPECTED_SEED),
            str(EXPECTED_STEPS),
            train_sha,
            test_sha,
        )
        if (
            protocol.groups() != expected_protocol
            or complete.groups() != expected_complete
            or metric.groups()[:7] != expected_metric_prefix
        ):
            return _empty(
                "MDN proof does not match the required target, seed, budget, counts, or data hashes."
            )

        nll = float(metric.group(8))
        params = int(metric.group(9))
        elapsed = float(metric.group(10))
        if not math.isfinite(nll) or not math.isfinite(elapsed):
            return _empty("MDN metrics must be finite.")
        if params <= 0 or elapsed <= 0:
            return _empty("MDN parameter count and elapsed time must be valid.")

        feedback = (
            f"Results ({cmd_label}):\n"
            f"  held-out mixture NLL:  {nll:.6f}   (nats, lower is better)\n"
            f"  params:                {params}"
        )
        return ParseResult(feedback=feedback, metrics={f"nll_{cmd_label}": nll})
