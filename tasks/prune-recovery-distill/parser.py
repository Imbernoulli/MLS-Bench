"""Strict parser shared by the full-scale CIFAR-10 pruning task family."""
from __future__ import annotations

import math
import re
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mlsbench.agent.parsers import OutputParser, ParseResult


PROTOCOL = "cifar10-resnet18-200ep-v1"
EXPECTED_TASK_ID = "prune-recovery-distill"
EXPECTED_SURFACE = "recovery_distill"
LABEL_SEEDS = {"cifar10": 42, "cifar10_seed1": 1}
EXPECTED_KEYS = {
    "protocol",
    "task",
    "surface",
    "setting",
    "acc",
    "sparsity",
    "dense_acc",
    "pruned_acc_prefinetune",
    "nparams",
    "flops",
    "dense_flops",
    "flops_budget",
    "chance",
    "train",
    "test",
    "recovery_epochs",
    "seed",
    "manifest_sha256",
    "checkpoint_sha256",
}
FAILURE_MARKERS = (
    "_FALLBACK",
    "[COMMAND FAILED",
    "[TIMEOUT]",
    "[BUDGET CHECK FAILED]",
    "Traceback (most recent call last):",
    "CUDA out of memory",
    "OutOfMemoryError",
    "Segmentation fault",
    "verification failed",
    "evaluation did not complete",
    "DENSE_TRAINED",
)
METRIC_LINE = re.compile(r"^PRUNE_METRICS(?:\s+[A-Za-z_][A-Za-z0-9_]*=\S+)+$")
NUMBER = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")


def _invalid(raw_output: str, reason: str) -> ParseResult:
    return ParseResult(
        feedback=f"Invalid full CIFAR-10 pruning result: {reason}\n{raw_output[-2000:]}",
        metrics={},
    )


class Parser(OutputParser):
    def parse(self, cmd_label: str, raw_output: str) -> ParseResult:
        expected_seed = LABEL_SEEDS.get(cmd_label)
        if expected_seed is None:
            return _invalid(raw_output, f"unexpected command label {cmd_label!r}")
        if any(marker.lower() in raw_output.lower() for marker in FAILURE_MARKERS):
            return _invalid(raw_output, "failure or fallback marker present")

        lines = [line.strip() for line in raw_output.splitlines() if line.strip()]
        indexes = [i for i, line in enumerate(lines) if line.startswith("PRUNE_METRICS")]
        if len(indexes) != 1 or indexes[0] != len(lines) - 1:
            return _invalid(raw_output, "expected one terminal PRUNE_METRICS record")
        line = lines[-1]
        if not METRIC_LINE.fullmatch(line):
            return _invalid(raw_output, "malformed terminal metric record")

        pairs: dict[str, str] = {}
        for token in line.split()[1:]:
            key, value = token.split("=", 1)
            if key in pairs:
                return _invalid(raw_output, f"duplicate field {key}")
            pairs[key] = value
        if set(pairs) != EXPECTED_KEYS:
            return _invalid(raw_output, f"expected fields {sorted(EXPECTED_KEYS)}")

        if pairs["task"] != EXPECTED_TASK_ID or pairs["surface"] != EXPECTED_SURFACE:
            return _invalid(raw_output, "wrong pruning task or surface")
        if pairs["protocol"] != PROTOCOL or pairs["setting"] != cmd_label:
            return _invalid(raw_output, "wrong protocol or setting")
        for key in ("manifest_sha256", "checkpoint_sha256"):
            if not re.fullmatch(r"[0-9a-f]{64}", pairs[key]):
                return _invalid(raw_output, f"invalid {key}")

        integer_expectations = {
            "train": 50_000,
            "test": 10_000,
            "recovery_epochs": 160,
            "seed": expected_seed,
        }
        for key, expected in integer_expectations.items():
            if not pairs[key].isdigit() or int(pairs[key]) != expected:
                return _invalid(raw_output, f"expected {key}={expected}")
        for key in ("nparams", "flops", "dense_flops"):
            if not pairs[key].isdigit() or int(pairs[key]) <= 0:
                return _invalid(raw_output, f"{key} must be a positive integer")

        values: dict[str, float] = {}
        for key in (
            "acc",
            "sparsity",
            "dense_acc",
            "pruned_acc_prefinetune",
            "chance",
            "flops_budget",
        ):
            if not NUMBER.fullmatch(pairs[key]):
                return _invalid(raw_output, f"{key} is not a decimal")
            value = float(pairs[key])
            if not math.isfinite(value):
                return _invalid(raw_output, f"{key} is non-finite")
            values[key] = value
        if any(not 0.0 <= values[key] <= 1.0 for key in ("acc", "dense_acc", "pruned_acc_prefinetune")):
            return _invalid(raw_output, "accuracy outside [0,1]")
        if not 0.0 <= values["sparsity"] < 1.0 or abs(values["chance"] - 0.1) > 1e-6:
            return _invalid(raw_output, "invalid sparsity or chance proof")
        if abs(values["flops_budget"] - 0.5) > 1e-6:
            return _invalid(raw_output, "wrong fixed FLOPs budget")
        flops = int(pairs["flops"])
        dense_flops = int(pairs["dense_flops"])
        if EXPECTED_SURFACE not in {"structured_criterion", "flops_budget"}:
            if abs(values["sparsity"] - 0.9) > 0.001:
                return _invalid(raw_output, "unstructured sparsity budget was not enforced")
            if flops != dense_flops:
                return _invalid(raw_output, "unstructured MAC context changed unexpectedly")
        else:
            if values["sparsity"] <= 0.0 or flops >= dense_flops:
                return _invalid(raw_output, "structured pruning did not reduce the model")
            if EXPECTED_SURFACE == "flops_budget" and flops > dense_flops * 0.55:
                return _invalid(raw_output, "structured result exceeds the MAC budget")

        data_records = [line for line in lines if line.startswith("DATA ")]
        expected_budget = 0.5 if EXPECTED_SURFACE in {
            "structured_criterion", "flops_budget"
        } else 0.9
        expected_data = (
            f"DATA task={EXPECTED_TASK_ID} surface={EXPECTED_SURFACE} cifar10 "
            f"train=50000 test=10000 classes=10 chance=0.100 "
            f"target_budget={expected_budget} manifest_sha256={pairs['manifest_sha256']}"
        )
        if data_records != [expected_data]:
            return _invalid(raw_output, "missing unique full-data proof")
        dense_records = [line for line in lines if line.startswith("DENSE_")]
        dense_pattern = re.compile(
            rf"DENSE_LOADED task={re.escape(EXPECTED_TASK_ID)} "
            rf"surface={re.escape(EXPECTED_SURFACE)} protocol={re.escape(PROTOCOL)} "
            rf"epochs=(\d+) dense_acc=({NUMBER.pattern[1:-1]}) "
            rf"checkpoint_sha256=({re.escape(pairs['checkpoint_sha256'])})"
        )
        dense_match = dense_pattern.fullmatch(dense_records[0]) if len(dense_records) == 1 else None
        if dense_match is None:
            return _invalid(raw_output, "missing unique dense-checkpoint proof")
        dense_proof_acc = float(dense_match.group(2))
        if (
            int(dense_match.group(1)) < 200
            or not math.isfinite(dense_proof_acc)
            or abs(dense_proof_acc - values["dense_acc"]) > 1e-6
        ):
            return _invalid(raw_output, "dense checkpoint proof mismatch")
        model_records = [line for line in lines if line.startswith("MODEL ")]
        model_pattern = re.compile(
            rf"MODEL task={re.escape(EXPECTED_TASK_ID)} "
            rf"surface={re.escape(EXPECTED_SURFACE)} resnet18_cifar "
            r"prunable_params=([1-9]\d*)"
        )
        if len(model_records) != 1 or not model_pattern.fullmatch(model_records[0]):
            return _invalid(raw_output, "missing unique model proof")
        device_records = [line for line in lines if line.startswith("DEVICE ")]
        if len(device_records) != 1 or not re.fullmatch(
            r"DEVICE cuda torch \S+", device_records[0]
        ):
            return _invalid(raw_output, "CUDA execution proof missing")
        proof_indexes = [
            lines.index(device_records[0]),
            lines.index(data_records[0]),
            lines.index(dense_records[0]),
            lines.index(model_records[0]),
            indexes[0],
        ]
        if proof_indexes != sorted(proof_indexes):
            return _invalid(raw_output, "pruning proof records are out of order")

        metrics = {
            f"acc_{cmd_label}": values["acc"],
            f"sparsity_{cmd_label}": values["sparsity"],
            f"dense_acc_{cmd_label}": values["dense_acc"],
            f"flops_{cmd_label}": float(flops),
        }
        return ParseResult(
            feedback=(
                f"Full CIFAR-10 {EXPECTED_TASK_ID}/{EXPECTED_SURFACE}: "
                f"acc={values['acc']:.4f}, "
                f"sparsity={values['sparsity']:.4f}, seed={expected_seed}"
            ),
            metrics=metrics,
        )
