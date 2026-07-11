"""Fail-closed parser for the complete normflows-density protocol proof."""
from __future__ import annotations

import math
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mlsbench.agent.parsers import OutputParser, ParseResult


PROTOCOL_VERSION = "flow-2d-community-20k-literal-ast-v3"
EXPECTED_STEPS = [*range(0, 20_000, 200), 19_999]
TASK_PROTOCOLS = {
    "flow-arch-family": ("architecture", {"checkerboard"}),
    "flow-autoregressive-coupling": ("conditioner", {"8gaussians"}),
    "flow-base-distribution": ("base_distribution", {"8gaussians"}),
    "flow-batch-size": ("batch_size", {"checkerboard"}),
    "flow-conditioner-width": ("conditioner_width", {"checkerboard"}),
    "flow-coupling-transform": (
        "coupling_transform", {"checkerboard", "moons", "8gaussians"}
    ),
    "flow-depth-permutation": ("depth", {"moons"}),
    "flow-learning-rate": ("learning_rate", {"moons"}),
    "flow-masking-pattern": ("masking_pattern", {"moons"}),
    "flow-spline-bins": ("spline_bins", {"checkerboard"}),
}
DATA_SHA256 = {
    "checkerboard": (
        "61a03c2d24a4a44eebbf61b7acd397b6df5834850889815aeaa4d4a3f1290ad4",
        "32137a42fad38a1363a3d0934ee3e46e6fbec3eca9ceca4d258b3b7e3d5b4f7f",
    ),
    "moons": (
        "48006975d1b5b065a9492f865fcfef808c2370312c6cf822cc7f0a94efa8b87e",
        "1e0bb490483a9992fc6f6c5321cdef6feb74c3e65a1e03a6099deac028024aac",
    ),
    "pinwheel": (
        "aec14ab3ed4d82f2976c369fdf53b51a1b600bbb9d32317702e0fc5670a88c84",
        "adeca11290413579bdf541647debfbfdfb82b725d2e98e0fdb267dab4a1606db",
    ),
    "8gaussians": (
        "54ec6fc49522ccea8526bd390037403034443fad7d484947ba3863538a51f332",
        "78f79e5b8e87cf865e38c46d303cf6ee99d4dddb7608d575e3abb57f514bf4ed",
    ),
}
FAILURE_MARKERS = (
    "Traceback (most recent call last)",
    "FLOW_FAILED",
    "FLOW_NONFINITE",
    "CUDA out of memory",
    "[COMMAND FAILED",
    "[STATUS: FAILED",
    "[BUDGET CHECK FAILED]",
)

_NUMBER = r"[-+]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][-+]?[0-9]+)?"
_HEX64 = r"[0-9a-f]{64}"
_PROTOCOL_RE = re.compile(
    rf"^FLOW_PROTOCOL version={PROTOCOL_VERSION} surface=(\S+) choice=(\S+) "
    rf"target=(\S+) device=(\S+) device_count=(\d+) seed=(\d+) "
    rf"steps=(\d+) batch_size=(\d+) lr=({_NUMBER}) "
    rf"optimizer=(\S+) objective=(\S+)$"
)
_DESIGN_RE = re.compile(
    r"^FLOW_DESIGN target=(\S+) n_transforms=(\d+) n_permutations=(\d+) "
    r"total_layers=(\d+) params=([1-9][0-9]*)$"
)
_DATA_RE = re.compile(
    rf"^FLOW_DATA target=(\S+) seed=(\d+) n_train=(\d+) n_test=(\d+) "
    rf"train_sha256=({_HEX64}) test_sha256=({_HEX64})$"
)
_TRAIN_RE = re.compile(rf"^FLOW_TRAIN step=(\d+) train_nll=({_NUMBER})$")
_METRIC_RE = re.compile(
    rf"^FLOW_METRICS nll=({_NUMBER}) bpd=({_NUMBER}) "
    rf"params=([1-9][0-9]*) elapsed=({_NUMBER})$"
)
_COMPLETE_RE = re.compile(
    rf"^FLOW_SETTING_COMPLETE version={PROTOCOL_VERSION} surface=(\S+) "
    r"choice=(\S+) target=(\S+) seed=(\d+) optimizer_steps=(\d+) "
    r"samples_seen=(\d+) n_train=(\d+) n_test=(\d+) params=([1-9][0-9]*)$"
)
_RECORDS = {
    "protocol": _PROTOCOL_RE,
    "design": _DESIGN_RE,
    "data": _DATA_RE,
    "train": _TRAIN_RE,
    "metric": _METRIC_RE,
    "complete": _COMPLETE_RE,
}


def _reject(reason: str) -> ParseResult:
    return ParseResult(feedback=f"Flow verification rejected: {reason}", metrics={})


def _task_id() -> str:
    task_file = Path(__file__).resolve().parent / "task_id"
    if task_file.is_file():
        try:
            return task_file.read_text(encoding="utf-8").strip()
        except OSError:
            return ""
    return Path(__file__).resolve().parent.name


def _choice_is_valid(surface: str, choice: str, batch_size: int, lr: float) -> bool:
    fixed_choices = {
        "architecture": {"affine", "maf", "spline"},
        "conditioner": {"affine", "maf", "spline"},
        "base_distribution": {"gaussian", "gaussian_trainable", "gmm"},
        "coupling_transform": {"affine", "spline4", "spline8"},
    }
    if surface in fixed_choices:
        return choice in fixed_choices[surface]
    try:
        if surface == "batch_size":
            value = int(choice)
            return str(value) == choice and 1 <= value <= 8192 and value == batch_size
        if surface == "conditioner_width":
            value = int(choice)
            return str(value) == choice and 2 <= value <= 512
        if surface == "depth":
            value = int(choice)
            return str(value) == choice and 1 <= value <= 32
        if surface == "learning_rate":
            value = float(choice)
            return math.isfinite(value) and 1e-6 <= value <= 1.0 and value == lr
        if surface == "spline_bins":
            value = int(choice)
            return str(value) == choice and 2 <= value <= 64
    except (TypeError, ValueError, OverflowError):
        return False
    if surface == "masking_pattern":
        return len(choice) == 16 and all(
            choice[index:index + 2] in {"01", "10"} for index in range(0, 16, 2)
        )
    return False


def _expected_layers(surface: str, choice: str) -> tuple[int, int]:
    if surface == "base_distribution":
        return 1, 1
    if surface == "masking_pattern":
        return 8, 0
    if surface == "depth":
        depth = int(choice)
        return depth, depth
    return 8, 8


class Parser(OutputParser):
    """Expose NLL only after a unique, ordered, terminal full-scale proof."""

    def parse(self, cmd_label: str, raw_output: str) -> ParseResult:
        task_id = _task_id()
        task_protocol = TASK_PROTOCOLS.get(task_id)
        if task_protocol is None:
            return _reject("unknown task identity")
        expected_surface, expected_targets = task_protocol
        if cmd_label not in expected_targets:
            return _reject("unknown or mismatched setting identity")
        if not raw_output.strip():
            return _reject("empty verifier output")
        if any(marker in raw_output for marker in FAILURE_MARKERS):
            return _reject("failure marker present")

        lines = [line.strip() for line in raw_output.splitlines() if line.strip()]
        records: dict[str, list[tuple[int, re.Match[str]]]] = {
            name: [] for name in _RECORDS
        }
        for index, line in enumerate(lines):
            matches = [
                (name, match)
                for name, pattern in _RECORDS.items()
                if (match := pattern.fullmatch(line)) is not None
            ]
            if line.startswith("FLOW_") and len(matches) != 1:
                return _reject("malformed or unknown protocol record")
            if len(matches) == 1:
                name, match = matches[0]
                records[name].append((index, match))

        for name in ("protocol", "design", "data", "metric", "complete"):
            if len(records[name]) != 1:
                return _reject(f"expected exactly one {name} proof")
        if len(records["train"]) != len(EXPECTED_STEPS):
            return _reject("training trace does not prove all 20,000 optimizer steps")

        protocol_index, protocol = records["protocol"][0]
        design_index, design = records["design"][0]
        data_index, data = records["data"][0]
        metric_index, metric = records["metric"][0]
        complete_index, complete = records["complete"][0]
        train_indices = [index for index, _match in records["train"]]
        if not (
            protocol_index < design_index < data_index < train_indices[0]
            and train_indices[-1] < metric_index < complete_index
            and complete_index == len(lines) - 1
        ):
            return _reject("proof records are reordered or completion is non-terminal")

        try:
            (
                surface,
                choice,
                target,
                device,
                device_count_raw,
                seed_raw,
                steps_raw,
                batch_size_raw,
                lr_raw,
                optimizer,
                objective,
            ) = protocol.groups()
            device_count = int(device_count_raw)
            seed = int(seed_raw)
            steps = int(steps_raw)
            batch_size = int(batch_size_raw)
            lr = float(lr_raw)

            (
                design_target,
                n_transforms_raw,
                n_permutations_raw,
                total_layers_raw,
                design_params_raw,
            ) = design.groups()
            n_transforms = int(n_transforms_raw)
            n_permutations = int(n_permutations_raw)
            total_layers = int(total_layers_raw)
            design_params = int(design_params_raw)

            data_target, data_seed_raw, n_train_raw, n_test_raw, train_sha, test_sha = (
                data.groups()
            )
            data_seed = int(data_seed_raw)
            n_train = int(n_train_raw)
            n_test = int(n_test_raw)

            train_steps = [int(match.group(1)) for _index, match in records["train"]]
            train_nlls = [float(match.group(2)) for _index, match in records["train"]]
            nll = float(metric.group(1))
            bpd = float(metric.group(2))
            metric_params = int(metric.group(3))
            elapsed = float(metric.group(4))

            (
                complete_surface,
                complete_choice,
                complete_target,
                complete_seed_raw,
                optimizer_steps_raw,
                samples_seen_raw,
                complete_n_train_raw,
                complete_n_test_raw,
                complete_params_raw,
            ) = complete.groups()
            complete_seed = int(complete_seed_raw)
            optimizer_steps = int(optimizer_steps_raw)
            samples_seen = int(samples_seen_raw)
            complete_n_train = int(complete_n_train_raw)
            complete_n_test = int(complete_n_test_raw)
            complete_params = int(complete_params_raw)
        except (TypeError, ValueError, OverflowError):
            return _reject("protocol proof contains an invalid scalar")

        if (
            surface != expected_surface
            or target != cmd_label
            or design_target != cmd_label
            or data_target != cmd_label
            or complete_surface != surface
            or complete_choice != choice
            or complete_target != cmd_label
        ):
            return _reject("task, surface, choice, or setting identity does not match")
        if (
            device != "cuda"
            or device_count != 1
            or seed != 42
            or data_seed != 42
            or complete_seed != 42
            or steps != 20_000
            or optimizer != "Adam"
            or objective != "exact_nll"
            or not math.isfinite(lr)
            or not 1e-6 <= lr <= 1.0
        ):
            return _reject("runtime or optimization protocol does not match")
        if surface != "batch_size" and batch_size != 512:
            return _reject("fixed batch-size protocol does not match")
        if surface != "learning_rate" and lr != 5e-4:
            return _reject("fixed learning-rate protocol does not match")
        if not _choice_is_valid(surface, choice, batch_size, lr):
            return _reject("selected design choice is invalid or inconsistent")

        expected_transforms, expected_permutations = _expected_layers(surface, choice)
        if (
            n_transforms != expected_transforms
            or n_permutations != expected_permutations
            or total_layers != n_transforms + n_permutations
            or not 1 <= design_params <= 100_000_000
            or metric_params != design_params
            or complete_params != design_params
        ):
            return _reject("layer accounting or parameter identity does not match")

        expected_train_sha, expected_test_sha = DATA_SHA256[cmd_label]
        if (
            n_train != 30_000
            or n_test != 30_000
            or complete_n_train != 30_000
            or complete_n_test != 30_000
            or train_sha != expected_train_sha
            or test_sha != expected_test_sha
        ):
            return _reject("data inventory or immutable identity does not match")
        if (
            train_steps != EXPECTED_STEPS
            or optimizer_steps != 20_000
            or samples_seen != batch_size * 20_000
        ):
            return _reject("optimizer-step or sample accounting is incomplete")

        numeric_values = [*train_nlls, nll, bpd, elapsed]
        if not all(math.isfinite(value) for value in numeric_values):
            return _reject("protocol contains NaN or Inf")
        if (
            any(abs(value) > 1_000_000 for value in [*train_nlls, nll, bpd])
            or not 0.0 < elapsed <= 172_800.0
        ):
            return _reject("metric or runtime is outside its valid range")
        expected_bpd = nll / (2.0 * math.log(2.0))
        if not math.isclose(bpd, expected_bpd, rel_tol=0.0, abs_tol=2e-6):
            return _reject("bits-per-dimension does not match exact NLL")

        return ParseResult(
            feedback=(
                f"Completed {cmd_label}: NLL={nll:.6f}, bpd={bpd:.6f}, "
                f"params={metric_params}, steps=20000, train/test=30000/30000, "
                f"elapsed={elapsed:.1f}s."
            ),
            metrics={f"nll_{cmd_label}": nll},
        )
