"""Strict completion-proof parser shared by the Mamba sibling tasks."""
from __future__ import annotations

import math
import re

from mlsbench.agent.parsers import OutputParser, ParseResult


NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
COMMON = (
    r"protocol=(?P<protocol>[A-Za-z0-9_.-]+) "
    r"task=(?P<task>[A-Za-z0-9_.-]+) "
    r"label=(?P<label>[A-Za-z0-9_.-]+) "
    r"surface=(?P<surface>[A-Za-z0-9_.-]+) "
    r"L=(?P<L>\d+) M=(?P<M>\d+) A=(?P<A>\d+) "
    r"d_model=(?P<d_model>\d+) d_state=(?P<d_state>\d+) "
    r"n_layer=(?P<n_layer>\d+) steps=(?P<steps>\d+) "
    rf"batch=(?P<batch>\d+) lr=(?P<lr>{NUMBER}) "
    r"optimizer=(?P<optimizer>[A-Za-z0-9_.-]+) "
    rf"weight_decay=(?P<weight_decay>{NUMBER}) "
    rf"grad_clip=(?P<grad_clip>{NUMBER}) eval_batches=(?P<eval_batches>\d+) "
    r"seed=(?P<seed>-?\d+) n_params=(?P<n_params>\d+) "
    r"train_examples=(?P<train_examples>\d+) train_tokens=(?P<train_tokens>\d+) "
    r"eval_examples=(?P<eval_examples>\d+) eval_tokens=(?P<eval_tokens>\d+)"
)
POOL = re.compile(rf"POOL_LOADED {COMMON} device=(?P<device>cpu|cuda)")
TRAIN_COMPLETE = re.compile(
    rf"MAMBA_TRAIN_COMPLETE {COMMON} final_loss=(?P<train_loss>{NUMBER})"
)
EVAL_COMPLETE = re.compile(
    rf"MAMBA_EVAL_COMPLETE {COMMON} eval_correct=(?P<eval_correct>\d+) "
    rf"copy_acc=(?P<eval_acc>{NUMBER})"
)
ANY_METRIC = re.compile(r"MAMBA_[A-Z_]+_METRICS(?:\s|$)")
BOUND_FIELDS = (
    "protocol", "task", "label", "surface", "L", "M", "A", "d_model", "d_state",
    "n_layer", "steps", "batch", "lr", "optimizer", "weight_decay",
    "grad_clip", "eval_batches", "seed", "n_params", "train_examples",
    "train_tokens", "eval_examples", "eval_tokens",
)


class StrictMambaParser(OutputParser):
    """Require unique, configuration-bound train/eval completion records."""

    TASK = ""
    METRIC_KIND = "MAMBA_COPY_METRICS"
    EXPECTED_PROTOCOL = "mamba_selective_copy_paper_e1_v1"
    EXPECTED_SURFACES: set[str] = set()
    EXPECTED_STEPS = 400000
    EXPECTED_N_LAYER = 2
    EXPECTED_LENGTHS = {"paper_e1": 4096}
    EXPECTED_M = 16
    EXPECTED_A = 16
    EXPECTED_D_MODEL = 64
    EXPECTED_D_STATE = 16
    EXPECTED_BATCH = 64
    EXPECTED_LR = 1e-4
    EXPECTED_OPTIMIZER = "adam"
    EXPECTED_WEIGHT_DECAY = 0.0
    EXPECTED_GRAD_CLIP = 1.0
    EXPECTED_EVAL_BATCHES = 16
    EXPECTED_SEEDS = {42}

    @staticmethod
    def _reject(reason: str) -> ParseResult:
        return ParseResult(feedback=f"Mamba verification rejected: {reason}", metrics={})

    @staticmethod
    def _proof_float(value: float) -> str:
        return format(float(value), ".12g")

    def parse(self, cmd_label: str, raw_output: str) -> ParseResult:
        if cmd_label not in self.EXPECTED_LENGTHS:
            return self._reject("unknown command label")
        lines = [line.strip() for line in raw_output.splitlines() if line.strip()]
        if any(
            line.startswith((
                "MAMBA_NONFINITE", "TRAIN_ERROR", "EVAL_FAILED", "Traceback ",
                "[ERROR]", "[TIMEOUT]", "Killed",
            ))
            or "CUDA out of memory" in line
            for line in lines
        ):
            return self._reject("the harness reported a failure")
        pool_lines = [line for line in lines if line.startswith("POOL_LOADED")]
        train_lines = [
            line for line in lines if line.startswith("MAMBA_TRAIN_COMPLETE")
        ]
        eval_lines = [
            line for line in lines if line.startswith("MAMBA_EVAL_COMPLETE")
        ]
        metric_lines = [line for line in lines if ANY_METRIC.match(line)]
        for kind, records in (
            ("POOL_LOADED", pool_lines),
            ("training-completion", train_lines),
            ("evaluation-completion", eval_lines),
            ("MAMBA metrics", metric_lines),
        ):
            if len(records) != 1:
                return self._reject(f"expected exactly one {kind} record")
        if not (
            lines.index(pool_lines[0])
            < lines.index(train_lines[0])
            < lines.index(eval_lines[0])
            < lines.index(metric_lines[0])
        ):
            return self._reject("completion records are out of order")

        pool = POOL.fullmatch(pool_lines[0])
        train = TRAIN_COMPLETE.fullmatch(train_lines[0])
        evaluation = EVAL_COMPLETE.fullmatch(eval_lines[0])
        metric_pattern = re.compile(
            rf"{re.escape(self.METRIC_KIND)} {COMMON} "
            rf"copy_acc=(?P<copy_acc>{NUMBER}) "
            rf"final_loss=(?P<metric_loss>{NUMBER}) wall_s=(?P<wall_s>{NUMBER}) "
            r"eval_correct=(?P<metric_correct>\d+)"
        )
        metric = metric_pattern.fullmatch(metric_lines[0])
        if pool is None or train is None or evaluation is None or metric is None:
            return self._reject("a completion record is malformed or has the wrong kind")
        if pool.group("device") != "cuda":
            return self._reject("formal Mamba verification requires CUDA")

        length = self.EXPECTED_LENGTHS[cmd_label]
        expected = {
            "protocol": self.EXPECTED_PROTOCOL,
            "task": self.TASK,
            "label": cmd_label,
            "L": str(length),
            "M": str(self.EXPECTED_M),
            "A": str(self.EXPECTED_A),
            "d_model": str(self.EXPECTED_D_MODEL),
            "d_state": str(self.EXPECTED_D_STATE),
            "n_layer": str(self.EXPECTED_N_LAYER),
            "steps": str(self.EXPECTED_STEPS),
            "batch": str(self.EXPECTED_BATCH),
            "lr": self._proof_float(self.EXPECTED_LR),
            "optimizer": self.EXPECTED_OPTIMIZER,
            "weight_decay": self._proof_float(self.EXPECTED_WEIGHT_DECAY),
            "grad_clip": self._proof_float(self.EXPECTED_GRAD_CLIP),
            "eval_batches": str(self.EXPECTED_EVAL_BATCHES),
            "train_examples": str(self.EXPECTED_STEPS * self.EXPECTED_BATCH),
            "train_tokens": str(
                self.EXPECTED_STEPS * self.EXPECTED_BATCH * length
            ),
            "eval_examples": str(
                self.EXPECTED_EVAL_BATCHES * self.EXPECTED_BATCH
            ),
            "eval_tokens": str(
                self.EXPECTED_EVAL_BATCHES * self.EXPECTED_BATCH * length
            ),
        }
        records = (pool, train, evaluation, metric)
        for field, value in expected.items():
            if any(record.group(field) != value for record in records):
                return self._reject(f"proof field {field} does not match the protocol")
        if any(int(record.group("seed")) not in self.EXPECTED_SEEDS for record in records):
            return self._reject("proof seed does not match the protocol")
        if not self.EXPECTED_SURFACES or any(
            record.group("surface") not in self.EXPECTED_SURFACES
            for record in records
        ):
            return self._reject("proof surface does not match the task")
        for field in BOUND_FIELDS:
            if len({record.group(field) for record in records}) != 1:
                return self._reject(f"proof field {field} is inconsistent")

        n_params = int(metric.group("n_params"))
        try:
            train_loss = float(train.group("train_loss"))
            metric_loss = float(metric.group("metric_loss"))
            eval_acc = float(evaluation.group("eval_acc"))
            copy_acc = float(metric.group("copy_acc"))
            wall_s = float(metric.group("wall_s"))
        except ValueError:
            return self._reject("metric values are not numeric")
        eval_correct = int(evaluation.group("eval_correct"))
        metric_correct = int(metric.group("metric_correct"))
        if n_params <= 0:
            return self._reject("n_params must be positive")
        if not all(math.isfinite(value) for value in (
            train_loss, metric_loss, eval_acc, copy_acc, wall_s,
        )):
            return self._reject("metric values must be finite")
        if train.group("train_loss") != metric.group("metric_loss"):
            return self._reject("final loss does not match the training proof")
        if evaluation.group("eval_acc") != metric.group("copy_acc"):
            return self._reject("accuracy does not match the evaluation proof")
        if eval_correct != metric_correct:
            return self._reject("correct count does not match the evaluation proof")
        prediction_count = int(metric.group("eval_examples")) * self.EXPECTED_M
        if not 0 <= eval_correct <= prediction_count:
            return self._reject("evaluation correct count is outside its valid range")
        if abs(copy_acc - eval_correct / prediction_count) > 0.5e-6:
            return self._reject("accuracy does not match the evaluation cardinality")
        if train_loss < 0 or not 0.0 <= copy_acc <= 1.0 or wall_s <= 0:
            return self._reject("metric values are outside their valid ranges")

        return ParseResult(
            feedback=(
                f"Mamba {cmd_label}: copy_acc={copy_acc:.6f}, "
                f"final_loss={metric_loss:.6f}, n_params={n_params}"
            ),
            metrics={f"copy_acc_{cmd_label}": copy_acc},
        )
