"""Fail-closed parser for the full-scale NLI training protocol."""
from __future__ import annotations

import math
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mlsbench.agent.parsers import OutputParser, ParseResult


PROTOCOL = "nli-full-snli-distilbert-v1"
EXPECTED_TASK = "nli-truncation"
EXPECTED_SURFACE = "truncation"
EXPECTED_POLICIES = frozenset({"majority"} | {f"len{value}" for value in range(8, 129)})
REQUIRE_CLASS_WEIGHTS = False
REGULARIZATION_BY_MODE: dict[str, tuple[float, float]] = {}
EXPECTED_MODEL_BY_MODE = {
    f"len{value}": ("cross", 2307) for value in range(8, 129)
}
EXPECTED_MAX_LENGTH_BY_MODE = {
    "majority": 128,
    **{f"len{value}": value for value in range(8, 129)},
}
TRAIN_ROWS = 549367
EVAL_ROWS = 29471
STEPS_PER_EPOCH = 17168
TOTAL_STEPS = 51504
EXPECTED_DATA = {
    "snli_train": (549367, "3cdde4e94e0c5ca8e7e3d95b0c7c7b9fc03b101d3b9e79c422150bf5c17f1f73"),
    "snli": (9824, "e30ea21eb677dab4806e1cc4c646dffc23985ffd982fd6bd15ab3617cd601dd8"),
    "mnli_m": (9815, "a612ccdf07b2fbe73e2904b061b9e278f552a39b553999bc626de6df6ec4b66d"),
    "mnli_mm": (9832, "a08757b4ddc34421f8f6eac69eb5dd97b2125693078c541cad2d54689013f68d"),
}
MODEL_REVISION = "12040accade4e8a0f71eabdb258fecc2e7e948be"
MODEL_CONFIG_SHA256 = "69c94b0222d5d1f4b0ad027ca7416cdafb98378cbbb8305d0bf47c9365c60c83"
MODEL_WEIGHTS_SHA256 = "5e3f1108e3cb34ee048634875d8482665b65ac713291a7e32396fb18f6ff0063"
TOKENIZER_SHA256 = "ce64fce797c24f68df90b40a3f74f579b336a493db14bd583fd520ea0d8c9a98"
TOKENIZER_CONFIG_SHA256 = "a025160ef0431f1a392f6f050c1310f4c5d9fb6f275932dbccba73c4d214bf10"
VOCAB_SHA256 = "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3"
MODEL_PARAMETERS = 66362880


def _unique_fullmatches(raw_output: str, prefix: str, pattern: re.Pattern[str]):
    matches = []
    malformed = 0
    for line in raw_output.splitlines():
        stripped = line.strip()
        if not stripped.startswith(prefix):
            continue
        match = pattern.fullmatch(stripped)
        if match is None:
            malformed += 1
        else:
            matches.append(match)
    return matches, malformed


def _training_fullmatches(
    raw_output: str,
    train_pattern: re.Pattern[str],
    done_pattern: re.Pattern[str],
):
    """Parse the reserved NLI_TRAIN* namespace without ignoring lookalikes."""
    trains = []
    completions = []
    malformed = 0
    for line in raw_output.splitlines():
        stripped = line.strip()
        if not stripped.startswith("NLI_TRAIN"):
            continue
        train_match = train_pattern.fullmatch(stripped)
        done_match = done_pattern.fullmatch(stripped)
        if train_match is not None:
            trains.append(train_match)
        elif done_match is not None:
            completions.append(done_match)
        else:
            malformed += 1
    return trains, completions, malformed


class Parser(OutputParser):
    def parse(self, cmd_label: str, raw_output: str) -> ParseResult:
        errors: list[str] = []
        metrics: dict[str, float] = {}

        if cmd_label != "nli":
            errors.append(f"unexpected command label {cmd_label!r}")
        nonempty_lines = [line.strip() for line in raw_output.splitlines() if line.strip()]
        if not nonempty_lines or nonempty_lines[-1] != "NLI_COMMAND_DONE rc=0":
            errors.append("successful command completion is not the terminal record")
        if re.search(
            r"(?:Traceback \(most recent call last\)|NLI_NONFINITE|SURFACE_ERROR|"
            r"VERIFICATION_FAILED|\[COMMAND FAILED|\[STATUS: FAILED|"
            r"\[BUDGET CHECK FAILED|\[exit code|\bTIMEOUT\b|"
            r"\bOUT_OF_MEMORY\b|\bCANCELLED\b|"
            r"\bNODE_FAIL\b|Segmentation fault|\[ERROR\])",
            raw_output,
        ):
            errors.append("runtime reported a failure or non-finite state")

        device_re = re.compile(r"NLI_DEVICE type=cuda visible=1")
        devices, malformed = _unique_fullmatches(raw_output, "NLI_DEVICE", device_re)
        if malformed or len(devices) != 1:
            errors.append(f"device proof malformed={malformed} count={len(devices)}")

        protocol_re = re.compile(
            rf"NLI_PROTOCOL version={PROTOCOL} task={re.escape(EXPECTED_TASK)} "
            rf"surface={re.escape(EXPECTED_SURFACE)} settings=3 train_rows={TRAIN_ROWS} "
            rf"eval_rows={EVAL_ROWS} epochs=3 train_batch=32 eval_batch=128 "
            rf"max_length=(\d+) seed=42"
        )
        protocol, malformed = _unique_fullmatches(
            raw_output, "NLI_PROTOCOL", protocol_re
        )
        if malformed or len(protocol) != 1:
            errors.append(
                f"protocol proof malformed={malformed} count={len(protocol)}"
            )
            protocol_max_length = None
        else:
            protocol_max_length = int(protocol[0].group(1))
            if not 8 <= protocol_max_length <= 128:
                errors.append("protocol sequence length outside fixed bounds")

        data_re = re.compile(
            r"NLI_DATA split=(\S+) rows=(\d+) sha256=([0-9a-f]{64})"
        )
        data_matches, malformed = _unique_fullmatches(raw_output, "NLI_DATA", data_re)
        if malformed:
            errors.append(f"malformed data proofs={malformed}")
        seen_data: set[str] = set()
        for match in data_matches:
            split, rows_text, digest = match.groups()
            if split not in EXPECTED_DATA or split in seen_data:
                errors.append(f"unexpected or duplicate data split {split!r}")
                continue
            seen_data.add(split)
            expected_rows, expected_digest = EXPECTED_DATA[split]
            if int(rows_text) != expected_rows:
                errors.append(f"wrong data row count for {split}")
            if digest != expected_digest:
                errors.append(f"wrong data digest for {split}")
        if seen_data != set(EXPECTED_DATA):
            errors.append(
                f"missing data proofs {sorted(set(EXPECTED_DATA) - seen_data)}"
            )

        mode_pattern = r"([a-z][a-z0-9_-]{0,31})"
        policy_re = re.compile(rf"NLI_POLICY mode={mode_pattern}")
        policies, malformed = _unique_fullmatches(
            raw_output, "NLI_POLICY", policy_re
        )
        if malformed or len(policies) != 1:
            errors.append(
                f"policy proof malformed={malformed} count={len(policies)}"
            )
            mode = None
        else:
            mode = policies[0].group(1)
            if mode not in EXPECTED_POLICIES:
                errors.append(f"policy {mode!r} is not valid for {EXPECTED_TASK}")
            elif protocol_max_length != EXPECTED_MAX_LENGTH_BY_MODE[mode]:
                errors.append("policy does not match protocol tokenizer cap")

        class_weight_re = re.compile(
            r"NLI_CLASS_WEIGHTS entailment=(\S+) neutral=(\S+) contradiction=(\S+)"
        )
        class_weights, malformed_weights = _unique_fullmatches(
            raw_output, "NLI_CLASS_WEIGHTS", class_weight_re
        )
        if REQUIRE_CLASS_WEIGHTS:
            if malformed_weights or len(class_weights) != 1:
                errors.append(
                    "class-weight proof malformed="
                    f"{malformed_weights} count={len(class_weights)}"
                )
            else:
                try:
                    weights = tuple(float(value) for value in class_weights[0].groups())
                except ValueError:
                    weights = (math.nan, math.nan, math.nan)
                if (any(not math.isfinite(value) or not 0.25 <= value <= 2.0
                        for value in weights)
                        or not math.isclose(
                            sum(weights), 3.0, rel_tol=0.0, abs_tol=1e-9
                        )):
                    errors.append("invalid class-weight proof")
        elif malformed_weights or class_weights:
            errors.append("unexpected class-weight proof")

        regularization_re = re.compile(
            r"NLI_REGULARIZATION dropout=(\S+) weight_decay=(\S+)"
        )
        regularization, malformed_regularization = _unique_fullmatches(
            raw_output, "NLI_REGULARIZATION", regularization_re
        )
        if REGULARIZATION_BY_MODE:
            if mode == "majority":
                if malformed_regularization or regularization:
                    errors.append("majority diagnostic emitted a regularization proof")
            elif malformed_regularization or len(regularization) != 1:
                errors.append(
                    "regularization proof malformed="
                    f"{malformed_regularization} count={len(regularization)}"
                )
            else:
                try:
                    actual_regularization = tuple(
                        float(value) for value in regularization[0].groups()
                    )
                except ValueError:
                    actual_regularization = (math.nan, math.nan)
                expected_regularization = REGULARIZATION_BY_MODE.get(mode)
                if (
                    expected_regularization is None
                    or any(not math.isfinite(value) for value in actual_regularization)
                    or actual_regularization != expected_regularization
                ):
                    errors.append("regularization proof does not match policy")
        elif malformed_regularization or regularization:
            errors.append("unexpected regularization proof")

        model_re = re.compile(
            r"NLI_MODEL model=distilbert-base-uncased "
            rf"revision=({MODEL_REVISION}) architecture=(\S+) "
            r"encoder_params=(\d+) head_params=(\d+) total_params=(\d+) "
            r"dtype=(\S+) "
            rf"config_sha256={MODEL_CONFIG_SHA256} "
            rf"weights_sha256={MODEL_WEIGHTS_SHA256} "
            rf"tokenizer_sha256={TOKENIZER_SHA256} "
            rf"tokenizer_config_sha256={TOKENIZER_CONFIG_SHA256} "
            rf"vocab_sha256={VOCAB_SHA256}"
        )
        models, malformed = _unique_fullmatches(raw_output, "NLI_MODEL", model_re)
        if malformed:
            errors.append(f"malformed model proofs={malformed}")

        train_re = re.compile(
            rf"NLI_TRAIN mode={mode_pattern} optimizer=adamw "
            r"encoder_lr=(\S+) head_lr=(\S+) weight_decay=(\S+) "
            r"warmup_ratio=(\S+) epochs=(\d+) batch=(\d+) "
            r"max_length=(\d+) expected_steps=(\d+)"
        )

        epoch_re = re.compile(
            rf"NLI_EPOCH mode={mode_pattern} epoch=(\d+) "
            r"optimizer_steps=(\d+) expected=(\d+) loss=(\S+)"
        )
        epochs, malformed_epochs = _unique_fullmatches(
            raw_output, "NLI_EPOCH", epoch_re
        )

        train_done_re = re.compile(
            r"NLI_TRAIN_DONE epochs=(\d+) optimizer_steps=(\d+) "
            r"expected_steps=(\d+)"
        )
        trains, train_done, malformed_training = _training_fullmatches(
            raw_output, train_re, train_done_re
        )
        if malformed_training:
            errors.append(f"malformed training proofs={malformed_training}")

        if mode == "majority":
            if models or malformed:
                errors.append("majority diagnostic emitted a model proof")
            if trains:
                errors.append("majority diagnostic emitted a neural train proof")
            if epochs or malformed_epochs:
                errors.append("majority diagnostic emitted epoch proofs")
            if len(train_done) != 1:
                errors.append("missing majority completion proof")
            elif tuple(map(int, train_done[0].groups())) != (0, 0, 0):
                errors.append("wrong majority completion inventory")
        elif mode is not None:
            if len(models) != 1:
                errors.append(f"model proof count={len(models)}")
            else:
                (_, architecture, encoder_text, head_text, total_text,
                 dtype) = models[0].groups()
                expected_model = EXPECTED_MODEL_BY_MODE.get(mode)
                encoder_params = int(encoder_text)
                head_params = int(head_text)
                total_params = int(total_text)
                if (
                    expected_model is None
                    or architecture != expected_model[0]
                    or encoder_params != MODEL_PARAMETERS
                    or head_params != expected_model[1]
                    or total_params != encoder_params + head_params
                ):
                    errors.append("model architecture or parameter proof mismatch")
                if dtype != "float32":
                    errors.append("wrong model dtype")

            if len(trains) != 1:
                errors.append(f"train proof count={len(trains)}")
            else:
                (train_mode, encoder_lr_text, head_lr_text, wd_text, warmup_text,
                 epoch_text, batch_text, max_length_text, steps_text) = trains[0].groups()
                try:
                    encoder_lr = float(encoder_lr_text)
                    head_lr = float(head_lr_text)
                    weight_decay = float(wd_text)
                    warmup = float(warmup_text)
                except ValueError:
                    encoder_lr = head_lr = weight_decay = warmup = math.nan
                expected_encoder_lr = 0.0 if mode == "frozen" else 2e-5
                expected_weight_decay = (
                    REGULARIZATION_BY_MODE[mode][1]
                    if mode in REGULARIZATION_BY_MODE
                    else 0.01
                )
                if (train_mode != mode
                        or encoder_lr != expected_encoder_lr
                        or head_lr != 1e-3
                        or weight_decay != expected_weight_decay
                        or warmup != 0.1
                        or (int(epoch_text), int(batch_text), int(max_length_text),
                            int(steps_text)) != (
                                3, 32, protocol_max_length, TOTAL_STEPS
                            )):
                    errors.append("wrong fixed training protocol")

            if malformed_epochs or len(epochs) != 3:
                errors.append(
                    f"epoch proofs malformed={malformed_epochs} count={len(epochs)}"
                )
            else:
                seen_epochs: set[int] = set()
                for match in epochs:
                    epoch_mode, epoch_text, steps_text, expected_text, loss_text = match.groups()
                    epoch = int(epoch_text)
                    try:
                        loss = float(loss_text)
                    except ValueError:
                        loss = math.nan
                    expected_steps = epoch * STEPS_PER_EPOCH
                    if (epoch_mode != mode or epoch not in {1, 2, 3}
                            or epoch in seen_epochs
                            or int(steps_text) != expected_steps
                            or int(expected_text) != expected_steps
                            or not math.isfinite(loss) or loss < 0.0):
                        errors.append(f"invalid epoch completion {epoch}")
                    seen_epochs.add(epoch)
                if seen_epochs != {1, 2, 3}:
                    errors.append("incomplete epoch inventory")

            if len(train_done) != 1:
                errors.append(f"train completion count={len(train_done)}")
            elif tuple(map(int, train_done[0].groups())) != (3, TOTAL_STEPS, TOTAL_STEPS):
                errors.append("wrong final optimizer step")
        else:
            errors.append("training policy could not be established")

        metrics_re = re.compile(
            r"NLI_METRICS setting=(\S+) acc=(\S+) n_eval=(\d+)"
        )
        metric_matches, malformed = _unique_fullmatches(
            raw_output, "NLI_METRICS", metrics_re
        )
        if malformed:
            errors.append(f"malformed metric lines={malformed}")
        seen_metrics: set[str] = set()
        for match in metric_matches:
            setting, accuracy_text, rows_text = match.groups()
            if setting not in EXPECTED_DATA or setting == "snli_train" or setting in seen_metrics:
                errors.append(f"unexpected or duplicate metric setting {setting!r}")
                continue
            seen_metrics.add(setting)
            try:
                accuracy = float(accuracy_text)
            except ValueError:
                accuracy = math.nan
            expected_rows = EXPECTED_DATA[setting][0]
            if (not math.isfinite(accuracy) or not 0.0 <= accuracy <= 1.0
                    or int(rows_text) != expected_rows):
                errors.append(f"invalid metric for {setting}")
                continue
            metrics[f"acc_{setting}"] = accuracy
        expected_settings = {"snli", "mnli_m", "mnli_mm"}
        if seen_metrics != expected_settings:
            errors.append(
                f"missing metric settings {sorted(expected_settings - seen_metrics)}"
            )

        setting_done_re = re.compile(
            r"NLI_SETTING_DONE setting=(\S+) predicted=(\d+) expected=(\d+)"
        )
        setting_done, malformed = _unique_fullmatches(
            raw_output, "NLI_SETTING_DONE", setting_done_re
        )
        if malformed:
            errors.append(f"malformed setting completion proofs={malformed}")
        seen_done: set[str] = set()
        for match in setting_done:
            setting, predicted_text, expected_text = match.groups()
            if setting not in expected_settings or setting in seen_done:
                errors.append(f"unexpected or duplicate setting completion {setting!r}")
                continue
            seen_done.add(setting)
            rows = EXPECTED_DATA[setting][0]
            if int(predicted_text) != rows or int(expected_text) != rows:
                errors.append(f"incomplete predictions for {setting}")
        if seen_done != expected_settings:
            errors.append(f"missing setting completions {sorted(expected_settings - seen_done)}")

        eval_done_re = re.compile(rf"NLI_EVAL_DONE settings=3 rows={EVAL_ROWS}")
        eval_done, malformed = _unique_fullmatches(
            raw_output, "NLI_EVAL_DONE", eval_done_re
        )
        if malformed or len(eval_done) != 1:
            errors.append(
                f"evaluation completion malformed={malformed} count={len(eval_done)}"
            )

        final_re = re.compile(
            rf"NLI_DONE settings=3 train_rows={TRAIN_ROWS} eval_rows={EVAL_ROWS} "
            r"seed=42 elapsed=(\S+)"
        )
        final, malformed = _unique_fullmatches(raw_output, "NLI_DONE", final_re)
        if malformed or len(final) != 1:
            errors.append(f"final proof malformed={malformed} count={len(final)}")
        else:
            try:
                elapsed = float(final[0].group(1))
            except ValueError:
                elapsed = math.nan
            if not math.isfinite(elapsed) or elapsed <= 0.0:
                errors.append("invalid final elapsed time")

        command_re = re.compile(r"NLI_COMMAND_DONE rc=(\d+)")
        command, malformed = _unique_fullmatches(
            raw_output, "NLI_COMMAND_DONE", command_re
        )
        if malformed or len(command) != 1 or (
                command and int(command[0].group(1)) != 0):
            errors.append(
                f"command completion malformed={malformed} count={len(command)}"
            )

        if errors:
            return ParseResult(
                feedback="Rejected incomplete NLI verification: " + "; ".join(errors),
                metrics={},
            )

        summary = "\n".join(
            f"[{setting}] accuracy={metrics[f'acc_{setting}']:.8f} "
            f"n={EXPECTED_DATA[setting][0]}"
            for setting in ("snli", "mnli_m", "mnli_mm")
        )
        return ParseResult(
            feedback="Full-scale NLI protocol completed:\n" + summary,
            metrics=metrics,
        )
