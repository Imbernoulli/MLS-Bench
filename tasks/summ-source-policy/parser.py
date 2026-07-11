"""Fail-closed parser for the full official summarization protocol."""
from __future__ import annotations

import math
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mlsbench.agent.parsers import OutputParser, ParseResult


SETTINGS = ("xsum", "cnndm", "samsum")
EXPECTED = {
    "xsum": {
        "rows": 11334,
        "data_sha256": "5b7854e6ee8b81493e38a5255ba794dec21735b30ba8f77280a394bf7d6dae53",
        "model": "distilbart-xsum-12-6",
        "revision": "5b2e376c845c201ddc34ec0e55fd1ad9890ba5ee",
        "params": 305510400,
        "weights_sha256": "6e9ebfc94e474225457570ad33c225cf66ca26279c5cd1cbfb67e089a03a791b",
    },
    "cnndm": {
        "rows": 11490,
        "data_sha256": "525e5bf29cb49a1ea2c58939d11435b6d6422f9a59d9c15e43d226c464b33278",
        "model": "distilbart-cnn-12-6",
        "revision": "a4f8f3ea906ed274767e9906dbaede7531d660ff",
        "params": 305510400,
        "weights_sha256": "3bac65d18c99463302d12ca75c2220ea714f9c81ce235f205fa818efe71df6ea",
    },
    "samsum": {
        "rows": 819,
        "data_sha256": "b9dd8165689b1c252b2db617b089b408ea068ef5bd6819a93c64e8ebde856b86",
        "model": "bart-large-cnn-samsum",
        "revision": "e49b3d60d923f12db22bdd363356f1a4c68532ad",
        "params": 406290432,
        "weights_sha256": "9f453aa6edef4dba1893723b7313b57b06b60214442d308a8acc3baa9583dd7b",
    },
}
TOTAL_DOCS = sum(int(EXPECTED[setting]["rows"]) for setting in SETTINGS)


SURFACE_PATTERNS = {
    "summ-beam-repetition": re.compile(
        r"SUMM_BEAM num_beams=(\d+) no_repeat_ngram_size=(\d+) "
        r"repetition_penalty=(\S+)"
    ),
    "summ-beam-width": re.compile(r"SUMM_BEAMWIDTH num_beams=(\d+)"),
    "summ-decoding-length": re.compile(
        r"SUMM_LENGTH min_length=(\d+) max_length=(\d+) length_penalty=(\S+)"
    ),
    "summ-decoding-temperature": re.compile(r"SUMM_TEMPERATURE temperature=(\S+)"),
    "summ-diverse-beam": re.compile(
        r"SUMM_DIVERSE num_beams=(\d+) num_beam_groups=(\d+) "
        r"diversity_penalty=(\S+)"
    ),
    "summ-norepeat-ngram": re.compile(
        r"SUMM_NOREPEAT no_repeat_ngram_size=(\d+)"
    ),
    "summ-nucleus-topp": re.compile(r"SUMM_TOPP top_p=(\S+)"),
    "summ-post-truncation": re.compile(r"SUMM_POSTTRUNC keep_sentences=(\d+)"),
    "summ-sampling-vs-beam": re.compile(
        r"SUMM_STRATEGY strategy=(sample|beam) num_beams=(\S+) "
        r"top_p=(\S+) top_k=(\S+) temperature=(\S+)"
    ),
    "summ-source-policy": re.compile(
        r"SUMM_SOURCE policy=(abstractive|lead3|copy_document|first_token|empty)"
    ),
}
SURFACE_PREFIXES = {
    label: pattern.pattern.split(" ", 1)[0] for label, pattern in SURFACE_PATTERNS.items()
}

PROTOCOL_RE = re.compile(
    rf"SUMM_PROTOCOL version=summ-full-official-test-v1 settings=3 total_docs={TOTAL_DOCS}"
)
DATA_RE = re.compile(
    r"SUMM_DATA setting=(\S+) n_docs=(\d+) sha256=([0-9a-f]{64})"
)
MODEL_RE = re.compile(
    r"SUMM_MODEL setting=(\S+) model=(\S+) revision=([0-9a-f]{40}) "
    r"params=(\d+) dtype=(\S+) weights_sha256=([0-9a-f]{64})"
)
METRIC_RE = re.compile(
    r"SUMM_METRICS setting=(\S+) rougeL=(\S+) rouge1=(\S+) "
    r"rouge2=(\S+) plen=(\S+) n_docs=(\d+)"
)
SETTING_DONE_RE = re.compile(
    r"SUMM_SETTING_DONE setting=(\S+) generated=(\d+) expected=(\d+)"
)
EVAL_DONE_RE = re.compile(rf"SUMM_EVAL_DONE settings=3 total_docs={TOTAL_DOCS}")
FINAL_RE = re.compile(
    rf"SUMM_DONE settings=3 total_docs={TOTAL_DOCS} seed=42 elapsed=(\S+)"
)
PROGRESS_RE = re.compile(
    r"SUMM_PROGRESS setting=(\S+) completed=(\d+)/(\d+)"
)

FAILURE_MARKERS = (
    "traceback (most recent call last)",
    "surface_error",
    "summ_nonfinite",
    "verification_failed",
    "command exited with code",
    "command exited with non-zero status",
    "non-zero exit",
    "out of memory",
    "node_fail",
    "segmentation fault",
    "assertionerror",
    "runtimeerror",
    "[error]",
    "timeout",
    "cancelled",
    "canceled",
    "killed",
)


def _has_prefix(line: str, prefix: str) -> bool:
    return line == prefix or line.startswith(prefix + " ")


def _finite(text: str, low: float, high: float, *, low_open: bool = False) -> bool:
    try:
        value = float(text)
    except ValueError:
        return False
    if not math.isfinite(value):
        return False
    return (value > low if low_open else value >= low) and value <= high


def _validate_surface(cmd_label: str, line: str) -> tuple[list[str], bool]:
    """Validate the task-specific surface proof and return model-required state."""
    errors: list[str] = []
    match = SURFACE_PATTERNS[cmd_label].fullmatch(line)
    if match is None:
        return [f"invalid surface proof for {cmd_label}"], True
    values = match.groups()

    if cmd_label == "summ-beam-repetition":
        beams, ngram, penalty = int(values[0]), int(values[1]), values[2]
        if not 1 <= beams <= 12 or not 0 <= ngram <= 20:
            errors.append("beam/repetition surface outside bounds")
        if not _finite(penalty, 0.0, 10.0, low_open=True):
            errors.append("invalid repetition penalty")
    elif cmd_label == "summ-beam-width":
        if not 1 <= int(values[0]) <= 12:
            errors.append("beam width outside bounds")
    elif cmd_label == "summ-decoding-length":
        minimum, maximum, penalty = int(values[0]), int(values[1]), values[2]
        if not 0 <= minimum <= maximum <= 200 or maximum < 1:
            errors.append("invalid length window")
        if not _finite(penalty, 0.0, 10.0, low_open=True):
            errors.append("invalid length penalty")
    elif cmd_label == "summ-decoding-temperature":
        if not _finite(values[0], 0.0, 5.0, low_open=True):
            errors.append("invalid temperature")
    elif cmd_label == "summ-diverse-beam":
        beams, groups, penalty = int(values[0]), int(values[1]), values[2]
        if not 1 <= beams <= 12 or not 1 <= groups <= beams or beams % groups:
            errors.append("invalid diverse beam grouping")
        valid_penalty = _finite(penalty, 0.0, 10.0)
        if not valid_penalty:
            errors.append("invalid diversity penalty")
        elif groups == 1 and float(penalty) != 0.0:
            errors.append("plain beam must not claim a diversity penalty")
    elif cmd_label == "summ-norepeat-ngram":
        if not 0 <= int(values[0]) <= 20:
            errors.append("no-repeat ngram size outside bounds")
    elif cmd_label == "summ-nucleus-topp":
        if not _finite(values[0], 0.05, 1.0):
            errors.append("top-p outside bounds")
    elif cmd_label == "summ-post-truncation":
        if not 0 <= int(values[0]) <= 10000:
            errors.append("post-truncation count outside bounds")
    elif cmd_label == "summ-sampling-vs-beam":
        strategy, beams, top_p, top_k, temperature = values
        if strategy == "beam":
            if not beams.isdigit() or not 1 <= int(beams) <= 12:
                errors.append("invalid strategy beam width")
            if (top_p, top_k, temperature) != ("None", "None", "None"):
                errors.append("beam strategy emitted sampling controls")
        else:
            if beams != "None":
                errors.append("sample strategy emitted beam width")
            if not _finite(top_p, 0.0, 1.0, low_open=True):
                errors.append("invalid strategy top-p")
            if not top_k.isdigit() or not 0 <= int(top_k) <= 1000:
                errors.append("invalid strategy top-k")
            if not _finite(temperature, 0.0, 5.0, low_open=True):
                errors.append("invalid strategy temperature")
    elif cmd_label == "summ-source-policy":
        return errors, values[0] == "abstractive"
    return errors, True


def _event(line: str) -> str | None:
    for prefix in SURFACE_PREFIXES.values():
        if _has_prefix(line, prefix):
            return "surface"
    for prefix, regex, name in (
        ("SUMM_PROTOCOL", PROTOCOL_RE, "protocol"),
        ("SUMM_DATA", DATA_RE, "data"),
        ("SUMM_MODEL", MODEL_RE, "model"),
        ("SUMM_METRICS", METRIC_RE, "metric"),
        ("SUMM_SETTING_DONE", SETTING_DONE_RE, "setting_done"),
        ("SUMM_EVAL_DONE", EVAL_DONE_RE, "eval_done"),
        ("SUMM_DONE", FINAL_RE, "final"),
    ):
        if _has_prefix(line, prefix):
            match = regex.fullmatch(line)
            if name in {"data", "model", "metric", "setting_done"}:
                setting = match.group(1) if match is not None else "?"
                return f"{name}:{setting}"
            return name
    if line.startswith("SUMM_") and PROGRESS_RE.fullmatch(line) is None:
        return "unknown"
    return None


class Parser(OutputParser):
    def parse(self, cmd_label: str, raw_output: str) -> ParseResult:
        if cmd_label not in SURFACE_PATTERNS:
            return ParseResult(
                feedback=f"Rejected unexpected summarization label {cmd_label!r}",
                metrics={},
            )

        errors: list[str] = []
        metrics: dict[str, float] = {}
        nonempty = [line.strip() for line in raw_output.splitlines() if line.strip()]
        lowered = raw_output.lower()
        present_failures = [marker for marker in FAILURE_MARKERS if marker in lowered]
        if present_failures:
            errors.append("failure markers=" + ",".join(present_failures))

        proof_lines = [(index, line, _event(line)) for index, line in enumerate(nonempty)]
        proof_lines = [item for item in proof_lines if item[2] is not None]
        if any(event == "unknown" for _, _, event in proof_lines):
            errors.append("unknown SUMM_ proof record")

        surface_lines = [line for _, line, event in proof_lines if event == "surface"]
        model_required = True
        if len(surface_lines) != 1:
            errors.append(f"surface proof count={len(surface_lines)}")
        else:
            expected_prefix = SURFACE_PREFIXES[cmd_label]
            if not _has_prefix(surface_lines[0], expected_prefix):
                errors.append("surface proof belongs to another sibling")
            surface_errors, model_required = _validate_surface(cmd_label, surface_lines[0])
            errors.extend(surface_errors)

        expected_events = ["surface", "protocol"]
        for setting in SETTINGS:
            expected_events.append(f"data:{setting}")
            if model_required:
                expected_events.append(f"model:{setting}")
            expected_events.extend((f"metric:{setting}", f"setting_done:{setting}"))
        expected_events.extend(("eval_done", "final"))
        actual_events = [event for _, _, event in proof_lines]
        if actual_events != expected_events:
            errors.append("proof records are missing, duplicated, unexpected, or out of order")

        protocols = [line for _, line, event in proof_lines if event == "protocol"]
        if len(protocols) != 1 or PROTOCOL_RE.fullmatch(protocols[0]) is None:
            errors.append("invalid protocol proof")

        seen_data: set[str] = set()
        for _, line, event in proof_lines:
            if not event.startswith("data:"):
                continue
            match = DATA_RE.fullmatch(line)
            if match is None:
                errors.append("malformed data proof")
                continue
            setting, count_text, digest = match.groups()
            if setting not in EXPECTED or setting in seen_data:
                errors.append(f"unexpected or duplicate data setting {setting!r}")
                continue
            seen_data.add(setting)
            expected = EXPECTED[setting]
            if int(count_text) != expected["rows"] or digest != expected["data_sha256"]:
                errors.append(f"wrong data inventory for {setting}")
        if seen_data != set(SETTINGS):
            errors.append("incomplete data inventory")

        seen_models: set[str] = set()
        for _, line, event in proof_lines:
            if not event.startswith("model:"):
                continue
            match = MODEL_RE.fullmatch(line)
            if match is None:
                errors.append("malformed model proof")
                continue
            setting, model, revision, params, dtype, digest = match.groups()
            if setting not in EXPECTED or setting in seen_models:
                errors.append(f"unexpected or duplicate model setting {setting!r}")
                continue
            seen_models.add(setting)
            expected = EXPECTED[setting]
            if (
                model != expected["model"]
                or revision != expected["revision"]
                or int(params) != expected["params"]
                or dtype != "float16"
                or digest != expected["weights_sha256"]
            ):
                errors.append(f"wrong exact checkpoint identity for {setting}")
        expected_models = set(SETTINGS) if model_required else set()
        if seen_models != expected_models:
            errors.append("incorrect model proof coverage")

        seen_metrics: set[str] = set()
        for _, line, event in proof_lines:
            if not event.startswith("metric:"):
                continue
            match = METRIC_RE.fullmatch(line)
            if match is None:
                errors.append("malformed metric proof")
                continue
            setting, rouge_l, rouge_1, rouge_2, pred_len, count_text = match.groups()
            if setting not in EXPECTED or setting in seen_metrics:
                errors.append(f"unexpected or duplicate metric setting {setting!r}")
                continue
            seen_metrics.add(setting)
            try:
                values = [float(rouge_l), float(rouge_1), float(rouge_2)]
                plen = float(pred_len)
            except ValueError:
                errors.append(f"unparseable metrics for {setting}")
                continue
            if (
                not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in values)
                or not math.isfinite(plen)
                or not 0.0 <= plen <= 10000.0
                or int(count_text) != EXPECTED[setting]["rows"]
            ):
                errors.append(f"invalid metric inventory for {setting}")
                continue
            metrics[f"rougeL_{setting}"] = values[0]
            metrics[f"rouge1_{setting}"] = values[1]
            metrics[f"rouge2_{setting}"] = values[2]
        if seen_metrics != set(SETTINGS):
            errors.append("incomplete metric coverage")

        seen_done: set[str] = set()
        for _, line, event in proof_lines:
            if not event.startswith("setting_done:"):
                continue
            match = SETTING_DONE_RE.fullmatch(line)
            if match is None:
                errors.append("malformed setting completion")
                continue
            setting, generated, expected_count = match.groups()
            if setting not in EXPECTED or setting in seen_done:
                errors.append(f"unexpected or duplicate completion {setting!r}")
                continue
            seen_done.add(setting)
            rows = EXPECTED[setting]["rows"]
            if int(generated) != rows or int(expected_count) != rows:
                errors.append(f"incomplete generation for {setting}")
        if seen_done != set(SETTINGS):
            errors.append("incomplete setting completion coverage")

        eval_lines = [line for _, line, event in proof_lines if event == "eval_done"]
        if len(eval_lines) != 1 or EVAL_DONE_RE.fullmatch(eval_lines[0]) is None:
            errors.append("invalid eval completion")

        final_records = [(index, line) for index, line, event in proof_lines if event == "final"]
        if len(final_records) != 1:
            errors.append(f"final completion count={len(final_records)}")
        else:
            final_index, final_line = final_records[0]
            final = FINAL_RE.fullmatch(final_line)
            if final is None:
                errors.append("malformed final completion")
            else:
                try:
                    elapsed = float(final.group(1))
                except ValueError:
                    elapsed = math.nan
                if not math.isfinite(elapsed) or elapsed <= 0.0:
                    errors.append("invalid final elapsed time")
            if final_index != len(nonempty) - 1:
                errors.append("trailing output after final completion")

        if errors:
            return ParseResult(
                feedback="Rejected summarization verification: " + "; ".join(errors),
                metrics={},
            )

        lines = [
            f"[{setting}] ROUGE-L={metrics[f'rougeL_{setting}']:.6f} "
            f"n={EXPECTED[setting]['rows']}"
            for setting in SETTINGS
        ]
        return ParseResult(
            feedback="Full official summarization protocol completed:\n" + "\n".join(lines),
            metrics=metrics,
        )
