"""Fail-closed parser for the full official summarization protocol."""
from __future__ import annotations

import math
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mlsbench.agent.parsers import OutputParser, ParseResult


EXPECTED = {
    "xsum": {
        "rows": 11334,
        "data_sha256": "5b7854e6ee8b81493e38a5255ba794dec21735b30ba8f77280a394bf7d6dae53",
        "model": "distilbart-xsum-12-6",
        "revision": "5b2e376c845c201ddc34ec0e55fd1ad9890ba5ee",
        "weights_sha256": "6e9ebfc94e474225457570ad33c225cf66ca26279c5cd1cbfb67e089a03a791b",
    },
    "cnndm": {
        "rows": 11490,
        "data_sha256": "525e5bf29cb49a1ea2c58939d11435b6d6422f9a59d9c15e43d226c464b33278",
        "model": "distilbart-cnn-12-6",
        "revision": "a4f8f3ea906ed274767e9906dbaede7531d660ff",
        "weights_sha256": "3bac65d18c99463302d12ca75c2220ea714f9c81ce235f205fa818efe71df6ea",
    },
    "samsum": {
        "rows": 819,
        "data_sha256": "b9dd8165689b1c252b2db617b089b408ea068ef5bd6819a93c64e8ebde856b86",
        "model": "bart-large-cnn-samsum",
        "revision": "e49b3d60d923f12db22bdd363356f1a4c68532ad",
        "weights_sha256": "9f453aa6edef4dba1893723b7313b57b06b60214442d308a8acc3baa9583dd7b",
    },
}
TOTAL_DOCS = sum(int(item["rows"]) for item in EXPECTED.values())


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


class Parser(OutputParser):
    def parse(self, cmd_label: str, raw_output: str) -> ParseResult:
        errors: list[str] = []
        metrics: dict[str, float] = {}

        protocol_re = re.compile(
            r"SUMM_PROTOCOL version=summ-full-official-test-v1 "
            rf"settings=3 total_docs={TOTAL_DOCS}"
        )
        protocol, malformed = _unique_fullmatches(
            raw_output, "SUMM_PROTOCOL", protocol_re
        )
        if malformed or len(protocol) != 1:
            errors.append(
                f"protocol proof malformed={malformed} count={len(protocol)}"
            )

        data_re = re.compile(
            r"SUMM_DATA setting=(\S+) n_docs=(\d+) sha256=([0-9a-f]{64})"
        )
        data_matches, malformed = _unique_fullmatches(
            raw_output, "SUMM_DATA", data_re
        )
        if malformed:
            errors.append(f"malformed data proofs={malformed}")
        seen_data: set[str] = set()
        for match in data_matches:
            setting, count_text, digest = match.groups()
            if setting not in EXPECTED or setting in seen_data:
                errors.append(f"unexpected or duplicate data setting {setting!r}")
                continue
            seen_data.add(setting)
            expected = EXPECTED[setting]
            if int(count_text) != expected["rows"]:
                errors.append(f"wrong row count for {setting}")
            if digest != expected["data_sha256"]:
                errors.append(f"wrong data digest for {setting}")
        if seen_data != set(EXPECTED):
            errors.append(f"missing data proofs {sorted(set(EXPECTED) - seen_data)}")

        model_re = re.compile(
            r"SUMM_MODEL setting=(\S+) model=(\S+) revision=([0-9a-f]{40}) "
            r"params=(\d+) dtype=(\S+) weights_sha256=([0-9a-f]{64})"
        )
        model_matches, malformed = _unique_fullmatches(
            raw_output, "SUMM_MODEL", model_re
        )
        if malformed:
            errors.append(f"malformed model proofs={malformed}")
        seen_models: set[str] = set()
        for match in model_matches:
            setting, model, revision, params, dtype, digest = match.groups()
            if setting not in EXPECTED or setting in seen_models:
                errors.append(f"unexpected or duplicate model setting {setting!r}")
                continue
            seen_models.add(setting)
            expected = EXPECTED[setting]
            if model != expected["model"] or revision != expected["revision"]:
                errors.append(f"wrong model provenance for {setting}")
            if digest != expected["weights_sha256"]:
                errors.append(f"wrong model digest for {setting}")
            if int(params) < 100_000_000 or dtype != "float16":
                errors.append(f"wrong model scale or dtype for {setting}")

        source_policy = re.search(r"(?m)^SUMM_SOURCE policy=(\S+)$", raw_output)
        non_model_policy = (
            source_policy is not None
            and source_policy.group(1) in {"lead3", "copy_document", "first_token", "empty"}
        )
        if non_model_policy:
            if seen_models:
                errors.append("non-model policy emitted model proofs")
        elif seen_models != set(EXPECTED):
            errors.append(
                f"missing model proofs {sorted(set(EXPECTED) - seen_models)}"
            )

        metric_re = re.compile(
            r"SUMM_METRICS setting=(\S+) rougeL=(\S+) rouge1=(\S+) "
            r"rouge2=(\S+) plen=(\S+) n_docs=(\d+)"
        )
        metric_matches, malformed = _unique_fullmatches(
            raw_output, "SUMM_METRICS", metric_re
        )
        if malformed:
            errors.append(f"malformed metric lines={malformed}")
        seen_metrics: set[str] = set()
        for match in metric_matches:
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
            if (not all(math.isfinite(value) and 0.0 <= value <= 1.0
                        for value in values)
                    or not math.isfinite(plen) or plen < 0.0):
                errors.append(f"invalid metrics for {setting}")
                continue
            if int(count_text) != EXPECTED[setting]["rows"]:
                errors.append(f"wrong metric inventory for {setting}")
                continue
            metrics[f"rougeL_{setting}"] = values[0]
            metrics[f"rouge1_{setting}"] = values[1]
            metrics[f"rouge2_{setting}"] = values[2]
        if seen_metrics != set(EXPECTED):
            errors.append(
                f"missing metric settings {sorted(set(EXPECTED) - seen_metrics)}"
            )

        setting_done_re = re.compile(
            r"SUMM_SETTING_DONE setting=(\S+) generated=(\d+) expected=(\d+)"
        )
        done_matches, malformed = _unique_fullmatches(
            raw_output, "SUMM_SETTING_DONE", setting_done_re
        )
        if malformed:
            errors.append(f"malformed setting completion proofs={malformed}")
        seen_done: set[str] = set()
        for match in done_matches:
            setting, generated, expected_count = match.groups()
            if setting not in EXPECTED or setting in seen_done:
                errors.append(f"unexpected or duplicate completion {setting!r}")
                continue
            seen_done.add(setting)
            rows = EXPECTED[setting]["rows"]
            if int(generated) != rows or int(expected_count) != rows:
                errors.append(f"incomplete generation for {setting}")
        if seen_done != set(EXPECTED):
            errors.append(
                f"missing setting completion {sorted(set(EXPECTED) - seen_done)}"
            )

        eval_done_re = re.compile(
            rf"SUMM_EVAL_DONE settings=3 total_docs={TOTAL_DOCS}"
        )
        eval_done, malformed = _unique_fullmatches(
            raw_output, "SUMM_EVAL_DONE", eval_done_re
        )
        if malformed or len(eval_done) != 1:
            errors.append(
                f"eval completion malformed={malformed} count={len(eval_done)}"
            )

        final_re = re.compile(
            rf"SUMM_DONE settings=3 total_docs={TOTAL_DOCS} seed=42 elapsed=(\S+)"
        )
        final, malformed = _unique_fullmatches(raw_output, "SUMM_DONE", final_re)
        if malformed or len(final) != 1:
            errors.append(
                f"final completion malformed={malformed} count={len(final)}"
            )
        else:
            try:
                elapsed = float(final[0].group(1))
            except ValueError:
                elapsed = math.nan
            if not math.isfinite(elapsed) or elapsed <= 0:
                errors.append("invalid final elapsed time")

        if errors:
            return ParseResult(
                feedback="Rejected incomplete summarization verification: "
                + "; ".join(errors),
                metrics={},
            )

        lines = [
            f"[{setting}] ROUGE-L={metrics[f'rougeL_{setting}']:.6f} "
            f"n={EXPECTED[setting]['rows']}"
            for setting in EXPECTED
        ]
        return ParseResult(
            feedback="Full official summarization protocol completed:\n"
            + "\n".join(lines),
            metrics=metrics,
        )
