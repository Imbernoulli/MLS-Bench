"""Strict terminal-proof parser for one machine-translation sibling."""
from __future__ import annotations

import math
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mlsbench.agent.parsers import OutputParser, ParseResult


EXPECTED_TASK = "mt-early-stopping"
EXPECTED_SURFACE = "build_early_stopping"
PROTOCOL_VERSION = "mt-opus100-provenance-v2"
DATASET = "Helsinki-NLP/opus-100"
DATASET_REVISION = "805090dc28bf78897da9641cdf08b61287580df9"
SOURCE_MANIFEST_SHA256 = "05d87a9da44f2bb3dcf514cdddd595639667105091247744fbc7e818f9f9e924"
EXPECTED_ROWS = 2000
EXPECTED_MODELS = {
    "de_en": {
        "repository": "Helsinki-NLP/opus-mt-de-en",
        "revision": "1a922f3b32a8e809e17a47d4b32142d8105924e5",
        "manifest": "008cc43d43dbab7913e920a9681f6939bca3fd4718df0bbd4b9b7cbf5712e430",
        "tokenizer": "db30174fffd23d5ab12dcadfd4e88ec87ddf97147680faa50969405933b3cb07",
        "checkpoint": "e743c3070f61f477cb62fe95ef2c9be2e77f3e488cb6b8030ff8a19e8295c87d",
        "parameters": 74_410_496,
        "split": "2e7a80586d269952371ff5e71f8840e26926416c399051e2371a3b14a1b0b6dc",
    },
    "fr_en": {
        "repository": "Helsinki-NLP/opus-mt-fr-en",
        "revision": "c4aed37b318c763fd177aa449b44e3b783cc6c02",
        "manifest": "ec5530ddd718e210e3877a7a7c170b32088aa5b354e98a174102141a88ee0b0d",
        "tokenizer": "a8ba4784fed48ffe588912400ccf4fed9db88cf5e6417afc2fa892416a1942eb",
        "checkpoint": "6e3837f34b903802c3d0d670362b997cee6e87584a1108eb3fa89e4625e4424a",
        "parameters": 75_133_952,
        "split": "09477f8a19e67d3f7c09c320d076f5a32168ab4cde55d8f1d88ffb66c02f68a1",
    },
    "ru_en": {
        "repository": "Helsinki-NLP/opus-mt-ru-en",
        "revision": "fbd6dc73284f95536648512cc21d57f19191961a",
        "manifest": "7d1af2e235e44ec595ded0977933ec103384184efb83a410769295e26315a0c9",
        "tokenizer": "883dbad1f02b3a4abb3c28088bf5c0a02a6534244e93882bebace69f7e45eb06",
        "checkpoint": "535450eb5613f3cc912f9ca3e54cfef6c14d201b319c24a88faf776a65538b5d",
        "parameters": 76_672_000,
        "split": "c072e931f99d1ed04829ec4f63b18c34ebc9ead4b1b19b25fceec60720650eb0",
    },
}

_TOKEN = r"[A-Za-z0-9_./-]+"
_HEX = r"[0-9a-f]{64}"
_NUMBER = r"(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?"
_PROTOCOL_RE = re.compile(
    rf"MT_PROTOCOL version=(?P<version>{_TOKEN}) task=(?P<task>{_TOKEN}) "
    rf"surface=(?P<surface>{_TOKEN}) direction=(?P<direction>{_TOKEN}) "
    r"seed=(?P<seed>[0-9]+)"
)
_MODEL_RE = re.compile(
    rf"MT_MODEL direction=(?P<direction>{_TOKEN}) "
    rf"repository=(?P<repository>{_TOKEN}) revision=(?P<revision>[0-9a-f]{{40}}) "
    rf"manifest_sha256=(?P<manifest>{_HEX}) "
    rf"tokenizer_manifest_sha256=(?P<tokenizer>{_HEX}) "
    rf"checkpoint_sha256=(?P<checkpoint>{_HEX}) "
    r"parameters=(?P<parameters>[0-9]+)"
)
_DATA_RE = re.compile(
    rf"MT_DATA direction=(?P<direction>{_TOKEN}) dataset=(?P<dataset>{_TOKEN}) "
    rf"revision=(?P<revision>[0-9a-f]{{40}}) "
    rf"manifest_sha256=(?P<manifest>{_HEX}) split_sha256=(?P<split>{_HEX}) "
    r"rows=(?P<rows>[0-9]+)"
)
_METRICS_RE = re.compile(
    rf"MT_METRICS task=(?P<task>{_TOKEN}) surface=(?P<surface>{_TOKEN}) "
    rf"direction=(?P<direction>{_TOKEN}) bleu=(?P<bleu>{_NUMBER}) "
    rf"chrf=(?P<chrf>{_NUMBER}) n_pairs=(?P<rows>[0-9]+) "
    rf"plen=(?P<plen>{_NUMBER}) elapsed=(?P<elapsed>{_NUMBER})"
)
_COMPLETE_RE = re.compile(
    rf"MT_COMPLETE task=(?P<task>{_TOKEN}) surface=(?P<surface>{_TOKEN}) "
    rf"direction=(?P<direction>{_TOKEN}) status=(?P<status>{_TOKEN})"
)


def _rejected(reason: str) -> ParseResult:
    return ParseResult(
        feedback=f"Rejected machine-translation verification: {reason}.",
        metrics={},
    )


class Parser(OutputParser):
    def parse(self, cmd_label: str, raw_output: str) -> ParseResult:
        if cmd_label not in EXPECTED_MODELS:
            return _rejected("unknown direction label")
        lines = raw_output.splitlines()
        if len(lines) != 5 or any(not line or line != line.strip() for line in lines):
            return _rejected("expected exactly five canonical terminal records")

        protocol = _PROTOCOL_RE.fullmatch(lines[0])
        model = _MODEL_RE.fullmatch(lines[1])
        data = _DATA_RE.fullmatch(lines[2])
        metrics = _METRICS_RE.fullmatch(lines[3])
        complete = _COMPLETE_RE.fullmatch(lines[4])
        if not all((protocol, model, data, metrics, complete)):
            return _rejected("malformed or out-of-order proof record")

        expected = EXPECTED_MODELS[cmd_label]
        if (
            protocol["version"] != PROTOCOL_VERSION
            or protocol["task"] != EXPECTED_TASK
            or protocol["surface"] != EXPECTED_SURFACE
            or protocol["direction"] != cmd_label
            or int(protocol["seed"]) != 42
        ):
            return _rejected("protocol identity mismatch")
        if (
            model["direction"] != cmd_label
            or model["repository"] != expected["repository"]
            or model["revision"] != expected["revision"]
            or model["manifest"] != expected["manifest"]
            or model["tokenizer"] != expected["tokenizer"]
            or model["checkpoint"] != expected["checkpoint"]
            or int(model["parameters"]) != expected["parameters"]
        ):
            return _rejected("model or tokenizer provenance mismatch")
        if (
            data["direction"] != cmd_label
            or data["dataset"] != DATASET
            or data["revision"] != DATASET_REVISION
            or data["manifest"] != SOURCE_MANIFEST_SHA256
            or data["split"] != expected["split"]
            or int(data["rows"]) != EXPECTED_ROWS
        ):
            return _rejected("dataset provenance mismatch")
        if (
            metrics["task"] != EXPECTED_TASK
            or metrics["surface"] != EXPECTED_SURFACE
            or metrics["direction"] != cmd_label
            or int(metrics["rows"]) != EXPECTED_ROWS
            or complete["task"] != EXPECTED_TASK
            or complete["surface"] != EXPECTED_SURFACE
            or complete["direction"] != cmd_label
            or complete["status"] != "ok"
        ):
            return _rejected("metric or completion identity mismatch")

        bleu = float(metrics["bleu"])
        chrf = float(metrics["chrf"])
        pred_len = float(metrics["plen"])
        elapsed = float(metrics["elapsed"])
        if (
            not all(math.isfinite(value) for value in (bleu, chrf, pred_len, elapsed))
            or not 0.0 <= bleu <= 100.0
            or not 0.0 <= chrf <= 100.0
            or pred_len < 0.0
            or elapsed <= 0.0
        ):
            return _rejected("non-finite or out-of-range metric")

        return ParseResult(
            feedback=(
                f"Verified {EXPECTED_TASK}/{cmd_label}: sacreBLEU={bleu:.6f}, "
                f"chrF={chrf:.6f}, pairs={EXPECTED_ROWS}, elapsed={elapsed:.6f}s"
            ),
            metrics={
                f"bleu_{cmd_label}": bleu,
                f"chrf_{cmd_label}": chrf,
            },
        )
