#!/usr/bin/env python3
"""Measure whether selected MT research surfaces change real full-split outputs."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import time


SURFACES = {"postprocess", "early_stopping", "maxlen"}
IMAGE_PATTERN = re.compile(r".+@sha256:[0-9a-f]{64}")
WHITESPACE = re.compile(r"\s+")
PUNCTUATION = re.compile(r"[^\w\s]", flags=re.UNICODE)


def _load_common(path: Path):
    spec = importlib.util.spec_from_file_location("mt_surface_probe_common", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load MT common module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _prediction_digest(predictions: list[str]) -> str:
    payload = json.dumps(
        predictions, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _record(common, predictions: list[str], references: list[str], elapsed: float) -> dict:
    metrics = common.score_bleu_chrf(predictions, references)
    if not all(math.isfinite(value) for value in metrics.values()):
        raise RuntimeError("non-finite surface-probe metric")
    return {
        "prediction_sha256": _prediction_digest(predictions),
        "bleu": metrics["bleu"],
        "chrf": metrics["chrf"],
        "mean_words": common.mean_pred_len_words(predictions),
        "rows": len(predictions),
        "elapsed_seconds": elapsed,
    }


def _diff_count(left: list[str], right: list[str]) -> int:
    if len(left) != len(right):
        raise RuntimeError("cannot compare prediction lists of different lengths")
    return sum(a != b for a, b in zip(left, right))


def _translate(common, model, tokenizer, sources, device, kwargs):
    started = time.monotonic()
    predictions = common.translate(model, tokenizer, sources, kwargs, device)
    return predictions, time.monotonic() - started


def _postprocess_probe(common, model, tokenizer, sources, references, device) -> dict:
    raw, elapsed = _translate(
        common,
        model,
        tokenizer,
        sources,
        device,
        {
            "num_beams": 5,
            "length_penalty": 1.0,
            "early_stopping": True,
            "max_new_tokens": 128,
        },
    )
    variants = {
        "identity": raw,
        "normalize": [WHITESPACE.sub(" ", item).strip() for item in raw],
        "lowercase": [WHITESPACE.sub(" ", item).strip().lower() for item in raw],
        "strip_punct": [
            WHITESPACE.sub(" ", PUNCTUATION.sub(" ", item)).strip()
            for item in raw
        ],
    }
    return {
        "records": {
            name: _record(common, predictions, references, elapsed if name == "identity" else 0.0)
            for name, predictions in variants.items()
        },
        "pairwise_output_differences": {
            "identity_vs_normalize": _diff_count(raw, variants["normalize"]),
            "identity_vs_lowercase": _diff_count(raw, variants["lowercase"]),
            "identity_vs_strip_punct": _diff_count(raw, variants["strip_punct"]),
        },
    }


def _early_stopping_probe(
    common, model, tokenizer, sources, references, device
) -> dict:
    predictions = {}
    records = {}
    for label, value in (("true", True), ("false", False), ("never", "never")):
        output, elapsed = _translate(
            common,
            model,
            tokenizer,
            sources,
            device,
            {
                "num_beams": 5,
                "length_penalty": 0.6,
                "early_stopping": value,
                "max_new_tokens": 128,
            },
        )
        predictions[label] = output
        records[label] = _record(common, output, references, elapsed)
    return {
        "records": records,
        "pairwise_output_differences": {
            "true_vs_false": _diff_count(predictions["true"], predictions["false"]),
            "true_vs_never": _diff_count(predictions["true"], predictions["never"]),
            "false_vs_never": _diff_count(predictions["false"], predictions["never"]),
        },
    }


def _maxlen_probe(common, model, tokenizer, sources, references, device) -> dict:
    configurations = {
        "m10": {
            "num_beams": 5,
            "length_penalty": 1.0,
            "early_stopping": True,
            "max_new_tokens": 10,
        },
        "m32": {
            "num_beams": 5,
            "length_penalty": 1.0,
            "early_stopping": True,
            "max_new_tokens": 32,
        },
        "m128": {
            "num_beams": 5,
            "length_penalty": 1.0,
            "early_stopping": True,
            "max_new_tokens": 128,
        },
        "length_norm1_m128": {
            "num_beams": 5,
            "no_repeat_ngram_size": 0,
            "early_stopping": True,
            "length_penalty": 1.0,
            "min_length": 0,
            "max_new_tokens": 128,
        },
    }
    predictions = {}
    records = {}
    for label, kwargs in configurations.items():
        output, elapsed = _translate(
            common, model, tokenizer, sources, device, kwargs
        )
        predictions[label] = output
        records[label] = _record(common, output, references, elapsed)
    return {
        "records": records,
        "pairwise_output_differences": {
            "m10_vs_m32": _diff_count(predictions["m10"], predictions["m32"]),
            "m10_vs_m128": _diff_count(predictions["m10"], predictions["m128"]),
            "m32_vs_m128": _diff_count(predictions["m32"], predictions["m128"]),
            "m128_vs_length_norm1_m128": _diff_count(
                predictions["m128"], predictions["length_norm1_m128"]
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--surface", choices=sorted(SURFACES), required=True)
    parser.add_argument("--common", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-ref", required=True)
    args = parser.parse_args()
    if IMAGE_PATTERN.fullmatch(args.image_ref) is None:
        raise ValueError("image-ref must be pinned by digest")

    os.environ["MT_DIRECTION"] = "de_en"
    os.environ["MT_MODEL"] = "/data/machine-translation/models/opus-mt-de-en"
    os.environ["MT_DATA"] = "/data/machine-translation/data"
    common = _load_common(args.common)
    device = common.setup(common.SEED)
    sources, references, data_proof = common.load_dataset("de_en")
    model, tokenizer, model_proof = common.load_model_and_tokenizer(device, "de_en")

    if args.surface == "postprocess":
        surface_result = _postprocess_probe(
            common, model, tokenizer, sources, references, device
        )
    elif args.surface == "early_stopping":
        surface_result = _early_stopping_probe(
            common, model, tokenizer, sources, references, device
        )
    else:
        surface_result = _maxlen_probe(
            common, model, tokenizer, sources, references, device
        )
    result = {
        "schema_version": 1,
        "surface": args.surface,
        "image_ref": args.image_ref,
        "direction": "de_en",
        "rows": len(sources),
        "model_proof": model_proof,
        "data_proof": data_proof,
        **surface_result,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(args.output)
    print(
        f"MT_SURFACE_PROBE_COMPLETE surface={args.surface} "
        f"rows={len(sources)} output={args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
