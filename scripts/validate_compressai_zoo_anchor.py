#!/usr/bin/env python3
"""Strictly replay a completed CompressAI anchor proof inside a worker."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
import types
from dataclasses import dataclass
from pathlib import Path


def install_parser_types() -> None:
    try:
        from mlsbench.agent.parsers import OutputParser, ParseResult  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    class OutputParser:
        pass

    @dataclass
    class ParseResult:
        feedback: str
        metrics: dict

    root = types.ModuleType("mlsbench")
    agent = types.ModuleType("mlsbench.agent")
    parsers = types.ModuleType("mlsbench.agent.parsers")
    parsers.OutputParser = OutputParser
    parsers.ParseResult = ParseResult
    root.agent = agent
    agent.parsers = parsers
    sys.modules.update(
        {
            "mlsbench": root,
            "mlsbench.agent": agent,
            "mlsbench.agent.parsers": parsers,
        }
    )


def load_module(path: Path):
    install_parser_types()
    spec = importlib.util.spec_from_file_location("compress_anchor_parser", path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError("parser loader is unavailable")
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parser", type=Path, required=True)
    parser.add_argument("--proof", type=Path, required=True)
    parser.add_argument("--family", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    raw = args.proof.read_text()
    module = load_module(args.parser)
    result = module.Parser().parse("kodak24_q1q8", raw)
    expected_keys = {
        f"{metric}_{setting}"
        for setting in module.SETTINGS
        for metric in ("mean_rd_utility", "psnr", "bpp")
    }
    if set(result.metrics) != expected_keys:
        raise SystemExit("strict parser did not return the complete metric inventory")
    if any(not math.isfinite(float(value)) for value in result.metrics.values()):
        raise SystemExit("strict parser returned a non-finite metric")
    final_lines = [line for line in raw.splitlines() if line.startswith("COMPRESS_FINAL ")]
    if len(final_lines) != 1 or f"family={args.family}" not in final_lines[0]:
        raise SystemExit("terminal family proof mismatch")

    invalid_proofs = {
        "empty": "",
        "missing_final": "\n".join(raw.splitlines()[:-1]),
        "trailing": raw + "\nunrecognized trailing output\n",
        "failure_marker": "COMPRESS_FAILED injected\n" + raw,
        "nonfinite": raw.replace(" bpp=", " bpp=nan replaced_bpp=", 1),
        "duplicate_final": raw + final_lines[0] + "\n",
    }
    rejected = {}
    for name, candidate in invalid_proofs.items():
        empty = module.Parser().parse("kodak24_q1q8", candidate).metrics == {}
        rejected[name] = empty
        if not empty:
            raise SystemExit(f"strict parser accepted invalid proof: {name}")

    payload = {
        "status": "success",
        "family": args.family,
        "proof_sha256": hashlib.sha256(raw.encode()).hexdigest(),
        "parser_sha256": hashlib.sha256(args.parser.read_bytes()).hexdigest(),
        "metric_count": len(result.metrics),
        "metrics": result.metrics,
        "invalid_proofs_rejected": rejected,
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        f"COMPRESS_STRICT_REPLAY family={args.family} metrics={len(result.metrics)} "
        f"proof_sha256={payload['proof_sha256']} invalid_rejected={len(rejected)}"
    )


if __name__ == "__main__":
    main()
