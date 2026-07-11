"""Strict parser binding for compress-low-rate-policy."""
from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_shared():
    candidates = (
        Path("/workspace/compressai/policy_parser.py"),
        Path(__file__).resolve().parents[2] / "vendor" / "compressai" / "policy_parser.py",
    )
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        raise ImportError("strict CompressAI policy parser is unavailable")
    spec = importlib.util.spec_from_file_location("compressai_policy_parser", path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise ImportError("strict CompressAI policy parser loader is unavailable")
    spec.loader.exec_module(module)
    return module


Parser = _load_shared().make_parser('quality', 'low_rate_policy')
