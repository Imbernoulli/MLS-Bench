"""Strict parser wrapper for the authenticated full-image ood-input-preproc protocol."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

TASK = "ood-input-preproc"
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

_candidates = (
    Path("/workspace/ood-detection-lab/parser_contract.py"),
    PROJECT_ROOT / "vendor/ood-detection-lab/parser_contract.py",
)
_contract_path = next((path for path in _candidates if path.is_file()), None)
if _contract_path is None:
    raise FileNotFoundError("verifier-only OOD parser contract is unavailable")
_spec = importlib.util.spec_from_file_location("ood_full_parser_contract", _contract_path)
if _spec is None or _spec.loader is None:
    raise RuntimeError("cannot import the verifier-only OOD parser contract")
_contract = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_contract)

PROTOCOL = _contract.PROTOCOL
EXPECTED_DATA_SHA256 = _contract.EXPECTED_DATA_SHA256
EXPECTED_MODEL_SHA256 = _contract.EXPECTED_MODEL_SHA256
Parser = _contract.build_parser(TASK)
