"""Host-/verifier-side NAS-Bench-201 table provider for optimization-nas.

This module is NEVER bind-mounted into the agent container. It backs two
consumers that both run outside the agent's process:

  1. ``tasks/optimization-nas/edits/mid_edit.py`` — materializes ONLY the
     per-dataset VALIDATION table into the workspace (the budgeted signal the
     search is allowed to query).
  2. ``tasks/optimization-nas/parser.py`` — after the search prints
     ``FINAL_ARCH arch=<arch_str>``, looks up the held-out TEST accuracy here.

The TEST split therefore never exists in the agent's process or filesystem
(for cifar10, where 'cifar10-valid' != 'cifar10'; for cifar100 and
ImageNet16-120 the standard NAS-Bench-201 search protocol uses the same
'eval_acc1es' figure for both, so the budgeted validation signal is inherently
the benchmark metric there).

Table source resolution (dual layout, mirroring the other out-of-process
tasks):
  - Harbor: a ``nb201_tables.json.gz`` staged next to this file under the
    verifier-only ``tests/`` mount.
  - Native: the pristine NAS-Bench-201 pickle under
    ``vendor/external_packages/naslib/naslib/data/nb201_all.pickle``.
"""

import gzip
import json
import pickle
from pathlib import Path

_HERE = Path(__file__).resolve()

# ENV label used by the eval scripts -> NAS-Bench-201 dataset key.
DATASET_MAP = {
    "cifar10": "cifar10",
    "cifar100": "cifar100",
    "imagenet16": "ImageNet16-120",
}

# Dataset key -> split used for the budgeted VALIDATION query (must stay in
# lockstep with BenchmarkAPI.query_val_accuracy's historical behaviour).
_VAL_SPLIT = {
    "cifar10": "cifar10-valid",
    "cifar100": "cifar100",
    "ImageNet16-120": "ImageNet16-120",
}

_TABLES_CACHE = None


def _build_from_pickle(pickle_path: Path) -> dict:
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)
    tables = {}
    for env, dsk in DATASET_MAP.items():
        val_split = _VAL_SPLIT[dsk]
        tables[env] = {
            "val": {a: float(rec[val_split]["eval_acc1es"]) for a, rec in data.items()},
            "test": {a: float(rec[dsk]["eval_acc1es"]) for a, rec in data.items()},
        }
    return tables


def tables() -> dict:
    """{env: {"val": {arch_str: acc}, "test": {arch_str: acc}}} for all envs."""
    global _TABLES_CACHE
    if _TABLES_CACHE is not None:
        return _TABLES_CACHE

    staged = _HERE.parent / "nb201_tables.json.gz"
    if staged.exists():
        with gzip.open(staged, "rt") as f:
            _TABLES_CACHE = json.load(f)
        return _TABLES_CACHE

    project_root = _HERE.parents[2]
    pkl = (
        project_root
        / "vendor"
        / "external_packages"
        / "naslib"
        / "naslib"
        / "data"
        / "nb201_all.pickle"
    )
    _TABLES_CACHE = _build_from_pickle(pkl)
    return _TABLES_CACHE


def val_table(env: str) -> dict:
    """Validation-accuracy table for one env (the only table the agent's
    process ever receives)."""
    return tables()[env]["val"]


def test_accuracy(env: str, arch_str: str) -> float:
    """Held-out test accuracy for one architecture (host/verifier-side only).

    Raises KeyError for an architecture string not in the benchmark.
    """
    return tables()[env]["test"][arch_str]
