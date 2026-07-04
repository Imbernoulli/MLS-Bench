"""Host-/verifier-side held-out targets for llm-scaling-law-discovery.

Never bind-mounted into the agent container. The container's baked test
JSONLs carry FEATURES ONLY (prepare_data.py strips the target columns at
image build); the program emits its test predictions on stdout (SLD_PRED)
and the task parser joins them against the targets here — host-side
natively, verifier-side in Harbor (this file plus the ``*_test.jsonl``
siblings are staged under ``tests/meta/``, the same layout as ``holdout/``).

The sibling JSONLs are the pinned SLDBench test splits
(pkuHaowei/sldbench @ 721b846056f031737ff7fa72572c021324e3ec0e) with all
columns intact, so the target extraction below mirrors exactly what the
benchmark's original in-container loader computed.
"""

import json
from pathlib import Path

_HERE = Path(__file__).resolve().parent

# benchmark name -> (sibling truth file, target keys in pick() order)
BENCHMARKS = {
    "sld-vocab": ("sld_vocab_test.jsonl", ("unigram_normalized_loss", "loss")),
    "sld-lrbsz": ("sld_lrbsz_test.jsonl", ("lm_loss", "loss")),
    "sld-dataconstrained": ("sld_dataconstrained_test.jsonl", ("loss",)),
}


def _pick(row: dict, keys):
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return None


def truth(benchmark: str) -> list:
    """Held-out test targets for one benchmark, in file row order (the same
    order the in-container loader presents the test features)."""
    fname, keys = BENCHMARKS[benchmark]
    y = []
    with (_HERE / fname).open() as f:
        for line in f:
            line = line.strip()
            if line:
                y.append(float(_pick(json.loads(line), keys)))
    return y
