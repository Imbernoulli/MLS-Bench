# Official DLM Cache Baseline Sources

This directory contains task-local source snapshots used by `dlm-dkv-policy`
to run paper-backed baselines without relying on machine-local clone paths.

Sources:

- `d2cache/`: selected LLaDA cache/model/generation files from
  `https://github.com/Kamichanw/d2Cache`, commit
  `72feb60bb288a5a4d14b4bab9afbbaa1fc1b509a`, Apache-2.0.
- `elastic_cache/`: selected LLaDA generation/model files from
  `https://github.com/VILA-Lab/Elastic-Cache`, commit
  `8c7beabfff847a7ff4252562309680b66c820d67`, Apache-2.0.

Compatibility changes are limited to:

- replacing dependency-heavy package initializers with minimal import shims;
- pruning unused non-LLaDA imports from the d2Cache generation utility path;
- routing d2Cache's requested attention-weight path through eager attention, as
  required by the official d2Cache cache implementation.

The cache/model/generation algorithms exercised by the task remain
source-backed by the official implementations above.
