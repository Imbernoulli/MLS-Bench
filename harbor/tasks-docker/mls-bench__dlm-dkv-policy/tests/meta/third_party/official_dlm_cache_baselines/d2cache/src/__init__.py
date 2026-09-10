"""Task-local import shim for the d2Cache LLaDA/cache subset.

The upstream package-level ``src.__init__`` initializes Hydra/lm-eval helpers.
This MLS-Bench task only imports the cache/model/generation modules, so the shim
keeps the official algorithm files importable without pulling evaluation-only
dependencies into the benchmark runtime.
"""
