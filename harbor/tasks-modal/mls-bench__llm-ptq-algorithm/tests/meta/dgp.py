"""Verifier-only data for llm-ptq-algorithm (no generator lives here).

The adapter stages holdout/<task>/ into the bundle's tests/meta/ only when this
file exists, and tests/ is mounted at verification time and never during the
agent's session. This directory carries the WikiText-2 (raw) test split the
perplexity is scored on -- data/withheld/wikitext-test.arrow, byte-identical to
the HF datasets cache file in
bohanlyu2022/mlsbench-harbor-gptq@sha256:8beac152e3677bbdd3002ff7f3a7a5126d6844b21a8de8e5fc583a228712e1b1
(/data/wikitext2/wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3/)
-- and its sha256. The image ships without it (config.json
harbor_extra_docker_steps); scripts/_withheld.sh rebuilds the cache for the
eval and refuses to run if the restored file does not hash to the original.
"""
