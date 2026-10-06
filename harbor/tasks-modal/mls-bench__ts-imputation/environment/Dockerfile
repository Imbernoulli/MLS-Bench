# Self-contained MLS-Bench task image.
#
# Stage 1 is the published bohanlyu2022/mlsbench-harbor-<pkg> image this
# task was verified on; it already carries the environment. The workspace
# and data stages below are still explicit so the agent-start tree is
# exactly tests/meta/pristine_manifest.json regardless of the image's age.
#
# Unlike the published 140 bundles this does NOT start from a
# bohanlyu2022/mlsbench-harbor-* image on Docker Hub: everything the task
# needs is described here, so the bundle is buildable by anyone and the
# research setup is not published as a pullable layer.
#
# The four stages mirror the workspace pipeline:
#   1. base image   2. environment   3. workspace   4./5. data
#
# _workspace/ is the agent-start state: vendored package source with the
# package-level pre_edit.py applied, overlaid by the task's mid_edit.py
# scaffold. It must hash-match tests/meta/pristine_manifest.json or the
# edit-range guard rejects every submission; scripts/harbor/selfcontained.py
# checks that before writing this file.
#
# The pristine baseline itself stays out of the image on purpose — the
# agent runs as root and could otherwise tamper with it before verification.

# 1. base image
FROM bohanlyu2022/mlsbench-harbor-time-series-library@sha256:3dbf272d06b2fa2f8b789e4becc9b4f1590fefab9251ff0fbac477b0e8f78592
# 2. environment
ENV PYTHONPATH="/workspace/Time-Series-Library:${PYTHONPATH}"
ENV HOME="/root"

# 3. workspace
RUN rm -rf /workspace && mkdir -p /workspace
COPY _workspace/ /workspace/
WORKDIR /workspace

# 5. data
# --- data deps declared by Time-Series-Library ---
# tslib_datasets -> /data
COPY _data_scripts/Time-Series-Library/vendor__pkg_configs__Time-Series-Library/ /tmp/prepare_Time-Series-Library__vendor__pkg_configs__Time-Series-Library/
RUN python3 -m pip install --no-cache-dir --target /tmp/prepdeps_Time-Series-Library__vendor__pkg_configs__Time-Series-Library 'huggingface_hub<1.0'
RUN PYTHONPATH=/tmp/prepdeps_Time-Series-Library__vendor__pkg_configs__Time-Series-Library:${PYTHONPATH} HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 HF_DATASETS_OFFLINE=0 python3 /tmp/prepare_Time-Series-Library__vendor__pkg_configs__Time-Series-Library/prepare_data.py --data-root /mlsb_data_root


# --- mlsbench: task-declared extra steps (scripts/harbor/extra_docker_steps.py) ---
RUN python3 -c "import hashlib, os, pathlib; spec = {\"ETT-small/ETTh1.csv\": [11520, \"f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066\", \"0ac0ce309049e14728c529a0fa98e682edecf8650331c5b977e5a8dc3636b3b0\"], \"weather/weather.csv\": [42157, \"34ee981d07313e51da2a50bb600072c8ae4a69cb4b0651f4cb93a069d7a2ba63\", \"5761af96e2c79ebce25e12ad2e942cdcf85c28c7eeed6bc5c884e67308ec45af\"], \"electricity/electricity.csv\": [21044, \"7e45845d54c5219bad0ae6bc1b5316cf8ff9cead5d33fa998a5a51c2e4a497ad\", \"2c9f85931626ab87ceb30ea28174907e7a2252444aa214f5444d51b5d7e261aa\"]}; exec('seen = set()\nfor rel, (k, full, vis) in spec.items():\n for root in (\"/data\", \"/mlsb_data_root\"):\n  p = pathlib.Path(root) / rel\n  if not p.is_file() or os.path.realpath(p) in seen:\n   continue\n  seen.add(os.path.realpath(p)); raw = p.read_bytes()\n  if hashlib.sha256(raw).hexdigest() == vis:\n   print(\"already truncated (hard link of a processed copy):\", p); continue\n  assert hashlib.sha256(raw).hexdigest() == full, \"unexpected source data: \" + str(p)\n  b = b\"\".join(raw.splitlines(keepends=True)[:k + 1])\n  assert hashlib.sha256(b).hexdigest() == vis, \"unexpected visible prefix: \" + str(p)\n  p.write_bytes(b)\n  (p.parent / \".mlsb_test_withheld\").write_text(\"The test rows of the scored CSV in this directory are withheld from the workspace and mounted only for the final evaluation.\")\n  print(\"withheld test rows:\", p, \"keeps\", k, \"data rows\")\nassert all(os.path.realpath(pathlib.Path(\"/data\") / r) in seen for r in spec), \"a scored CSV is missing under /data\"\n')"
RUN python3 -c "import os; names = {\"ETTh1.csv\": 11520, \"weather.csv\": 42157, \"electricity.csv\": 21044}; hits = [os.path.join(d, f) for d, _, fs in os.walk(\"/\") if not d.startswith((\"/proc\", \"/sys\", \"/dev\")) for f in fs if f in names]; bad = [h for h in hits if sum(1 for _ in open(h, \"rb\")) != names[os.path.basename(h)] + 1]; print(\"scored CSV copies:\", hits); assert hits and not bad, bad"
