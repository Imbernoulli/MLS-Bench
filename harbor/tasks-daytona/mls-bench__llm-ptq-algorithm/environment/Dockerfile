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
FROM bohanlyu2022/mlsbench-harbor-gptq@sha256:8beac152e3677bbdd3002ff7f3a7a5126d6844b21a8de8e5fc583a228712e1b1
# 2. environment
ENV HF_HOME="/data/huggingface"
ENV TRANSFORMERS_CACHE="/data/huggingface/transformers"
ENV HF_DATASETS_CACHE="/data/wikitext2"
ENV PYTHONPATH="/workspace/gptq:${PYTHONPATH}"

# 3. workspace
RUN rm -rf /workspace && mkdir -p /workspace
COPY _workspace/ /workspace/
WORKDIR /workspace

# 5. data
# --- data deps declared by gptq ---
# mistral-7b-v01 -> /data/mistral-7b-v01
COPY _data_scripts/gptq/vendor__data_scripts__gptq/ /tmp/prepare_gptq__vendor__data_scripts__gptq/
RUN python3 -m pip install --no-cache-dir --target /tmp/prepdeps_gptq__vendor__data_scripts__gptq huggingface_hub==0.26.2 datasets==3.1.0
RUN PYTHONPATH=/tmp/prepdeps_gptq__vendor__data_scripts__gptq:${PYTHONPATH} HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 HF_DATASETS_OFFLINE=0 python3 /tmp/prepare_gptq__vendor__data_scripts__gptq/prepare_model.py --data-root /data
# wikitext2 -> /data/wikitext2
RUN PYTHONPATH=/tmp/prepdeps_gptq__vendor__data_scripts__gptq:${PYTHONPATH} HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 HF_DATASETS_OFFLINE=0 python3 /tmp/prepare_gptq__vendor__data_scripts__gptq/prepare_wikitext2.py --data-root /data


# --- mlsbench: task-declared extra steps (scripts/harbor/extra_docker_steps.py) ---
RUN python3 -c "import hashlib, json, pathlib; t = pathlib.Path(\"/data/wikitext2/wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-test.arrow\"); assert hashlib.sha256(t.read_bytes()).hexdigest() == \"2b8a3efac7b468cbe6432edba5f55c21e435d93873acc6727431f08d5ed328ea\", \"unexpected wikitext-test.arrow\"; q = pathlib.Path(\"/data/wikitext2/downloads/5e548f01091b2f01906ea7be35b8a1a6aa74842629dc550b7a02ca0821f1002d\"); assert \"test-00000-of-00001.parquet\" in json.loads(pathlib.Path(str(q) + \".json\").read_text())[\"url\"]; [p.unlink() for p in (t, q, pathlib.Path(str(q) + \".json\"))]; pathlib.Path(\"/data/wikitext2/.mlsb_test_withheld\").write_text(\"The WikiText-2 test split is withheld from the workspace and mounted only for the final evaluation.\"); print(\"withheld: WikiText-2 test split\")"
RUN test -z "$(find / -xdev \( -name 'wikitext-test*' -o -name 'test-00000-of-00001.parquet' \) -not -path '/proc/*' 2>/dev/null)" && ! grep -rlq --include='*.json' 'wikitext-2-raw-v1/test' /data/wikitext2/downloads
