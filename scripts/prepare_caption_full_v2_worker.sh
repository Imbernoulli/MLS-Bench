#!/usr/bin/env bash
set -euo pipefail

stage=${CAPTION_STAGE:-/mnt/moonfs/lvbohan-b0/image-captioning-full-v1}
repo="${stage}/repo"
data_root="${stage}/data_root-v2"
run_id="${CAPTION_PREP_RUN_ID:-official-v2-streaming-canonical}"
if [[ ! "${run_id}" =~ ^[a-zA-Z0-9._-]+$ ]]; then
    echo "invalid CAPTION_PREP_RUN_ID: ${run_id}" >&2
    exit 2
fi
run="${stage}/data-prep/${run_id}"

mkdir -p "${stage}/data-prep"
if ! mkdir "${run}"; then
    echo "refusing to reuse caption data-prep output: ${run}" >&2
    exit 2
fi
exec >> "${run}/worker.log" 2>&1

finish() {
    rc=$?
    trap - EXIT HUP INT TERM
    if [[ ${rc} -eq 0 ]]; then
        required_outputs=(
            "${run}/data-proof.json"
            "${data_root}/image-captioning/source_manifest.json"
            "${data_root}/image-captioning/train_clip.pt"
            "${data_root}/image-captioning/train_refs.json"
            "${data_root}/image-captioning/eval_clip.pt"
            "${data_root}/image-captioning/eval_refs.json"
        )
        for output in "${required_outputs[@]}"; do
            if [[ ! -s "${output}" ]]; then
                echo "caption data-prep success gate is missing: ${output}" >&2
                rc=70
                break
            fi
        done
    fi
    printf '%s\n' "${rc}" > "${run}/rc"
    if [[ ${rc} -eq 0 ]]; then
        printf 'success\n' > "${run}/status"
        date -Iseconds > "${run}/SUCCESS"
    else
        printf 'failed\n' > "${run}/status"
    fi
    exit "${rc}"
}
trap finish EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

date -Iseconds > "${run}/STARTED"
printf 'running\n' > "${run}/status"
echo "CAPTION_DATA_PREP_START host=$(hostname) date=$(date -Iseconds)"
(cd "${repo}" && sha256sum -c "${stage}/repo.sha256") \
    > "${run}/repo.sha256.log"
echo "CAPTION_DATA_PREP_STAGE_OK files=$(wc -l < "${stage}/repo.sha256")"

unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export HF_HOME="${stage}/hf-cache-v2"
export HUGGINGFACE_HUB_CACHE="${stage}/hf-cache-v2/hub"
export HF_DATASETS_CACHE="${stage}/hf-cache-v2/datasets"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export FLICKR8K_CANONICAL_ARCHIVE="${stage}/canonical/caption_datasets.zip"
export CAPTION_PREP_GPT2_SEED="${data_root}/image-captioning.staging-1524168/gpt2"

python - <<'PY'
import importlib.metadata as metadata
import os
import torch

observed_open_clip = metadata.version("open_clip_torch")
observed_transformers = metadata.version("transformers")
if observed_open_clip != "3.3.0":
    raise SystemExit(
        f"caption data-prep package mismatch: open_clip_torch={observed_open_clip}"
    )
allow_cpu = os.environ.get("CAPTION_PREP_ALLOW_CPU") == "1"
if allow_cpu:
    device = "cpu-host-fallback"
else:
    if observed_transformers != "4.53.2":
        raise SystemExit(
            f"caption data-prep package mismatch: transformers={observed_transformers}"
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise SystemExit("caption data preparation requires exactly one visible GPU")
    device = torch.cuda.get_device_name(0)
print(
    f"CAPTION_DATA_PREP_RUNTIME torch={torch.__version__} cuda={torch.version.cuda}"
    f" device={device} open_clip={observed_open_clip}"
    f" transformers={observed_transformers}",
    flush=True,
)
PY

exec 9>"${stage}/data-prepare.lock"
flock 9
echo "CAPTION_DATA_PREP_LOCK_ACQUIRED run_id=${run_id} date=$(date -Iseconds)"
python "${repo}/vendor/data_scripts/image-captioning/prepare_data.py" \
    --data-root "${data_root}"
flock -u 9

python - "${stage}" "${run}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

import torch

root = Path(sys.argv[1])
data = root / "data_root-v2/image-captioning"
run = Path(sys.argv[2])
if run.parent != root / "data-prep" or not run.is_dir():
    raise RuntimeError(f"invalid caption data-prep proof directory: {run}")

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

manifest = json.loads((data / "source_manifest.json").read_text())
train = torch.load(data / "train_clip.pt", map_location="cpu", weights_only=True)
test = torch.load(data / "eval_clip.pt", map_location="cpu", weights_only=True)
train_refs = json.loads((data / "train_refs.json").read_text())
test_refs = json.loads((data / "eval_refs.json").read_text())
assert manifest["schema_version"] == 3
assert tuple(train.shape) == (6000, 512) and train.dtype == torch.float32
assert tuple(test.shape) == (1000, 512) and test.dtype == torch.float32
assert torch.isfinite(train).all() and torch.isfinite(test).all()
assert len(train_refs) == 6000 and all(len(refs) == 5 for refs in train_refs)
assert len(test_refs) == 1000 and all(len(refs) == 5 for refs in test_refs)
assert manifest["canonical_filename_set_sha256"] == {
    "train": "fbb334d8b4d4bab05a65950cb0b8123079c40ba8d1c38d8aa360fa27459e8cf4",
    "test": "25d2fec0836bb4728d4672c46a5694dfbdb953a2ff5ba146f5ffaa7062512489",
}
paths = [
    data / "source_manifest.json",
    data / "train_clip.pt",
    data / "train_refs.json",
    data / "eval_clip.pt",
    data / "eval_refs.json",
]
proof = {
    "manifest_schema_version": manifest["schema_version"],
    "train_images": len(train_refs),
    "train_pairs": sum(len(refs) for refs in train_refs),
    "eval_images": len(test_refs),
    "references_per_image": 5,
    "train_clip_shape": list(train.shape),
    "eval_clip_shape": list(test.shape),
    "split_sha256": manifest["split_sha256"],
    "canonical_filename_set_sha256": manifest["canonical_filename_set_sha256"],
    "artifact_sha256": {path.name: sha256(path) for path in paths},
}
(run / "data-proof.json").write_text(json.dumps(proof, indent=2, sort_keys=True) + "\n")
print("CAPTION_DATA_PREP_PROOF " + json.dumps(proof, sort_keys=True), flush=True)
PY

echo "CAPTION_DATA_PREP_DONE date=$(date -Iseconds)"
