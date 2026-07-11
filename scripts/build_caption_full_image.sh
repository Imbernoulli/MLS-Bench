#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
STAGE=${CAPTION_STAGE:-/mnt/moonfs/lvbohan-b0/image-captioning-full-v1}
DATA_ROOT="${STAGE}/data_root-v2/image-captioning"
STAGED_REPO="${STAGE}/repo"
CRANE=${CRANE:-/launchpad/crane}
RUNTIME_IMAGE=${CAPTION_RUNTIME_IMAGE:-msai-cn-beijing.cr.volces.com/public/bohanlyu2022/mlsbench-harbor-image-captioning-full@sha256:737e3b315476b44d5f538375087382729f8fee9b796477f427715bdbf31d8153}
IMAGE_REPO=${CAPTION_IMAGE_REPO:-msai-cn-beijing.cr.volces.com/public/bohanlyu2022/mlsbench-harbor-image-captioning-full}
RUN_ID=${1:-"full-v3-$(date -u +%Y%m%dT%H%M%SZ)"}
BUILD="${STAGE}/image-builds/${RUN_ID}"
IMAGE="${IMAGE_REPO}:${RUN_ID}"
CLEAN_IMAGE="${IMAGE_REPO}:${RUN_ID}-clean"
LAYER_IMAGE="${IMAGE_REPO}:${RUN_ID}-layer"

if [[ ! "${RUN_ID}" =~ ^[a-zA-Z0-9._-]+$ ]]; then
    echo "invalid caption image RUN_ID: ${RUN_ID}" >&2
    exit 2
fi
if [[ -e "${BUILD}" ]]; then
    echo "refusing to reuse caption image build: ${BUILD}" >&2
    exit 2
fi
mkdir -p "${STAGE}/image-builds"
mkdir "${BUILD}"
exec >> "${BUILD}/build.log" 2>&1

finish() {
    rc=$?
    trap - EXIT HUP INT TERM
    if [[ ${rc} -eq 0 ]]; then
        for output in \
                "${BUILD}/image.ref" \
                "${BUILD}/image.digest" \
                "${BUILD}/IMAGE_READY"; do
            if [[ ! -s "${output}" ]]; then
                echo "caption image success gate is missing: ${output}" >&2
                rc=70
                break
            fi
        done
    fi
    printf '%s\n' "${rc}" > "${BUILD}/rc"
    if [[ ${rc} -eq 0 ]]; then
        printf 'success\n' > "${BUILD}/status"
        date -Iseconds > "${BUILD}/SUCCESS"
    else
        printf 'failed\n' > "${BUILD}/status"
    fi
    exit "${rc}"
}
trap finish EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM
date -Iseconds > "${BUILD}/STARTED"
printf 'running\n' > "${BUILD}/status"

required=(
    "${DATA_ROOT}/source_manifest.json"
    "${DATA_ROOT}/train_clip.pt"
    "${DATA_ROOT}/train_refs.json"
    "${DATA_ROOT}/eval_clip.pt"
    "${DATA_ROOT}/eval_refs.json"
    "${DATA_ROOT}/gpt2/config.json"
    "${DATA_ROOT}/gpt2/model.safetensors"
    "${STAGE}/repo.sha256"
)
for path in "${required[@]}"; do
    if [[ ! -s "${path}" ]]; then
        echo "cannot build caption image; staged input is missing: ${path}" >&2
        exit 3
    fi
done

(cd "${STAGED_REPO}" && sha256sum -c "${STAGE}/repo.sha256") \
    > "${BUILD}/repo.sha256.log"

python - "${DATA_ROOT}" "${BUILD}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

import torch

data = Path(sys.argv[1])
build = Path(sys.argv[2])

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

manifest_path = data / "source_manifest.json"
manifest = json.loads(manifest_path.read_text())
train = torch.load(data / "train_clip.pt", map_location="cpu", weights_only=True)
test = torch.load(data / "eval_clip.pt", map_location="cpu", weights_only=True)
train_refs = json.loads((data / "train_refs.json").read_text())
test_refs = json.loads((data / "eval_refs.json").read_text())

assert manifest["schema_version"] == 3
assert manifest["protocol"] == "flickr8k_official_v1"
assert manifest["train_count"] == 6000 and manifest["eval_count"] == 1000
assert manifest["references_per_image"] == 5
assert tuple(train.shape) == (6000, 512) and train.dtype == torch.float32
assert tuple(test.shape) == (1000, 512) and test.dtype == torch.float32
assert bool(torch.isfinite(train).all()) and bool(torch.isfinite(test).all())
assert len(train_refs) == 6000 and all(len(row) == 5 for row in train_refs)
assert len(test_refs) == 1000 and all(len(row) == 5 for row in test_refs)
assert sum(map(len, train_refs)) == 30000
for name, expected in manifest["prepared_sha256"].items():
    path = data / name
    assert path.is_file(), name
    assert sha256(path) == expected, name

proof = {
    "manifest_sha256": sha256(manifest_path),
    "split_sha256": manifest["split_sha256"],
    "canonical_filename_set_sha256": manifest["canonical_filename_set_sha256"],
    "train_images": 6000,
    "train_pairs": 30000,
    "eval_images": 1000,
    "references_per_image": 5,
    "train_clip_shape": list(train.shape),
    "eval_clip_shape": list(test.shape),
}
(build / "input-proof.json").write_text(
    json.dumps(proof, indent=2, sort_keys=True) + "\n"
)
print("CAPTION_IMAGE_INPUT_PROOF " + json.dumps(proof, sort_keys=True), flush=True)
PY

manifest_sha=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["manifest_sha256"])' "${BUILD}/input-proof.json")
train_split_sha=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["split_sha256"]["train"])' "${BUILD}/input-proof.json")
test_split_sha=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["split_sha256"]["test"])' "${BUILD}/input-proof.json")

whiteout="${BUILD}/whiteout-layer"
content="${BUILD}/content-layer"
mkdir -p \
    "${whiteout}/data/image-captioning" \
    "${whiteout}/workspace/image-captioning" \
    "${whiteout}/opt/mlsbench-caption/repo" \
    "${content}/data/image-captioning" \
    "${content}/workspace/image-captioning" \
    "${content}/opt/mlsbench-caption/repo"
: > "${whiteout}/data/image-captioning/.wh..wh..opq"
: > "${whiteout}/workspace/image-captioning/.wh..wh..opq"
: > "${whiteout}/opt/mlsbench-caption/repo/.wh..wh..opq"

tar -C "${DATA_ROOT}" -cf - . \
    | tar -C "${content}/data/image-captioning" -xf -
tar --exclude='__pycache__' --exclude='*.py[co]' \
    -C "${STAGED_REPO}/vendor/image-captioning" -cf - . \
    | tar -C "${content}/workspace/image-captioning" -xf -
tar --exclude='__pycache__' --exclude='*.py[co]' \
    -C "${STAGED_REPO}" -cf - . \
    | tar -C "${content}/opt/mlsbench-caption/repo" -xf -
cp "${BUILD}/input-proof.json" "${content}/opt/mlsbench-caption/input-proof.json"
cp "${STAGE}/repo.sha256" "${content}/opt/mlsbench-caption/repo.sha256"

(
    cd "${DATA_ROOT}"
    find . -type f ! -path '*/__pycache__/*' ! -name '*.py[co]' -print0 \
        | sort -z | xargs -0 sha256sum > "${BUILD}/data.sha256"
)
(
    cd "${STAGED_REPO}"
    find . -type f ! -path '*/__pycache__/*' ! -name '*.py[co]' -print0 \
        | sort -z | xargs -0 sha256sum > "${BUILD}/source.sha256"
)
cp "${BUILD}/data.sha256" "${content}/opt/mlsbench-caption/data.sha256"
cp "${BUILD}/source.sha256" "${content}/opt/mlsbench-caption/source.sha256"

tar --sort=name --mtime='UTC 2026-01-01' --owner=0 --group=0 --numeric-owner \
    -C "${whiteout}" -cf "${BUILD}/whiteout.tar" .
tar --sort=name --mtime='UTC 2026-01-01' --owner=0 --group=0 --numeric-owner \
    -C "${content}" -cf "${BUILD}/content.tar" .
sha256sum "${BUILD}/whiteout.tar" "${BUILD}/content.tar" \
    > "${BUILD}/layers.sha256"

printf '%s\n' \
    "runtime_image=${RUNTIME_IMAGE}" \
    "target_image=${IMAGE}" \
    "protocol=flickr8k_official_v1" \
    "train_images=6000" \
    "train_pairs=30000" \
    "eval_images=1000" \
    "epochs=10" \
    "batch_size=40" \
    "optimizer_steps=7500" \
    "manifest_sha256=${manifest_sha}" \
    "train_split_sha256=${train_split_sha}" \
    "test_split_sha256=${test_split_sha}" \
    "runtime_install=false" \
    "runtime_download=false" \
    > "${BUILD}/build-request.txt"

"${CRANE}" append --base "${RUNTIME_IMAGE}" \
    --new_layer "${BUILD}/whiteout.tar" --new_tag "${CLEAN_IMAGE}" \
    2>&1 | tee "${BUILD}/push-whiteout.log"
"${CRANE}" append --base "${CLEAN_IMAGE}" \
    --new_layer "${BUILD}/content.tar" --new_tag "${LAYER_IMAGE}" \
    2>&1 | tee "${BUILD}/push-content.log"
"${CRANE}" mutate "${LAYER_IMAGE}" --tag "${IMAGE}" \
    --workdir /workspace \
    --env CAPTION_DATA=/data/image-captioning \
    --env CAPTION_GPT2=/data/image-captioning/gpt2 \
    --env HF_HUB_OFFLINE=1 \
    --env HF_DATASETS_OFFLINE=1 \
    --env TRANSFORMERS_OFFLINE=1 \
    --env TOKENIZERS_PARALLELISM=false \
    --label org.mlsbench.caption.protocol=flickr8k_official_v1 \
    --label org.mlsbench.caption.manifest-sha256="${manifest_sha}" \
    --label org.mlsbench.caption.test-split-sha256="${test_split_sha}" \
    --label org.mlsbench.caption.train-images=6000 \
    --label org.mlsbench.caption.train-pairs=30000 \
    --label org.mlsbench.caption.eval-images=1000 \
    --label org.mlsbench.caption.optimizer-steps=7500 \
    2>&1 | tee "${BUILD}/mutate.log"
"${CRANE}" validate --remote "${IMAGE}" --fast
digest=$("${CRANE}" digest "${IMAGE}")
if [[ ! "${digest}" =~ ^sha256:[0-9a-f]{64}$ ]]; then
    echo "failed to resolve immutable caption image digest: ${digest}" >&2
    exit 4
fi
printf '%s@%s\n' "${IMAGE_REPO}" "${digest}" > "${BUILD}/image.ref"
printf '%s\n' "${digest}" > "${BUILD}/image.digest"
date -Iseconds > "${BUILD}/IMAGE_READY"
echo "CAPTION_FULL_IMAGE_READY image=${IMAGE_REPO}@${digest} manifest_sha256=${manifest_sha}"
