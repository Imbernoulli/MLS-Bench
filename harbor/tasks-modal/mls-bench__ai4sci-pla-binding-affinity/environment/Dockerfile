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
FROM bohanlyu2022/mlsbench-harbor-ehign_pla@sha256:b6c61ad0f6c4913c161ddefdfd21e88a2a91e3ed7af0606d9543b757f45dbeea
# 2. environment
ENV PYTHONPATH="/workspace/EHIGN_PLA:${PYTHONPATH}"
ENV DATA_DIR="/data"

# 3. workspace
RUN rm -rf /workspace && mkdir -p /workspace
COPY _workspace/ /workspace/
WORKDIR /workspace

# 5. data
# --- data deps declared by EHIGN_PLA ---
RUN if [ -d "/data" ] && [ ! -L "/data" ]; then mkdir -p /mlsb_data_root/EHIGN_PLA && (cp -aln /data/. /mlsb_data_root/EHIGN_PLA/ 2>/dev/null || cp -an /data/. /mlsb_data_root/EHIGN_PLA/); fi
# ehign-pla -> /data
COPY _data_scripts/EHIGN_PLA/vendor__data_scripts__EHIGN_PLA/ /tmp/prepare_EHIGN_PLA__vendor__data_scripts__EHIGN_PLA/
RUN python3 -m pip install --no-cache-dir --target /tmp/prepdeps_EHIGN_PLA__vendor__data_scripts__EHIGN_PLA torch gdown
RUN PYTHONPATH=/tmp/prepdeps_EHIGN_PLA__vendor__data_scripts__EHIGN_PLA:${PYTHONPATH} HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 HF_DATASETS_OFFLINE=0 python3 /tmp/prepare_EHIGN_PLA__vendor__data_scripts__EHIGN_PLA/prepare_data.py --data-root /mlsb_data_root
RUN if [ -e "/data" ] && [ ! -L "/data" ]; then cp -aln /mlsb_data_root/EHIGN_PLA/. /data/ 2>/dev/null || cp -an /mlsb_data_root/EHIGN_PLA/. /data/; else mkdir -p "$(dirname /data)" && ln -sfn /mlsb_data_root/EHIGN_PLA /data; fi
RUN test -e "/mlsb_data_root/EHIGN_PLA/train_data.pt" || (echo "MISSING DATA DEP: ehign-pla -> /mlsb_data_root/EHIGN_PLA/train_data.pt" >&2; exit 1)
RUN test -e "/mlsb_data_root/EHIGN_PLA/valid_data.pt" || (echo "MISSING DATA DEP: ehign-pla -> /mlsb_data_root/EHIGN_PLA/valid_data.pt" >&2; exit 1)
RUN test -e "/mlsb_data_root/EHIGN_PLA/test2013_data.pt" || (echo "MISSING DATA DEP: ehign-pla -> /mlsb_data_root/EHIGN_PLA/test2013_data.pt" >&2; exit 1)
RUN test -e "/mlsb_data_root/EHIGN_PLA/test2016_data.pt" || (echo "MISSING DATA DEP: ehign-pla -> /mlsb_data_root/EHIGN_PLA/test2016_data.pt" >&2; exit 1)
RUN test -e "/mlsb_data_root/EHIGN_PLA/test2019_data.pt" || (echo "MISSING DATA DEP: ehign-pla -> /mlsb_data_root/EHIGN_PLA/test2019_data.pt" >&2; exit 1)


# --- mlsbench: task-declared extra steps (scripts/harbor/extra_docker_steps.py) ---
RUN python3 -c "import hashlib, os, torch; pt = {\"test2013_data.pt\": \"31a3986a3ea2d6c251dd2fdffd2973e84aa23210e8613c463a746a7aaf0cd88a\", \"test2016_data.pt\": \"ae88ad3d270953738008319d3f16414ee9755cfd98d393b915a976eee9c96b46\", \"test2019_data.pt\": \"00d8a3c0213b77c311e535206e5f079d00b9d86a5f3faf77e44a675386729cd0\"}; csv = {\"test2013.csv\": \"79404f55a435d3d885ab526cd369549018fc3036c4e29f7337f0ce09f6dc9f33\", \"test2016.csv\": \"731302829895b598a65390c97d56e28b1ddf29886ffe8d2349f30fa070f788e8\", \"test2019.csv\": \"693c6072cc5e04584ffcbed40a2cee4aeac5965f63483fee80b7fd16976db65f\"}; exec('hits = [os.path.join(d, f) for d, _, fs in os.walk(\"/\") if not d.startswith((\"/proc\", \"/sys\", \"/dev\")) for f in fs if f in pt or f in csv]\ndone = set()\nfor h in hits:\n rp = os.path.realpath(h)\n if rp in done:\n  continue\n done.add(rp); name = os.path.basename(h); digest = hashlib.sha256(open(h, \"rb\").read()).hexdigest()\n if digest != (pt.get(name) or csv.get(name)) and name in pt and not any(\"label\" in it for it in torch.load(h, weights_only=False)):\n  print(\"already label-free (hard link of a processed copy):\", h); continue\n assert digest == (pt.get(name) or csv.get(name)), \"unexpected source data: \" + h\n if name in csv:\n  os.remove(h)\n else:\n  items = torch.load(h, weights_only=False); [it.pop(\"label\") for it in items]; torch.save(items, h)\n open(os.path.join(os.path.dirname(h), \".mlsb_test_withheld\"), \"w\").write(\"The PDBbind test labels are withheld from the workspace and mounted only for the final evaluation.\")\n print(\"withheld test labels:\", h)\nassert all(os.path.realpath(\"/data/\" + n) in done for n in pt), \"a test set is missing under /data\"\n')"
RUN python3 -c "import os, torch; names = [\"test2013.csv\", \"test2013_data.pt\", \"test2016.csv\", \"test2016_data.pt\", \"test2019.csv\", \"test2019_data.pt\"]; hits = [os.path.join(d, f) for d, _, fs in os.walk(\"/\") if not d.startswith((\"/proc\", \"/sys\", \"/dev\")) for f in fs if f in names]; assert not [h for h in hits if h.endswith(\".csv\")], hits; assert all(not any(\"label\" in it for it in torch.load(h, weights_only=False)) for h in hits), hits; print(\"label-free test sets:\", hits)"
