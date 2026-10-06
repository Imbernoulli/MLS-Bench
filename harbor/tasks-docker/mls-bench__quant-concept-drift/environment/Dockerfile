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
FROM bohanlyu2022/mlsbench-harbor-qlib@sha256:58f0c45df7cba61d9a2b72c6b4b08320cc4d333af6b9fb45de3cb3ec45c25f14
# 2. environment
ENV HOME="/root"

# 3. workspace
RUN rm -rf /workspace && mkdir -p /workspace
COPY _workspace/ /workspace/
WORKDIR /workspace

# 5. data
# no data deps declared by this task's packages.


# --- mlsbench: task-declared extra steps (scripts/harbor/extra_docker_steps.py) ---
RUN python3 -c "import hashlib, pathlib, shutil, struct; CUT = \"2016-01-01\"; CAL = \"557a9ef7a12fbefb654270f7cd825d72faa68a10da4ceed09d1b1031fe12ef4c\"; INST = {\"csi300\": \"fc0b62443aaffbd1d3a6c7aa96a40fdbc054f73b3a33ed814222f45842ffbe64\", \"csi100\": \"2eebacab048c6c3fc93f783a42805d7c09ec197013c3e06ba3135f2e40ad7b1a\"}; AGG_FULL = \"69bc06861408dad9b529b354155db1bf6709d08275041179d18050fd516d874c\"; AGG_VIS = \"9e83ad2eace13da8b46b12dbd68dee3a9a29558a668fb37962bef51c1e69dbc7\"; exec('root = pathlib.Path(\"/root/.qlib/qlib_data/cn_data\")\nassert hashlib.sha256((root / \"calendars/day.txt\").read_bytes()).hexdigest() == CAL, \"unexpected calendar\"\ncal = (root / \"calendars/day.txt\").read_text().splitlines()\nK = sum(1 for d in cal if d < CUT); last = cal[K - 1]\nU = {\"sh000300\"}\nfor n, h in INST.items():\n p = root / \"instruments\" / (n + \".txt\"); assert hashlib.sha256(p.read_bytes()).hexdigest() == h, \"unexpected \" + n\n rows = [l.split(chr(9)) for l in p.read_text().splitlines() if l.strip()]\n U |= {r[0].lower() for r in rows}\n p.write_text(\"\".join(r[0] + chr(9) + r[1] + chr(9) + min(r[2], last) + chr(10) for r in rows if r[1] < CUT))\nfull = hashlib.sha256(); vis = hashlib.sha256()\nfor inst in sorted(U):\n d = root / \"features\" / inst\n if not d.is_dir():\n  continue\n for f in sorted(d.iterdir()):\n  raw = f.read_bytes(); rel = \"features/\" + inst + \"/\" + f.name\n  s = int(struct.unpack(\"<f\", raw[:4])[0]); n = (len(raw) - 4) // 4; keep = max(0, min(n, K - s)); v = raw[:4 + 4 * keep]\n  full.update(rel.encode() + hashlib.sha256(raw).digest()); vis.update(rel.encode() + hashlib.sha256(v).digest()); f.write_bytes(v)\nassert full.hexdigest() == AGG_FULL, \"unexpected source bins\"\nassert vis.hexdigest() == AGG_VIS, \"unexpected visible prefix\"\n(root / \"calendars/day.txt\").write_text(\"\".join(d + chr(10) for d in cal[:K]))\n[z.unlink() for z in root.glob(\"*.zip\")]\nshutil.rmtree(\"/root/rl_data\", ignore_errors=True)\n(root / \".mlsb_test_withheld\").write_text(\"Data after \" + last + \" of the scored universe (CSI300/CSI100 constituents, SH000300) are withheld from the workspace and mounted only for the final evaluation.\")\nprint(\"withheld qlib data after\", last, \"for\", len(U), \"instruments\")\n')"
RUN test -z "$(find / -xdev \( -name '*qlib_data*.zip' -o -path '/root/rl_data' \) -not -path '/proc/*' 2>/dev/null)" && python3 -c "import pathlib; cal = pathlib.Path(\"/root/.qlib/qlib_data/cn_data/calendars/day.txt\").read_text().split(); assert cal[-1] < \"2016-01-01\", cal[-1]; print(\"calendar ends\", cal[-1])"
