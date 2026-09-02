#!/bin/bash
# Diagnostic-only wrapper: run the real training script for three iterations.
# The production task and its scale are left untouched.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cfg=/workspace/humanoid-gym/humanoid/envs/custom/humanoid_config.py
backup="${cfg}.mlsbench-smoke-backup"
cp "$cfg" "$backup"
restore() {
  cp "$backup" "$cfg"
  rm -f "$backup"
}
trap restore EXIT

python3 - "$cfg" <<'PY'
from pathlib import Path
import re
import sys
p = Path(sys.argv[1])
s = p.read_text()
updated, n = re.subn(r"(max_iterations\s*=\s*)3001(\s*#)", r"\g<1>3\g<2>", s, count=1)
if n != 1:
    raise SystemExit(f"expected one max_iterations=3001 in {p}, found {n}")
p.write_text(updated)
print("SMOKE override: max_iterations=3 (runtime only)")
PY

bash "$SCRIPT_DIR/train.sh"
