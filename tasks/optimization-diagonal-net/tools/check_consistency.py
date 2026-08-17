#!/usr/bin/env python3
"""Static self-consistency checks for the optimization-diagonal-net task.

This task is duplicated across the source tree and four rendered Harbor
locations, and the Harbor adapter currently CANNOT re-render it (see
``adapter.py`` ``_ALLOWED_OP_IMPORT_ROOTS``: this task's ``mid_edit.py``
imports base64/io/os/numpy/dgp, none of which are whitelisted).  Until that is
fixed, the copies are maintained by hand — so run this script after any edit.

    python3 tasks/optimization-diagonal-net/tools/check_consistency.py

Checks:
  1. no residual ``sigma`` / ``_s01`` / ``_s02`` identifiers
  2. byte-identical duplicates (fixed_benchmark.py x5, mid_edit.py x3, ...)
  3. the four setting scripts agree across all four rendered locations
  4. each script's ``--inputs-glob`` prefix matches ``_input_key``'s output for
     that script's own (dim, sparsity, alpha-init) -- a mismatch means the eval
     looks for blobs that the input generator never wrote
  5. settings that share (dim, sparsity) still get DISJOINT blob prefixes
     (regression guard for issue #82)
  6. ``get_hyperparameters`` signature identical in the scaffold and in all
     edit ops, and every edit op still targets the config's editable range
  7. ``pristine_manifest.json`` matches the scaffold bytes
  8. every .py parses, every .sh parses
"""
from __future__ import annotations

import ast
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "tasks/optimization-diagonal-net"
HARBOR = ROOT / "harbor/tasks/mls-bench__optimization-diagonal-net"
INPUTGEN = HARBOR / "tests/eval/_inputgen/tasks/optimization-diagonal-net"

SCRIPT_DIRS = [SRC / "scripts", INPUTGEN / "scripts",
               HARBOR / "tests/eval/scripts", HARBOR / "tests/meta/scripts"]
EDIT_DIRS = [SRC / "edits", INPUTGEN / "edits", HARBOR / "tests/meta/edits"]

failures: list[str] = []
checks = 0


def check(ok: bool, label: str) -> None:
    global checks
    checks += 1
    if not ok:
        failures.append(label)
        print(f"  FAIL  {label}")


def alpha_tag(a: float) -> str:
    """Must mirror fixed_benchmark._alpha_tag / mid_edit._alpha_tag."""
    return f"{float(a):g}".replace("-", "m").replace(".", "p")


def script_args(text: str) -> dict[str, str | None]:
    """Mirror mid_edit._parse_script_args."""
    toks = text.replace("\\\n", " ").split()
    args: dict[str, str | None] = {}
    i = 0
    while i < len(toks):
        if toks[i].startswith("--"):
            has_val = i + 1 < len(toks) and not toks[i + 1].startswith("--")
            args[toks[i][2:]] = toks[i + 1] if has_val else None
            i += 2 if has_val else 1
        else:
            i += 1
    return args


# --- 1. no residual sigma / old setting labels -------------------------------
print("[1] residual sigma / old labels")
STALE = re.compile(r"\b_?sigma\b|--sigma|_s01\b|_s02\b|sigma-keyed")
for root in (SRC, HARBOR):
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.suffix not in {".py", ".sh", ".json", ".md", ".csv", ".toml"}:
            continue
        # vendored scoring lib uses `sigma` for the sigmoid; not ours.
        # this script names the retired token in its own docs/regex.
        # HANDOFF.md documents the sigma -> alpha migration, so it must be able
        # to name what was retired and what the old setting labels were.
        if "mlsbench_src" in p.parts or "tools" in p.parts:
            continue
        if p.name == "HANDOFF.md":
            continue
        for i, line in enumerate(p.read_text(errors="replace").splitlines(), 1):
            if STALE.search(line):
                check(False, f"stale token {p.relative_to(ROOT)}:{i}: {line.strip()[:70]}")
check(True, "residual scan completed")

# --- 2. byte-identical duplicates -------------------------------------------
print("[2] duplicate files identical")
DUPES = {
    "fixed_benchmark.py": [
        SRC / "edits/fixed_benchmark.py",
        HARBOR / "environment/_scaffold/RAIN/opt_diagonal_net/fixed_benchmark.py",
        INPUTGEN / "edits/fixed_benchmark.py",
        HARBOR / "tests/meta/edits/fixed_benchmark.py",
        HARBOR / "tests/meta/pristine/RAIN/opt_diagonal_net/fixed_benchmark.py",
    ],
    "mid_edit.py": [d / "mid_edit.py" for d in EDIT_DIRS],
    "custom_optimizer.py": [
        HARBOR / "environment/_scaffold/RAIN/opt_diagonal_net/custom_optimizer.py",
        HARBOR / "tests/meta/pristine/RAIN/opt_diagonal_net/custom_optimizer.py",
    ],
    "fixed_entry.py": [SRC / "scripts/fixed_entry.py",
                       HARBOR / "tests/eval/scripts/fixed_entry.py",
                       HARBOR / "tests/meta/scripts/fixed_entry.py"],
}
for name, paths in DUPES.items():
    digests = {hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    check(len(digests) == 1, f"{name}: {len(paths)} copies, {len(digests)} distinct digests")

for name in sorted(p.name for p in (SRC / "edits").glob("*.edit.py")):
    digests = {hashlib.sha256((d / name).read_bytes()).hexdigest() for d in EDIT_DIRS}
    check(len(digests) == 1, f"{name}: {len(digests)} distinct digests across {len(EDIT_DIRS)} dirs")

# --- 3./4./5. setting scripts ------------------------------------------------
print("[3] setting scripts consistent across rendered copies")
cfg = json.loads((SRC / "config.json").read_text())
labels = [tc["label"] for tc in cfg["test_cmds"]]
KEYED = ("dim", "sparsity", "delta", "alpha-init", "n-test", "grid-max")

prefixes: dict[str, str] = {}
for label in labels:
    per_dir = {}
    for d in SCRIPT_DIRS:
        p = d / f"{label}.sh"
        if not p.exists():
            check(False, f"missing script {p.relative_to(ROOT)}")
            continue
        a = script_args(p.read_text())
        per_dir[d.relative_to(ROOT).as_posix()] = {k: a.get(k) for k in KEYED}
    check(len({json.dumps(v, sort_keys=True) for v in per_dir.values()}) == 1,
          f"{label}: keyed args agree across {len(per_dir)} copies")

    # 4. glob prefix must equal what _input_key() will produce
    ref = next(iter(per_dir.values()))
    want = f"d{int(ref['dim'])}_k{int(ref['sparsity'])}_a{alpha_tag(ref['alpha-init'] or 1e-3)}_"
    prefixes[label] = want
    for d in SCRIPT_DIRS:
        p = d / f"{label}.sh"
        if not p.exists():
            continue
        m = re.search(r'--inputs-glob "([^"]+)"', p.read_text())
        check(bool(m) and Path(m.group(1)).name.startswith(want),
              f"{label}: --inputs-glob prefix == {want!r} in {d.name}")

print("[5] settings sharing (dim, sparsity) keep disjoint blob prefixes  [issue #82]")
seen: dict[str, str] = {}
for label, pre in prefixes.items():
    if pre in seen:
        check(False, f"blob prefix collision: {label} and {seen[pre]} both use {pre!r}")
    seen[pre] = label
check(len(set(prefixes.values())) == len(prefixes),
      f"{len(prefixes)} settings -> {len(set(prefixes.values()))} distinct blob prefixes")

# --- 6. editable signature + edit ranges ------------------------------------
print("[6] get_hyperparameters signature and edit ranges")
scaffold = (HARBOR / "environment/_scaffold/RAIN/opt_diagonal_net/custom_optimizer.py").read_text()
SIG_RE = re.compile(r"def get_hyperparameters\((.*?)\) -> dict\[str, Any\]:", re.S)


def sig_params(text: str) -> list[str] | None:
    m = SIG_RE.search(text)
    if not m:
        return None
    return [p.split(":")[0].strip() for p in m.group(1).split(",") if p.strip()]


want_sig = sig_params(scaffold)
check(want_sig == ["dim", "sparsity", "delta", "alpha_init"],
      f"scaffold signature == (dim, sparsity, delta, alpha_init), got {want_sig}")

edit_range = cfg["files"][0]["edit"][0]
for d in EDIT_DIRS:
    for p in sorted(d.glob("*.edit.py")):
        ns: dict = {}
        exec(compile(p.read_text(), str(p), "exec"), ns, ns)
        for op in ns["OPS"]:
            check(op["start_line"] == edit_range["start"] and op["end_line"] == edit_range["end"],
                  f"{p.relative_to(ROOT)}: op range {op['start_line']}-{op['end_line']} "
                  f"== config {edit_range['start']}-{edit_range['end']}")
            check(sig_params(op["content"]) == want_sig,
                  f"{p.relative_to(ROOT)}: signature matches scaffold")

# --- 7. pristine manifest ---------------------------------------------------
print("[7] pristine_manifest.json")
mani = json.loads((HARBOR / "tests/meta/pristine_manifest.json").read_text())
for rel, want in mani.items():
    f = HARBOR / "environment/_scaffold" / rel
    if not f.exists():
        continue  # ships with the RAIN package
    check(hashlib.sha256(f.read_bytes()).hexdigest() == want, f"manifest hash for {rel}")

# --- 8. syntax --------------------------------------------------------------
print("[8] syntax")
for root in (SRC, HARBOR):
    for p in sorted(root.rglob("*.py")):
        if "mlsbench_src" in p.parts:
            continue
        try:
            ast.parse(p.read_text())
        except SyntaxError as exc:
            check(False, f"{p.relative_to(ROOT)}: {exc}")
    for p in sorted(root.rglob("*.sh")):
        if subprocess.run(["bash", "-n", str(p)], capture_output=True).returncode:
            check(False, f"bash -n failed: {p.relative_to(ROOT)}")
check(True, "syntax scan completed")

print(f"\n{checks - len(failures)}/{checks} checks passed")
if failures:
    print(f"\n{len(failures)} FAILURE(S):")
    for f in failures:
        print(f"  - {f}")
sys.exit(1 if failures else 0)
