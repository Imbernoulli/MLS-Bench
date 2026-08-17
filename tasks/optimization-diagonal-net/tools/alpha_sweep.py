#!/usr/bin/env python3
"""Calibrate the initialisation scale alpha for the third diagonal-net setting.

The three visible settings must not all reward the same optimizer.  Two of them
(``d200_k5_a1e3``, ``d500_k10_a1e3``) sit deep in the RICH regime (alpha=1e-3),
where the diagonal-net's implicit bias is sparsity-inducing.  The third,
``d500_k10_a5e1``, keeps (d, k) identical and raises ONLY alpha, moving training
toward the LAZY / kernel regime -- so alpha is the single controlled variable.

This script sweeps alpha x baseline on the (d=500, k=10) problem and reports the
two things that decide whether a candidate alpha is usable:

  saturation  n* pinned at the top of the search grid (recovery never succeeds,
              so the setting cannot discriminate), or all baselines landing on
              the SAME n* (no signal).  Either one disqualifies the alpha.
  reordering  the baseline ranking differs from the alpha=1e-3 reference.
              Desirable -- it proves the setting probes a different regime --
              but not required.

Requires a GPU for the full grid.  Sanity-check the plumbing on CPU first:

    python3 tasks/optimization-diagonal-net/tools/alpha_sweep.py \
        --smoke --dim 50 --sparsity 3 --n-test 200 \
        --alphas 0.001 1.0 --baselines sgd adam

Full calibration run (GPU, hours):

    python3 tasks/optimization-diagonal-net/tools/alpha_sweep.py \
        --alphas 0.001 0.01 0.1 0.5 1.0 --grid-max 2000 --out sweep.json
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "tasks/optimization-diagonal-net"
HARBOR = ROOT / "harbor/tasks/mls-bench__optimization-diagonal-net"
SCAFFOLD = HARBOR / "environment/_scaffold/RAIN/opt_diagonal_net/custom_optimizer.py"
FIXED = SRC / "edits/fixed_benchmark.py"
DGP_DIR = HARBOR / "tests/eval/_inputgen/holdout/optimization-diagonal-net"

# Mirrors fixed_benchmark.SearchConfig / its --smoke branch.
FULL_GRID = (50, 75, 100, 150, 200, 300, 400, 600, 800, 1200, 1600)
SMOKE_GRID = (100, 200, 400, 800)
EDIT_START, EDIT_END = 23, 90


def alpha_tag(a: float) -> str:
    """Mirrors fixed_benchmark._alpha_tag."""
    return f"{float(a):g}".replace("-", "m").replace(".", "p")


def input_key(dim, sparsity, alpha_init, n_max_train, n_test, seed) -> str:
    """Mirrors fixed_benchmark._input_key."""
    return (f"d{int(dim)}_k{int(sparsity)}_a{alpha_tag(alpha_init)}"
            f"_nmax{int(n_max_train)}_nt{int(n_test)}_seed{int(seed)}")


def resolve_grid(base: tuple[int, ...], grid_max: int | None) -> list[int]:
    """Mirrors fixed_benchmark's --grid-max override."""
    if grid_max is None:
        return list(base)
    ext = [v for v in base if v <= grid_max]
    if not ext or ext[-1] < grid_max:
        ext.append(grid_max)
    return ext


def edit_content(baseline: str) -> str:
    p = SRC / "edits" / f"{baseline}.edit.py"
    ns: dict = {}
    exec(compile(p.read_text(), str(p), "exec"), ns, ns)
    (op,) = ns["OPS"]
    assert (op["start_line"], op["end_line"]) == (EDIT_START, EDIT_END), \
        f"{p}: unexpected edit range {op['start_line']}-{op['end_line']}"
    return op["content"]


def apply_baseline(baseline: str) -> str:
    """scaffold with the editable region replaced by this baseline's body."""
    lines = SCAFFOLD.read_text().split("\n")
    content = edit_content(baseline).split("\n")
    if content and content[-1] == "":
        content.pop()
    return "\n".join(lines[:EDIT_START - 1] + content + lines[EDIT_END:])


def encode_payloads(dim, sparsity, n_max, n_test, seeds) -> dict[int, str]:
    """Generate each seed's dataset once; alpha only changes the FILENAME."""
    sys.path.insert(0, str(DGP_DIR))
    import numpy as np
    import dgp  # host-only; never importable from inside the task container

    out = {}
    for seed in seeds:
        X_train, y_train, X_test, y_test = dgp.gen_input(
            dim=dim, sparsity=sparsity, n_max_train=n_max, n_test=n_test, seed=seed)
        buf = io.BytesIO()
        np.savez(buf,
                 X_train=np.ascontiguousarray(X_train, dtype=np.float64),
                 y_train=np.ascontiguousarray(y_train, dtype=np.float64),
                 X_test=np.ascontiguousarray(X_test, dtype=np.float64),
                 y_test=np.ascontiguousarray(y_test, dtype=np.float64))
        out[seed] = base64.b64encode(buf.getvalue()).decode("ascii")
    return out


def run_one(work: Path, baseline: str, alpha: float, args, grid) -> dict:
    pkg = work / "RAIN/opt_diagonal_net"
    (pkg / "custom_optimizer.py").write_text(apply_baseline(baseline))
    label = f"{baseline}_a{alpha_tag(alpha)}"
    out_dir = work / "out"
    cmd = [sys.executable, str(pkg / "custom_optimizer.py"),
           "--dim", str(args.dim), "--sparsity", str(args.sparsity),
           "--delta", str(args.delta), "--alpha-init", repr(float(alpha)),
           "--n-test", str(args.n_test), "--seed", str(args.seed),
           "--label", label, "--output-dir", str(out_dir)]
    if args.smoke:
        cmd.append("--smoke")
    if args.grid_max is not None:
        cmd += ["--grid-max", str(args.grid_max)]

    env = dict(os.environ)
    # Leave EPHEMERAL unset: the blobs are reused across baselines.
    env.pop("MLSBENCH_EPHEMERAL_INPUTS", None)

    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    elapsed = time.time() - t0

    res = {"baseline": baseline, "alpha": float(alpha), "elapsed_s": round(elapsed, 1),
           "returncode": proc.returncode, "n_star": None, "score": None}
    jf = out_dir / f"{label}_seed{args.seed}.json"
    if jf.exists():
        d = json.loads(jf.read_text())
        res["n_star"] = d["n_star"]
        res["score"] = d["score"]
    else:
        res["stderr_tail"] = proc.stderr[-1500:]
    res["saturated"] = res["n_star"] is not None and res["n_star"] >= max(grid)
    return res


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--alphas", type=float, nargs="+",
                   default=[0.001, 0.01, 0.1, 0.5, 1.0])
    p.add_argument("--baselines", nargs="+",
                   default=["sgd", "adagrad", "adam", "adam2"])
    p.add_argument("--dim", type=int, default=500)
    p.add_argument("--sparsity", type=int, default=10)
    p.add_argument("--delta", type=float, default=0.5)
    p.add_argument("--n-test", type=int, default=4096)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--grid-max", type=int, default=2000)
    p.add_argument("--smoke", action="store_true",
                   help="Use the small grid / 2 seeds (for CPU plumbing checks).")
    p.add_argument("--out", type=Path, default=None, help="Write results JSON here.")
    p.add_argument("--keep", action="store_true", help="Keep the scratch workspace.")
    p.add_argument("--reference-alpha", type=float, default=0.001,
                   help="Alpha whose baseline ranking is the comparison point.")
    args = p.parse_args()

    base_grid = SMOKE_GRID if args.smoke else FULL_GRID
    grid = resolve_grid(base_grid, args.grid_max)
    num_seeds = 2 if args.smoke else 5
    seeds = list(range(args.seed, args.seed + num_seeds))
    n_max = max(grid)

    print(f"grid      : {grid}")
    print(f"seeds     : {seeds}   (n_max_train={n_max}, n_test={args.n_test})")
    print(f"problem   : d={args.dim} k={args.sparsity} delta={args.delta}")
    print(f"sweep     : {len(args.alphas)} alphas x {len(args.baselines)} baselines "
          f"= {len(args.alphas) * len(args.baselines)} runs\n")

    work = Path(tempfile.mkdtemp(prefix="diagnet_alpha_sweep_"))
    pkg = work / "RAIN/opt_diagonal_net"
    (pkg / "_inputs").mkdir(parents=True)
    shutil.copy(FIXED, pkg / "fixed_benchmark.py")

    print("generating datasets ...", flush=True)
    payloads = encode_payloads(args.dim, args.sparsity, n_max, args.n_test, seeds)
    for alpha in args.alphas:
        for seed in seeds:
            key = input_key(args.dim, args.sparsity, alpha, n_max, args.n_test, seed)
            (pkg / "_inputs" / f"{key}.npz.b64").write_text(payloads[seed])
    print(f"wrote {len(args.alphas) * len(seeds)} blobs into {pkg / '_inputs'}\n")

    results = []
    try:
        for alpha in args.alphas:
            for baseline in args.baselines:
                print(f"--> alpha={alpha:<8g} {baseline:<8} ", end="", flush=True)
                r = run_one(work, baseline, alpha, args, grid)
                results.append(r)
                if r["n_star"] is None:
                    print(f"FAILED (rc={r['returncode']}) after {r['elapsed_s']}s")
                    print("     " + (r.get("stderr_tail", "").strip().splitlines() or ["<no stderr>"])[-1])
                else:
                    flag = "  [SATURATED]" if r["saturated"] else ""
                    print(f"n*={r['n_star']:<6} score={r['score']:.6f}  "
                          f"{r['elapsed_s']}s{flag}")
    finally:
        if args.keep:
            print(f"\nscratch workspace kept at {work}")
        else:
            shutil.rmtree(work, ignore_errors=True)

    # ---------------- verdict ----------------
    print("\n" + "=" * 72)
    by_alpha: dict[float, dict[str, int | None]] = {}
    for r in results:
        by_alpha.setdefault(r["alpha"], {})[r["baseline"]] = r["n_star"]

    def ranking(d: dict[str, int | None]) -> list[str] | None:
        if any(v is None for v in d.values()):
            return None
        return sorted(d, key=lambda b: (d[b], b))

    ref = ranking(by_alpha.get(args.reference_alpha, {}))
    print(f"reference ranking (alpha={args.reference_alpha:g}): "
          f"{' < '.join(ref) if ref else '<incomplete>'}\n")

    print(f"{'alpha':<10}{'ranking (best n* first)':<44}{'verdict'}")
    print("-" * 72)
    for alpha in args.alphas:
        d = by_alpha.get(alpha, {})
        rank = ranking(d)
        notes = []
        if rank is None:
            notes.append("INCOMPLETE (a run failed)")
        else:
            vals = [d[b] for b in rank]
            if any(v >= max(grid) for v in vals):
                pinned = [b for b in rank if d[b] >= max(grid)]
                notes.append(f"SATURATED: {','.join(pinned)} pinned at n*={max(grid)}")
            if len(set(vals)) == 1:
                notes.append(f"DEGENERATE: all baselines n*={vals[0]}")
            if ref and rank != ref:
                notes.append("reordered vs reference")
            elif ref and rank == ref and alpha != args.reference_alpha:
                notes.append("same ranking as reference")
        shown = " < ".join(f"{b}({d[b]})" for b in rank) if rank else "-"
        print(f"{alpha:<10g}{shown:<44}{'; '.join(notes) or 'usable'}")

    usable = [a for a in args.alphas
              if (rk := ranking(by_alpha.get(a, {}))) is not None
              and not any(by_alpha[a][b] >= max(grid) for b in rk)
              and len({by_alpha[a][b] for b in rk}) > 1]
    print("\nusable (not saturated, not degenerate):",
          ", ".join(f"{a:g}" for a in usable) or "NONE")
    reordered = [a for a in usable
                 if ref and ranking(by_alpha[a]) != ref]
    print("of those, also reordering the baselines:",
          ", ".join(f"{a:g}" for a in reordered) or "none")
    print("\nThe task's committed third setting is d500_k10_a5e1 (alpha=0.5). If 0.5 "
          "is not in the usable list,\npick one that is and update: the four "
          "scripts/d500_k10_a5e1.sh copies (--alpha-init, --inputs-glob,\nheader), the "
          "label in config.json + score_spec.py + leaderboard.csv columns, and "
          "task_description.md.\nThen re-run tools/check_consistency.py.")

    if args.out:
        args.out.write_text(json.dumps(
            {"problem": {"dim": args.dim, "sparsity": args.sparsity,
                         "delta": args.delta, "n_test": args.n_test,
                         "grid": grid, "seeds": seeds, "smoke": args.smoke},
             "results": results}, indent=2) + "\n")
        print(f"\nwrote {args.out}")

    return 0 if all(r["n_star"] is not None for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
