#!/usr/bin/env python3
"""FIXED evaluation entry for MLS-Bench ephemeral-input tasks.

Problem this solves: the evaluated program is the agent-EDITABLE module, and
Python executes module top-level statements at import time. Launched
directly (``python <editable_module>.py``), any top-level statement the
agent places inside its editable range runs while the staged input blobs
(withheld labels / held-out targets / oracle tables) are still on disk —
BEFORE the module's fixed runner gets a chance to load and scrub them.

This wrapper is FIXED: it lives with the task's eval scripts (native
``tasks/<t>/scripts/``, Harbor ``tests/eval/scripts/``), which are mounted
read-only from outside the agent-editable workspace. It inverts the order:

  1. read every staged blob matching ``--inputs-glob`` into memory;
  2. unlink the blobs (gated on MLSBENCH_EPHEMERAL_INPUTS=1, the harness's
     "inputs are re-staged before every evaluation" marker);
  3. only THEN import the editable module — a hostile top-level statement
     now finds an empty inputs dir;

IMPORTANT — the glob must be scoped to the ACTIVE RUN's blobs only. The
Harbor verifier starts a whole group wave of evaluations CONCURRENTLY
(score_task.py _run_eval_wave), each staging its own (ENV, SEED) blobs into
the shared workspace; a wildcard sweep here would unlink a sibling eval's
just-staged files before that sibling could read them. Each eval script
therefore passes a glob that interpolates its own setting constants and
${SEED} — mirroring the blob naming of the task's mid_edit.py (parity:
{tag}_seed{seed}_s*.labels.b64; diag: the sigma-keyed setting family of
fixed_benchmark._input_key, the same scope as its in-module _scrub_inputs;
nas: the exact nb201_tables_{env}_s{seed}.json). Native runs are strictly
serialized, where exact scoping is simply a further tightening.
  4. inject the in-memory payloads into the FIXED loader module
     (``--inject-module``, default the editable module itself) as
     ``_PRELOADED_INPUTS = {basename: content}``;
  5. call the module's FIXED entry function (``--entry``).

Back-compat: when the module tree predates the injection protocol (the
loader has no ``_PRELOADED_INPUTS`` marker, or the entry function is
missing), the behavior depends on the task type. On a NON-ephemeral task
(nothing is withheld) the wrapper leaves the blobs on disk and executes the
module unchanged as ``__main__`` — byte-for-byte the legacy flow, whose
fixed runner loads and scrubs in-process. On an EPHEMERAL task there is NO
on-disk fallback: the blobs were already read into memory and unlinked
above, and the wrapper REFUSES to run the module (exit 5). Every template
and scaffold carries the marker since the protocol shipped, so a
marker-less ephemeral module means a stale or tampered workspace, and the
legacy on-disk flow is exactly the path that exposed the withheld data.

Accepted residual (documented, not solved here): once the editable module
is imported, agent code shares the interpreter with the fixed runner and
could in principle walk interpreter memory (gc, frames) to reach the loaded
arrays — that in-memory boundary is inherent to the in-process task design
(the fixed runner already holds withheld data in memory while calling agent
hooks). This wrapper's contract is filesystem-only: no withheld bytes
remain ON DISK once any agent-authored code can run.

Usage:
    python fixed_entry.py --module <editable.py> --inputs-glob <glob> \
        [--inputs-glob <glob> ...] [--inject-module <name>] [--entry main] \
        -- [args passed to the module]

Globs are resolved relative to the current working directory (the eval
scripts cd to the workspace root / package dir first).
"""

from __future__ import annotations

import argparse
import glob as _glob
import importlib.util
import os
import runpy
import sys


def _parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    if "--" in argv:
        split = argv.index("--")
        own, rest = argv[:split], argv[split + 1:]
    else:
        own, rest = argv, []
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--module", required=True,
                        help="Path to the agent-editable module to run.")
    parser.add_argument("--inputs-glob", action="append", default=[],
                        help="Glob of staged input blobs to preload+unlink "
                             "(repeatable; relative to the CWD).")
    parser.add_argument("--inject-module", default=None,
                        help="Module name holding the FIXED loaders to inject "
                             "_PRELOADED_INPUTS into (default: the editable "
                             "module itself). Must be importable from the "
                             "editable module's directory.")
    parser.add_argument("--entry", default="main",
                        help="FIXED entry function to invoke (default: main).")
    args = parser.parse_args(own)
    return args, rest


def _read_text(path: str) -> str | None:
    try:
        with open(path, "r") as fh:
            return fh.read()
    except OSError:
        return None


def main() -> int:
    args, module_argv = _parse_args(sys.argv[1:])
    module_path = os.path.abspath(args.module)
    module_dir = os.path.dirname(module_path)
    module_name = os.path.splitext(os.path.basename(module_path))[0]
    inject_name = args.inject_module or module_name

    module_text = _read_text(module_path)
    if module_text is None:
        print(f"[fixed-entry] ERROR: cannot read module {module_path}",
              file=sys.stderr)
        return 2

    sys.argv = [module_path] + list(module_argv)
    if module_dir and module_dir not in sys.path:
        sys.path.insert(0, module_dir)

    # ── 1+2: preload the staged blobs, then unlink them — BEFORE the
    # protocol decision and BEFORE any module code runs. Doing the unlink
    # first means that even a marker-less (stale/tampered) module never sees
    # the withheld data on disk. BOTH steps are FATAL on failure: a read
    # failure means we cannot run the eval; an unlink failure means the
    # secret would remain readable, which breaks the isolation contract. In
    # either case: exit non-zero, run NO module code. (Blobs live in the
    # writable package dir, not the read-only task mount, so a healthy run
    # always CAN delete them; a failure here is a real filesystem fault.)
    preloaded: dict[str, str] = {}
    ephemeral = os.environ.get("MLSBENCH_EPHEMERAL_INPUTS") == "1"
    for pattern in args.inputs_glob:
        for path in sorted(_glob.glob(pattern)):
            if not os.path.isfile(path):
                continue
            try:
                with open(path, "r") as fh:
                    preloaded[os.path.basename(path)] = fh.read()
            except OSError as exc:
                print(
                    f"[fixed-entry] FATAL: could not read staged input "
                    f"{path}: {exc} — refusing to run the evaluation",
                    file=sys.stderr, flush=True,
                )
                return 3
            if ephemeral:
                try:
                    os.remove(path)
                except OSError as exc:
                    print(
                        f"[fixed-entry] FATAL: could not unlink staged input "
                        f"{path}: {exc} — the withheld data would remain "
                        "readable by editable code; refusing to run",
                        file=sys.stderr, flush=True,
                    )
                    return 4
    print(
        f"[fixed-entry] preloaded {len(preloaded)} input blob(s)"
        + (" and unlinked them" if ephemeral else " (kept on disk: not "
           "marked ephemeral)")
        + " before deciding how to run the module",
        flush=True,
    )

    # ── Protocol detection (AFTER the ephemeral unlink) ─────────────────
    # The injection protocol requires (a) the FIXED loader module to consult
    # _PRELOADED_INPUTS and (b) the editable module to expose the entry
    # function. Both live in guard-protected fixed regions, so a plain text
    # scan is authoritative. Round 7 shipped the marker into every
    # template + scaffold in lockstep, so a marker-LESS module in an
    # EPHEMERAL task means a stale or tampered workspace.
    if inject_name == module_name:
        inject_text = module_text
    else:
        inject_text = _read_text(os.path.join(module_dir, inject_name + ".py"))
    protocol_ok = (
        inject_text is not None
        and "_PRELOADED_INPUTS" in inject_text
        and f"def {args.entry}(" in module_text
    )

    if not protocol_ok:
        if ephemeral:
            # REFUSE. The blobs were already read into memory and unlinked
            # above, so injection COULD still work if the module were
            # compliant — but it is not, and running a marker-less module
            # (as __main__, on-disk) is exactly the round-7 fallback that
            # exposed secrets. For an ephemeral task a marker-less module is
            # a stale/tampered workspace: fail, do not execute it.
            print(
                "[fixed-entry] FATAL: module lacks the _PRELOADED_INPUTS "
                "injection protocol but the task is ephemeral (withheld "
                "inputs). This workspace predates the protocol or was "
                "tampered with — refusing to run it. Recreate / re-render "
                "the workspace from the current template.",
                file=sys.stderr, flush=True,
            )
            return 5
        # Non-ephemeral: nothing is withheld, so the legacy on-disk flow is
        # safe. (The blobs above were kept on disk for exactly this path.)
        print(
            "[fixed-entry] legacy module (no _PRELOADED_INPUTS protocol) on a "
            "non-ephemeral task; running it directly as __main__",
            flush=True,
        )
        runpy.run_path(module_path, run_name="__main__")
        return 0

    # ── 3: import the editable module (top-level agent code runs NOW, on ──
    # ── an already-empty inputs dir; _PRELOADED_INPUTS is still None)    ──
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    # ── 4: inject the payloads into the FIXED loader module ────────────
    target = sys.modules.get(inject_name)
    if target is None:
        # The editable module did not import it itself; do so now (fixed
        # loader modules contain no agent code).
        import importlib as _importlib
        target = _importlib.import_module(inject_name)
    setattr(target, "_PRELOADED_INPUTS", preloaded)

    # ── 5: run the FIXED entry ──────────────────────────────────────────
    getattr(module, args.entry)()
    return 0


if __name__ == "__main__":
    sys.exit(main())
