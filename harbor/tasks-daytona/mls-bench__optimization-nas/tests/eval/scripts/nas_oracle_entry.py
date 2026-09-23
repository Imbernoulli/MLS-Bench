#!/usr/bin/env python3
"""FIXED evaluation entry for optimization-nas: out-of-process query oracle.

The search program (``custom_nas_search.py``) is agent-EDITABLE, and anything
that lives in its interpreter is reachable from agent code (``__closure__``,
``__dict__``, ``gc.get_referents``, frame walking, ctypes). An in-process
lookup table plus an in-process query counter therefore cannot enforce the
K-query budget: agent code can read the whole validation table or reset the
counter. This wrapper keeps both OUT of the agent's process:

  1. read this run's staged validation table (JSON on stdin in Harbor via
     ``apply.py --emit-json``; a staged file natively, unlinked here when
     MLSBENCH_EPHEMERAL_INPUTS=1) — no editable code has run yet;
  2. mark this process non-dumpable (``prctl(PR_SET_DUMPABLE, 0)``) so an
     unprivileged process cannot read its memory via /proc/<pid>/mem;
  3. start the search as a FRESH interpreter (fork+exec; the table is never
     sent to it) connected to this process by two pipes, with stdin closed;
  4. answer its queries here: every request counts, the budget
     (NAS_EPOCHS, default 30) is enforced here, and a request past the
     budget is refused (the search's BenchmarkAPI raises
     BudgetExceededError);
  5. after the search exits, print the authoritative ``ORACLE_QUERIES`` line
     and exit with the search's return code.

The child side (``--child``) imports the editable module, hands it the pipe
file descriptors, and calls its FIXED entry function (``_main``).

Usage:
    python nas_oracle_entry.py --module custom_nas_search.py \
        [--inputs-json-stdin] [--inputs-glob <glob> ...] [--entry _main] --
"""

from __future__ import annotations

import argparse
import glob as _glob
import importlib.util
import json
import os
import subprocess
import sys
import threading

_ENV_KEYS = ("cifar10", "cifar100", "imagenet16")
_FDS_ENV = "MLSBENCH_NAS_ORACLE_FDS"


def _parse_args(argv):
    if "--" in argv:
        split = argv.index("--")
        own, rest = argv[:split], argv[split + 1:]
    else:
        own, rest = argv, []
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--module", required=True)
    p.add_argument("--inputs-json-stdin", action="store_true")
    p.add_argument("--inputs-glob", action="append", default=[])
    p.add_argument("--entry", default="_main")
    p.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    return p.parse_args(own), rest


def _fatal(msg, code):
    print(f"[nas-oracle] FATAL: {msg}", file=sys.stderr, flush=True)
    return code


def _set_nondumpable():
    try:
        import ctypes
        libc = ctypes.CDLL(None, use_errno=True)
        PR_SET_DUMPABLE = 4
        return libc.prctl(PR_SET_DUMPABLE, 0, 0, 0, 0) == 0
    except Exception:
        return False


def _load_inputs(args):
    """Return {basename: text} of the staged blobs (stdin and/or disk)."""
    blobs = {}
    ephemeral = os.environ.get("MLSBENCH_EPHEMERAL_INPUTS") == "1"
    if args.inputs_json_stdin:
        raw = sys.stdin.read()
        data = json.loads(raw) if raw.strip() else {}
        if not isinstance(data, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in data.items()
        ):
            raise ValueError("expected a JSON object mapping path -> text on stdin")
        blobs.update({os.path.basename(k): v for k, v in data.items()})
    for pattern in args.inputs_glob:
        for path in sorted(_glob.glob(pattern)):
            if not os.path.isfile(path):
                continue
            with open(path, "r") as fh:
                blobs.setdefault(os.path.basename(path), fh.read())
            if ephemeral:
                os.remove(path)  # failure is fatal: the table would stay readable
    return blobs


def _serve(req_fd, resp_fd, val, budget, state):
    """Answer newline-delimited arch-string queries until EOF."""
    with os.fdopen(req_fd, "r", encoding="utf-8", newline="\n") as rin, \
            os.fdopen(resp_fd, "w", encoding="utf-8", newline="\n") as rout:
        for line in rin:
            arch = line.rstrip("\n")
            if state["count"] >= budget:
                state["refused"] += 1
                reply = f"BUDGET {budget}"
            else:
                state["count"] += 1
                acc = val.get(arch)
                reply = "KEYERR" if acc is None else f"OK {float(acc)!r}"
            try:
                rout.write(reply + "\n")
                rout.flush()
            except (BrokenPipeError, OSError):
                return


def _parent(args, module_argv):
    env_name = os.environ.get("ENV", "cifar10")
    seed = int(os.environ.get("SEED", 42))
    budget = int(os.environ.get("NAS_EPOCHS", 30))
    if env_name not in _ENV_KEYS:
        return _fatal(f"unknown ENV {env_name!r}", 2)
    nondumpable = _set_nondumpable()

    name = f"nb201_tables_{env_name}_s{seed}.json"
    try:
        blobs = _load_inputs(args)
    except (OSError, ValueError) as exc:
        return _fatal(f"could not load the staged validation table: {exc}", 3)
    payload = blobs.pop(name, None)
    blobs.clear()
    if payload is None:
        return _fatal(f"validation table {name} was not staged", 3)
    val = {str(k): float(v) for k, v in json.loads(payload)["val"].items()}
    del payload
    print(f"[nas-oracle] serving {len(val)} validation entries for {env_name} "
          f"out of process (budget={budget}, non-dumpable={nondumpable})",
          flush=True)

    req_r, req_w = os.pipe()
    resp_r, resp_w = os.pipe()
    child_env = dict(os.environ)
    child_env[_FDS_ENV] = f"{req_w},{resp_r}"
    cmd = [sys.executable, os.path.abspath(__file__), "--child",
           "--module", args.module, "--entry", args.entry, "--"] + module_argv
    child = subprocess.Popen(cmd, env=child_env, stdin=subprocess.DEVNULL,
                             close_fds=True, pass_fds=(req_w, resp_r))
    os.close(req_w)
    os.close(resp_r)

    state = {"count": 0, "refused": 0}
    server = threading.Thread(target=_serve,
                              args=(req_r, resp_w, val, budget, state),
                              daemon=True)
    server.start()
    rc = child.wait()
    server.join(timeout=5)
    print(f"ORACLE_QUERIES used={state['count']} budget={budget} "
          f"refused={state['refused']}", flush=True)
    return rc


def _child(args, module_argv):
    module_path = os.path.abspath(args.module)
    module_dir = os.path.dirname(module_path)
    module_name = os.path.splitext(os.path.basename(module_path))[0]
    sys.argv = [module_path] + list(module_argv)
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    getattr(module, args.entry)()
    return 0


def main():
    args, module_argv = _parse_args(sys.argv[1:])
    if args.child:
        return _child(args, module_argv)
    return _parent(args, module_argv)


if __name__ == "__main__":
    sys.exit(main())
