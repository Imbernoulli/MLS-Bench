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
  2. mark this process non-dumpable (``prctl(PR_SET_DUMPABLE, 0)``);
  3. draw this run's hidden RELABELING from OS entropy: the four non-``none``
     operation names the search uses are mapped to the NAS-Bench-201
     operations through a secret random permutation (``none`` stays
     ``none``), so an index list or cell string names a different
     benchmark cell in every run and a known-good cell from the literature
     means nothing without queries;
  4. start the search as a FRESH interpreter (the table is never sent to it)
     connected to this process by two pipes, with stdin closed. When this
     wrapper runs as root (the Harbor verifier) the search is dropped to an
     unprivileged user with no_new_privs, after the verifier-only files
     (``/tests``: tables, parser, scorer; the base image's full benchmark
     pickle) and the verifier log directory have been made unreachable for
     that user; the search refuses to start if any of them is still
     reachable;
  5. answer its queries here: every request counts, the budget
     (NAS_EPOCHS, default 30) is enforced here, and a request past the
     budget is refused (the search's BenchmarkAPI raises
     BudgetExceededError);
  6. relay the search's output (stdout+stderr) to the log line by line,
     sanitized to printable ASCII; a line of its own that starts with
     ``FINAL_ARCH`` is recorded as its choice (last one wins) and relayed
     with a ``[search]`` prefix, so the search can never write a line the
     parser accepts;
  7. after the search exits (rc 0), print the authoritative
     ``FINAL_ARCH arch=<benchmark cell> queries=<n>`` line — the search's
     choice mapped through the hidden relabeling — and exit with the
     search's return code.

``run_sandboxed`` (the same unprivileged launcher) is also used by
``budget_check.py``, which imports the agent's module only inside it.

Usage:
    python -I -B nas_oracle_entry.py --module custom_nas_search.py \
        [--inputs-json-stdin] [--inputs-glob <glob> ...] [--entry _main] --
"""

from __future__ import annotations

import argparse
import glob as _glob
import json
import os
import random
import re
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import threading

_ENV_KEYS = ("cifar10", "cifar100", "imagenet16")
_FDS_ENV = "MLSBENCH_NAS_ORACLE_FDS"

# NAS-Bench-201 operation vocabulary (the search's OP_NAMES) and cell format.
_OPS = ("skip_connect", "none", "nor_conv_3x3", "nor_conv_1x1", "avg_pool_3x3")
_ARCH_RE = re.compile(
    r"\|(\w+)~0\|\+\|(\w+)~0\|(\w+)~1\|\+\|(\w+)~0\|(\w+)~1\|(\w+)~2\|")
_FINAL_RE = re.compile(r"FINAL_ARCH\s+arch=(\S+)")

UNPRIVILEGED_UID = 65534  # nobody
# Made unreachable for UNPRIVILEGED_UID before any untrusted code starts
# (root only): (path, permission bits cleared, access the sandboxed process
# must NOT have). Missing paths are skipped (native runs have none of them).
_LOCKDOWN = (
    ("/tests", 0o077, os.X_OK),           # verifier-only: tables, parser, scorer
    ("/logs", 0o022, os.W_OK),
    ("/logs/verifier", 0o022, os.W_OK),   # Harbor creates it 0777; logs + reward
    ("/opt/mlsbench/original/naslib/naslib/data/nb201_all.pickle",
     0o077, os.R_OK),                     # the base image's full benchmark pickle
)
_SANDBOX_ENV_KEYS = ("PATH", "LANG", "LC_ALL", "LC_CTYPE", "PYTHONPATH",
                     "ENV", "SEED", "NAS_EPOCHS")
_SANDBOX_ENV_PREFIXES = ("OMP_", "MKL_", "OPENBLAS_", "NUMEXPR_", "VECLIB_",
                         "BLIS_")

# Runs first in every sandboxed interpreter (trusted until it execs the
# payload): refuse to run untrusted code while a locked-down path is still
# reachable, or while still privileged.
_SANDBOX_BOOT = r"""
import json, os, sys
_cfg = json.loads(sys.argv[1])
_entry = None
for _entry in _cfg["deny"]:
    if os.access(_entry[0], _entry[1]):
        sys.stderr.write("[nas-sandbox] FATAL: %s is reachable (access mode %d) "
                         "for uid %d; refusing to run the search\n"
                         % (_entry[0], _entry[1], os.getuid()))
        sys.stderr.flush()
        os._exit(3)
if _cfg["unprivileged"] and 0 in (os.getuid(), os.geteuid()):
    sys.stderr.write("[nas-sandbox] FATAL: still privileged\n")
    sys.stderr.flush()
    os._exit(3)
_src = sys.argv[2]
sys.argv = ["-c"] + sys.argv[3:]
del _cfg, _entry
exec(compile(_src, "<sandboxed>", "exec"), {"__name__": "__main__"})
"""

# Payload of the search process: import the editable module and call its
# FIXED entry function.
_SEARCH_PAYLOAD = r"""
import importlib.util, os, sys
module_path, entry = sys.argv[1], sys.argv[2]
module_dir = os.path.dirname(module_path)
module_name = os.path.splitext(os.path.basename(module_path))[0]
sys.argv = [module_path] + sys.argv[3:]
if module_dir not in sys.path:
    sys.path.insert(0, module_dir)
spec = importlib.util.spec_from_file_location(module_name, module_path)
module = importlib.util.module_from_spec(spec)
sys.modules[module_name] = module
spec.loader.exec_module(module)
getattr(module, entry)()
"""


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
    return p.parse_args(own), rest


def _fatal(msg, code):
    print(f"[nas-oracle] FATAL: {msg}", file=sys.stderr, flush=True)
    return code


def _libc():
    import ctypes
    return ctypes.CDLL(None, use_errno=True)


def _set_nondumpable():
    try:
        PR_SET_DUMPABLE = 4
        return _libc().prctl(PR_SET_DUMPABLE, 0, 0, 0, 0) == 0
    except Exception:
        return False


# ── hidden relabeling ─────────────────────────────────────────────────────
def draw_relabeling(rng=None):
    """Secret per-run map: search op name -> NAS-Bench-201 op name.

    ``none`` is fixed (it removes an edge, and the fixed ``is_valid_arch``
    helper relies on index 1 being ``none``); the other four operations are
    permuted uniformly at random.
    """
    rng = rng or random.SystemRandom()
    movable = [op for op in _OPS if op != "none"]
    image = list(movable)
    rng.shuffle(image)
    mapping = dict(zip(movable, image))
    mapping["none"] = "none"
    return mapping


def relabel(arch, mapping):
    """Map a search-space cell string to its benchmark cell (None if malformed)."""
    m = _ARCH_RE.fullmatch(arch)
    if m is None:
        return None
    ops = [mapping.get(op) for op in m.groups()]
    if None in ops:
        return None
    return "|{}~0|+|{}~0|{}~1|+|{}~0|{}~1|{}~2|".format(*ops)


# ── unprivileged sandbox ──────────────────────────────────────────────────
def lock_down():
    """Root only: remove the lockdown paths' permission bits for others."""
    if os.geteuid() != 0:
        return
    for path, clear, _mode in _LOCKDOWN:
        try:
            st = os.stat(path)
            os.chmod(path, stat.S_IMODE(st.st_mode) & ~clear)
        except OSError:
            pass  # missing, or read-only: the sandbox boot check decides


def _sandbox_preexec():
    parent = os.getpid()

    def _fn():
        os.setsid()
        if os.geteuid() == 0:
            os.setgroups([])
            os.setgid(UNPRIVILEGED_UID)
            os.setuid(UNPRIVILEGED_UID)
        libc = _libc()
        PR_SET_PDEATHSIG, PR_SET_NO_NEW_PRIVS = 1, 38
        # After setuid (a credential change clears the death signal): the
        # sandbox dies with this process, e.g. at the eval deadline.
        libc.prctl(PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0)
        if os.getppid() != parent:
            os._exit(1)
        if libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
            raise OSError("prctl(PR_SET_NO_NEW_PRIVS) failed")
        os.umask(0o077)
    return _fn


def _kill_group(pid):
    try:
        os.killpg(pid, signal.SIGKILL)
    except OSError:
        pass


def start_sandboxed(payload, payload_argv, extra_env=None, **popen_kw):
    """Start ``payload`` (Python source) in a fresh, unprivileged interpreter.

    Returns ``(proc, home)``; the caller reaps ``proc`` and removes ``home``
    (the sandbox's private HOME/TMPDIR) with ``finish_sandboxed``.
    """
    unprivileged = os.geteuid() == 0
    lock_down()
    deny = ([[p, mode] for p, _clear, mode in _LOCKDOWN if os.path.lexists(p)]
            if unprivileged else [])
    home = tempfile.mkdtemp(prefix="nas_sandbox_", dir="/tmp")
    os.chmod(home, 0o700)
    if unprivileged:
        os.chown(home, UNPRIVILEGED_UID, UNPRIVILEGED_UID)
    env = {k: v for k, v in os.environ.items()
           if k in _SANDBOX_ENV_KEYS or k.startswith(_SANDBOX_ENV_PREFIXES)}
    env["HOME"] = env["TMPDIR"] = home
    if not unprivileged and "OUTPUT_DIR" in os.environ:
        env["OUTPUT_DIR"] = os.environ["OUTPUT_DIR"]
    else:
        env["OUTPUT_DIR"] = os.path.join(home, "output")
    env.update(extra_env or {})
    cfg = json.dumps({"deny": deny, "unprivileged": unprivileged})
    cmd = [sys.executable, "-B", "-s", "-c", _SANDBOX_BOOT, cfg, payload]
    cmd += list(payload_argv)
    proc = subprocess.Popen(cmd, env=env, preexec_fn=_sandbox_preexec(),
                            **popen_kw)
    return proc, home


def finish_sandboxed(proc, home):
    _kill_group(proc.pid)  # reap anything the sandboxed code left behind
    shutil.rmtree(home, ignore_errors=True)


def run_sandboxed(payload, payload_argv, timeout):
    """Run ``payload`` sandboxed to completion; return (rc, output text)."""
    proc, home = start_sandboxed(payload, payload_argv,
                                 stdin=subprocess.DEVNULL,
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, close_fds=True)
    try:
        out, _ = proc.communicate(timeout=timeout)
        rc = proc.returncode
    except subprocess.TimeoutExpired:
        _kill_group(proc.pid)
        out, _ = proc.communicate()
        rc = 124
    finally:
        finish_sandboxed(proc, home)
    return rc, out.decode("utf-8", "replace")


# ── oracle ────────────────────────────────────────────────────────────────
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


def _serve(req_fd, resp_fd, val, budget, state, mapping):
    """Answer newline-delimited arch-string queries until EOF."""
    with os.fdopen(req_fd, "r", encoding="utf-8", errors="replace",
                   newline="\n") as rin, \
            os.fdopen(resp_fd, "w", encoding="utf-8", newline="\n") as rout:
        for line in rin:
            arch = line.rstrip("\n")
            if state["count"] >= budget:
                state["refused"] += 1
                reply = f"BUDGET {budget}"
            else:
                state["count"] += 1
                real = relabel(arch, mapping)
                acc = None if real is None else val.get(real)
                reply = "KEYERR" if acc is None else f"OK {float(acc)!r}"
            try:
                rout.write(reply + "\n")
                rout.flush()
            except (BrokenPipeError, OSError):
                return


def _sanitize(raw):
    text = raw.rstrip(b"\n").decode("utf-8", "replace")
    return "".join(ch if " " <= ch <= "~" else "?" for ch in text)


def _relay(stream, state, lock):
    """Copy the search's output to our stdout; record its FINAL_ARCH choice."""
    for raw in iter(lambda: stream.readline(65536), b""):
        line = _sanitize(raw)
        if line.lstrip(" ").startswith("FINAL_ARCH"):
            m = _FINAL_RE.match(line.lstrip(" "))
            if m:
                state["final"] = m.group(1)
            line = "[search] " + line
        with lock:
            if not state["closed"]:
                sys.stdout.write(line + "\n")
                sys.stdout.flush()


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
    mapping = draw_relabeling()

    req_r, req_w = os.pipe()
    resp_r, resp_w = os.pipe()
    home = None
    try:
        child, home = start_sandboxed(
            _SEARCH_PAYLOAD,
            [os.path.abspath(args.module), args.entry] + list(module_argv),
            extra_env={_FDS_ENV: f"{req_w},{resp_r}"},
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, close_fds=True,
            pass_fds=(req_w, resp_r))
    except (OSError, subprocess.SubprocessError) as exc:
        if home:
            shutil.rmtree(home, ignore_errors=True)
        return _fatal(f"could not start the search process: {exc}", 3)
    finally:
        os.close(req_w)
        os.close(resp_r)
    print(f"[nas-oracle] serving {len(val)} validation entries for {env_name} "
          f"out of process (budget={budget}, non-dumpable={nondumpable}, "
          f"search uid={'%d' % UNPRIVILEGED_UID if os.geteuid() == 0 else 'same'}, "
          f"hidden op relabeling drawn)", flush=True)

    state = {"count": 0, "refused": 0, "final": None, "closed": False}
    lock = threading.Lock()
    server = threading.Thread(
        target=_serve, args=(req_r, resp_w, val, budget, state, mapping),
        daemon=True)
    relay = threading.Thread(target=_relay, args=(child.stdout, state, lock),
                             daemon=True)
    server.start()
    relay.start()
    rc = child.wait()
    finish_sandboxed(child, home)
    relay.join(timeout=5)
    server.join(timeout=5)
    with lock:
        state["closed"] = True
        final = state["final"]
        count, refused = state["count"], state["refused"]
    print(f"ORACLE_QUERIES used={count} budget={budget} refused={refused}",
          flush=True)
    if rc != 0:
        return _fatal(f"the search exited with rc={rc}; no architecture scored",
                      rc if rc > 0 else 1)
    if final is None:
        return _fatal("the search reported no FINAL_ARCH", 1)
    real = relabel(final, mapping)
    if real is None:
        return _fatal(f"FINAL_ARCH {final!r} is not a NAS-Bench-201 cell", 1)
    print(f"FINAL_ARCH arch={real} queries={count}", flush=True)
    return 0


def main():
    args, module_argv = _parse_args(sys.argv[1:])
    return _parent(args, module_argv)


if __name__ == "__main__":
    sys.exit(main())
