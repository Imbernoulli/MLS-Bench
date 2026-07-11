"""Shared utilities for the code-generation (codegen-*) MLS-Bench tasks.

Fixed, un-gameable evaluation path for EXECUTION-BASED (functional-correctness)
program synthesis:

  * A single FROZEN small code LM (Qwen2.5-Coder-1.5B-Instruct) is loaded once.
  * A FIXED set of Python function-synthesis problems is used. Each problem has a
    docstring/spec, an entry-point function name, a small set of VISIBLE example
    tests the agent's policy MAY use for candidate selection, and a DISJOINT set
    of HIDDEN unit tests used ONLY for scoring. The visible and hidden tests are
    different assertions, so a candidate that overfits to the visible tests but
    is wrong in general fails the hidden tests and earns nothing.
  * The agent controls ONLY an INFERENCE-TIME policy (sampling parameters,
    candidate selection/reranking, or prompt+postprocessing) — never the model
    weights, the problems, the hidden tests, or the scoring.
  * The metric is pass@1: the fraction of problems whose FINAL chosen program
    passes ALL hidden unit tests, executed SAFELY in a subprocess with a
    wall-clock timeout. A degenerate empty/constant program scores ~0.

The score therefore measures REAL functional correctness on held-out tests.
Because selection may use only the visible tests but scoring uses the hidden
tests, a solution overfit to the visible tests cannot get full credit, and a
naive single greedy sample is clearly beatable by good sampling + execute-filter.

The model + problem set are staged offline under $CG_MODEL / $CG_DATA so the
container runs with HF_HUB_OFFLINE=1.
"""
from __future__ import annotations

import ast
import importlib.util
import hashlib
import json
import math
import multiprocessing as mp
import numbers
import os
import random
import re
import signal
import sys
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths / model
# ---------------------------------------------------------------------------

def model_path() -> str:
    """FROZEN small code LM. All codegen-* tasks generate from this model."""
    return os.environ.get(
        "CG_MODEL", "/data/code-generation/models/Qwen2.5-Coder-1.5B-Instruct"
    )


MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
MODEL_REVISION = "357b899b4714bf46d935fb9911e8139b5b9efc29"
MODEL_MANIFEST = "model_manifest.json"
UNMATERIALIZED_PIN = "TO_BE_FILLED_BY_WORKER"
MODEL_FILES = (
    "config.json",
    "generation_config.json",
    "merges.txt",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)
EXPECTED_MODEL_FILE_PROOFS = {
    "config.json": {"bytes": 0, "sha256": UNMATERIALIZED_PIN},
    "generation_config.json": {"bytes": 0, "sha256": UNMATERIALIZED_PIN},
    "merges.txt": {"bytes": 0, "sha256": UNMATERIALIZED_PIN},
    "model.safetensors": {"bytes": 0, "sha256": UNMATERIALIZED_PIN},
    "tokenizer.json": {"bytes": 0, "sha256": UNMATERIALIZED_PIN},
    "tokenizer_config.json": {"bytes": 0, "sha256": UNMATERIALIZED_PIN},
    "vocab.json": {"bytes": 0, "sha256": UNMATERIALIZED_PIN},
}


def data_root() -> Path:
    return Path(os.environ.get("CG_DATA", "/data/code-generation/data"))


def set_seeds(seed: int = 42) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_surface(sol_path: str, attr: str):
    """Import the agent-editable callable `attr` from solution/<file>.py."""
    global _POLICY_GENERATION_BLOCKED
    p = Path(sol_path)
    spec = importlib.util.spec_from_file_location("agent_surface", str(p))
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(p.parent))
    previous = _POLICY_GENERATION_BLOCKED
    _POLICY_GENERATION_BLOCKED = True
    try:
        spec.loader.exec_module(mod)  # type: ignore
    finally:
        _POLICY_GENERATION_BLOCKED = previous
    fn = getattr(mod, attr, None)
    if not callable(fn):
        raise SystemExit(f"solution must define callable `{attr}(...)`")
    return fn


_POLICY_GENERATION_BLOCKED = False


def call_policy_without_generation(policy, *args, **kwargs):
    """Call an editable policy while enforcing the harness-owned model budget."""
    global _POLICY_GENERATION_BLOCKED
    previous = _POLICY_GENERATION_BLOCKED
    _POLICY_GENERATION_BLOCKED = True
    try:
        return policy(*args, **kwargs)
    finally:
        _POLICY_GENERATION_BLOCKED = previous


_TOKENIZER = None
_MODEL = None


def _validate_model_artifacts(path: Path) -> None:
    """Require every staged model file to match a source-pinned proof."""
    if not path.is_dir():
        raise SystemExit(f"frozen model directory is missing: {path}")
    manifest_path = path / MODEL_MANIFEST
    if not manifest_path.is_file():
        raise SystemExit(f"frozen model manifest is missing: {manifest_path}")
    if set(EXPECTED_MODEL_FILE_PROOFS) != set(MODEL_FILES) or any(
        type(proof.get("bytes")) is not int
        or proof["bytes"] <= 0
        or not re.fullmatch(r"[0-9a-f]{64}", str(proof.get("sha256", "")))
        for proof in EXPECTED_MODEL_FILE_PROOFS.values()
    ):
        raise SystemExit("canonical model artifact pins are not materialized")
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"frozen model manifest is unreadable: {exc}") from exc
    if (
        manifest.get("model_id") != MODEL_ID
        or manifest.get("revision") != MODEL_REVISION
        or set(manifest.get("files", {})) != set(MODEL_FILES)
    ):
        raise SystemExit("frozen model manifest does not match the pinned snapshot")
    for name in MODEL_FILES:
        file_path = path / name
        proof = manifest["files"].get(name, {})
        expected = EXPECTED_MODEL_FILE_PROOFS[name]
        if (
            not file_path.is_file()
            or not isinstance(proof, dict)
            or set(proof) != {"bytes", "sha256"}
            or type(proof.get("bytes")) is not int
            or proof != expected
            or file_path.stat().st_size != expected["bytes"]
            or _sha256(file_path) != expected["sha256"]
        ):
            raise SystemExit(f"frozen model asset proof failed: {name}")


def load_model():
    """Load (once) the frozen code model + tokenizer onto GPU, fp16."""
    global _TOKENIZER, _MODEL
    if _MODEL is not None:
        return _TOKENIZER, _MODEL
    path = Path(model_path())
    _validate_model_artifacts(path)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise SystemExit("code-generation verification requires CUDA")
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.float16
    ).eval()
    model = model.cuda()
    _TOKENIZER, _MODEL = tok, model
    return tok, model


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

EXPECTED_PROBLEMS = 257
EXPECTED_SEED = 42
DATASET_ID = "google-research-datasets/mbpp"
DATASET_REVISION = "4bb6404fdc6cacfda99d4ac4205087b89d32030c"
PROTOCOL = "mbpp-sanitized-reserved-v2"
EXPECTED_PROBLEMS_SHA256 = UNMATERIALIZED_PIN
_DATA_MANIFEST: dict | None = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assertion_fingerprint(assert_src: str, task_id: str) -> str:
    try:
        return ast.dump(ast.parse(assert_src), include_attributes=False)
    except SyntaxError as exc:
        raise SystemExit(f"problem {task_id} contains an invalid test assertion") from exc


def load_problems(n: int | None = None) -> list[dict]:
    """FIXED function-synthesis problem set.

    Each item:
      {"task_id": str,
       "entry_point": str,          # the function name the program must define
       "prompt": str,               # docstring/spec shown to the policy
       "visible_tests": [str, ...], # example assertions the policy MAY use
       "hidden_tests": [str, ...],  # DISJOINT assertions used ONLY for scoring
       "test_setup": str}           # any imports/helpers needed by the tests
    """
    global _DATA_MANIFEST
    root = data_root()
    problems_path = root / "problems.json"
    manifest_path = root / "manifest.json"
    if not problems_path.is_file() or not manifest_path.is_file():
        raise SystemExit("pinned MBPP problems/manifest are missing")
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"pinned MBPP manifest is unreadable: {exc}") from exc
    if manifest.get("protocol") != PROTOCOL:
        raise SystemExit("unexpected code-generation protocol")
    if manifest.get("dataset") != DATASET_ID:
        raise SystemExit("unexpected MBPP dataset id")
    if manifest.get("dataset_revision") != DATASET_REVISION:
        raise SystemExit("unexpected MBPP dataset revision")
    if manifest.get("model") != MODEL_ID or manifest.get("model_revision") != MODEL_REVISION:
        raise SystemExit("unexpected frozen model revision")
    if manifest.get("split") != "test" or int(manifest.get("count", -1)) != EXPECTED_PROBLEMS:
        raise SystemExit("unexpected MBPP split/count")
    if not re.fullmatch(r"[0-9a-f]{64}", EXPECTED_PROBLEMS_SHA256):
        raise SystemExit("canonical MBPP problem digest pin is not materialized")
    if (
        manifest.get("problems_sha256") != EXPECTED_PROBLEMS_SHA256
        or _sha256(problems_path) != EXPECTED_PROBLEMS_SHA256
    ):
        raise SystemExit("MBPP problem manifest hash mismatch")

    items = json.loads(problems_path.read_text())
    if not isinstance(items, list) or len(items) != EXPECTED_PROBLEMS:
        raise SystemExit(f"expected {EXPECTED_PROBLEMS} MBPP problems")
    seen = set()
    for item in items:
        required = {"task_id", "entry_point", "prompt", "visible_tests", "hidden_tests", "test_setup"}
        if not isinstance(item, dict) or set(item) != required:
            raise SystemExit("malformed MBPP problem record")
        task_id = str(item["task_id"])
        if task_id in seen:
            raise SystemExit(f"duplicate MBPP task id: {task_id}")
        seen.add(task_id)
        visible = item["visible_tests"]
        hidden = item["hidden_tests"]
        if not isinstance(visible, list) or not visible or not isinstance(hidden, list) or not hidden:
            raise SystemExit(f"problem {task_id} lacks disjoint evaluation tests")
        if set(visible) & set(hidden):
            raise SystemExit(f"problem {task_id} reuses a visible test for scoring")
        if not all(isinstance(value, str) and value.strip() for value in visible + hidden):
            raise SystemExit(f"problem {task_id} contains an invalid test assertion")
        visible_fingerprints = [_assertion_fingerprint(value, task_id) for value in visible]
        hidden_fingerprints = [_assertion_fingerprint(value, task_id) for value in hidden]
        if (
            len(set(visible_fingerprints)) != len(visible_fingerprints)
            or len(set(hidden_fingerprints)) != len(hidden_fingerprints)
            or set(visible_fingerprints) & set(hidden_fingerprints)
        ):
            raise SystemExit(f"problem {task_id} reuses an assertion across the protocol")
    if n is not None:
        if int(n) != EXPECTED_PROBLEMS:
            raise SystemExit(f"full protocol requires n={EXPECTED_PROBLEMS}")
        items = items[:n]
    _DATA_MANIFEST = manifest
    return items


def safe_problem(problem: dict) -> dict:
    """Return the exact policy-visible view, excluding scoring assertions."""
    return {
        "task_id": problem["task_id"],
        "entry_point": problem["entry_point"],
        "prompt": problem["prompt"],
        "visible_tests": list(problem["visible_tests"]),
        "test_setup": problem.get("test_setup", ""),
    }


def seal_private_data() -> None:
    """Remove the verifier's temporary assertion files after loading them.

    The task script copies the two private files into a verifier-owned temporary
    directory. Harnesses load the complete inventory before importing the
    editable policy, then call this function so policy code cannot reopen the
    scoring assertions through ``CG_DATA`` during evaluation.
    """
    root = data_root().resolve()
    tmp_root = Path(tempfile.gettempdir()).resolve()
    if os.environ.get("CG_PRIVATE_DATA_COPY") != "1" or not root.is_relative_to(tmp_root):
        raise RuntimeError("refusing to seal a non-temporary code-generation data root")
    for name in ("problems.json", "manifest.json"):
        path = root / name
        if not path.is_file():
            raise RuntimeError(f"private verifier data disappeared before sealing: {name}")
        path.unlink()
    os.environ.pop("CG_DATA", None)
    os.environ.pop("CG_PRIVATE_DATA_COPY", None)


def require_int(value, name: str, lower: int, upper: int) -> int:
    """Validate an agent-returned integer without silently truncating floats."""
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError(f"{name} must return an integer")
    result = int(value)
    if not lower <= result <= upper:
        raise ValueError(f"{name} must return an integer in [{lower},{upper}]")
    return result


def require_real(value, name: str) -> float:
    """Validate an agent-returned scalar without coercing strings or booleans."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must return a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must return a finite real number")
    return result


def emit_protocol(task_id: str, seed: int, problems: list[dict]) -> None:
    """Emit a pinned full-inventory proof before any per-problem records."""
    if not isinstance(task_id, str) or not re.fullmatch(r"codegen-[a-z0-9-]+", task_id):
        raise ValueError("invalid code-generation task identity")
    if seed != EXPECTED_SEED:
        raise ValueError(f"full protocol requires seed={EXPECTED_SEED}")
    if len(problems) != EXPECTED_PROBLEMS or _DATA_MANIFEST is None:
        raise RuntimeError("full pinned problem inventory was not loaded")
    digest = str(_DATA_MANIFEST.get("problems_sha256", ""))
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise RuntimeError("problem inventory digest is missing")
    print(
        "CG_PROTOCOL "
        f"task={task_id} protocol={PROTOCOL} n={EXPECTED_PROBLEMS} seed={EXPECTED_SEED} "
        f"model_revision={MODEL_REVISION} dataset_revision={DATASET_REVISION} "
        f"problems_sha256={digest}",
        flush=True,
    )


def emit_item(index: int, passed: bool, **diagnostics: int | bool) -> None:
    """Emit one strict, aggregate-recomputable record for a scored problem."""
    if type(index) is not int or not 1 <= index <= EXPECTED_PROBLEMS:
        raise RuntimeError("invalid proof index")
    fields = [f"CG_ITEM i={index}", f"passed={int(bool(passed))}"]
    for key, value in diagnostics.items():
        if not re.fullmatch(r"[a-z][a-z0-9_]*", key):
            raise RuntimeError(f"invalid proof key {key!r}")
        if type(value) is bool:
            value = int(value)
        if type(value) is not int or value < 0:
            raise RuntimeError(f"invalid proof value for {key}")
        fields.append(f"{key}={value}")
    print(" ".join(fields), flush=True)


def emit_progress(completed: int, passed: int) -> None:
    if completed % 20 or not 20 <= completed < EXPECTED_PROBLEMS:
        raise RuntimeError("invalid progress proof")
    if not 0 <= passed <= completed:
        raise RuntimeError("invalid progress aggregate")
    print(
        f"CG_PROGRESS completed={completed} total={EXPECTED_PROBLEMS} passed={passed}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Code extraction (FIXED default; a task may expose it as an editable surface)
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_code(text: str) -> str:
    """FIXED default extractor: take the first ```python ...``` fenced block if
    present, else the whole text. Trailing prose after the last fence is dropped.
    This is what an agent gets 'for free'; the postprocess task lets the agent
    replace it."""
    if not text:
        return ""
    m = _FENCE_RE.search(text)
    if m:
        return m.group(1)
    return text


# ---------------------------------------------------------------------------
# SAFE execution of generated code against unit tests
# ---------------------------------------------------------------------------

_UNSAFE_BUILTINS_SETUP = r"""
import builtins as _b
import os as _os
# Neutralize the most dangerous side-effecting entry points so a generated
# program cannot delete files, spawn processes, or open network sockets while
# still allowing ordinary pure-Python computation used by these problems.
for _n in ("system",):
    if hasattr(_os, _n):
        setattr(_os, _n, lambda *a, **k: (_ for _ in ()).throw(RuntimeError("blocked")))
"""


def _run_one(program: str, test_setup: str, tests: list[str]) -> tuple:
    """Exec the program + tests IN THIS PROCESS. Returns (n_passed, n_total, ok,
    err). Used inside the isolated grandchild fork (never in the CUDA parent)."""
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))
    except Exception:
        pass
    g: dict = {}
    n_total = len(tests)
    n_passed = 0
    try:
        exec(_UNSAFE_BUILTINS_SETUP, g)
        if test_setup:
            exec(test_setup, g)
        exec(program, g)
    except BaseException as e:  # syntax / import / top-level error
        return (0, n_total, False, f"program-error: {type(e).__name__}: {e}")
    err = ""
    for t in tests:
        try:
            exec(t, g)
            n_passed += 1
        except BaseException as e:  # noqa: BLE001
            if not err:
                err = f"{type(e).__name__}: {e}"
    ok = n_passed == n_total and n_total > 0
    return (n_passed, n_total, ok, err)


# --- Persistent executor server ------------------------------------------------
# We must NOT fork the main process once CUDA is initialized (forking a
# CUDA-initialized process deadlocks), and `spawn` is both slow (re-imports) and
# breaks when the entrypoint is `-c`/<stdin>. So we fork ONE long-lived executor
# server EARLY (before the model is loaded); that server has NO CUDA state and
# can safely fork a fresh grandchild per job for isolation + a hard timeout.

_EXECUTOR = None  # (conn, pid)


def start_executor() -> None:
    """Fork the executor server ONCE, BEFORE loading the model / touching CUDA.

    Harnesses call this at startup. Idempotent."""
    global _EXECUTOR
    if _EXECUTOR is not None:
        return
    import pickle

    parent_conn, child_conn = mp.Pipe()
    pid = os.fork()
    if pid == 0:
        # ---- executor server (child) ----
        parent_conn.close()
        try:
            _executor_serve(child_conn)
        finally:
            os._exit(0)
    # ---- main process (parent) ----
    child_conn.close()
    _EXECUTOR = (parent_conn, pid)


def _executor_serve(conn) -> None:
    """Loop forever: receive (program, test_setup, tests, timeout), run each in a
    fresh grandchild fork with a wall-clock timeout, send back the result."""
    while True:
        try:
            job = conn.recv()
        except EOFError:
            return
        if job is None:
            return
        program, test_setup, tests, timeout = job
        result = _fork_run(program, test_setup, tests, timeout)
        try:
            conn.send(result)
        except Exception:
            return


def _fork_run(program: str, test_setup: str, tests: list[str],
              timeout: float) -> tuple:
    """Run one job in a fresh grandchild fork (safe: server has no CUDA).
    Returns (n_passed, n_total, ok, err)."""
    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:
        # ---- grandchild: run the (untrusted) program, report over the pipe ----
        os.close(r)
        try:
            res = _run_one(program, test_setup, tests)
        except BaseException as e:  # noqa: BLE001
            res = (0, len(tests), False, f"exec-crash: {type(e).__name__}: {e}")
        try:
            import pickle
            os.write(w, pickle.dumps(res))
        except Exception:
            pass
        os.close(w)
        os._exit(0)
    # ---- server (parent of grandchild) ----
    os.close(w)
    import select
    import pickle

    buf = b""
    deadline = _now() + timeout
    got = False
    while True:
        remaining = deadline - _now()
        if remaining <= 0:
            break
        ready, _, _ = select.select([r], [], [], remaining)
        if not ready:
            break
        chunk = os.read(r, 65536)
        if not chunk:
            got = True
            break
        buf += chunk
    os.close(r)
    if not got or not buf:
        # timeout (or crash before writing) -> kill the grandchild
        try:
            os.kill(pid, signal.SIGKILL)
        except Exception:
            pass
        try:
            os.waitpid(pid, 0)
        except Exception:
            pass
        return (0, len(tests), False, "timeout")
    try:
        os.waitpid(pid, 0)
    except Exception:
        pass
    try:
        return pickle.loads(buf)
    except Exception:
        return (0, len(tests), False, "no-result")


def _now() -> float:
    import time
    return time.time()


def run_tests(
    program: str,
    tests: list[str],
    test_setup: str = "",
    timeout: float = 8.0,
) -> dict:
    """Execute `program` against `tests` SAFELY (isolated fork + wall-clock
    timeout) via the persistent executor server.

    Returns {"passed": int, "total": int, "ok": bool, "err": str}. ``ok`` means
    ALL tests passed. A crash / timeout / syntax error -> ok=False. This is the
    SAFE, un-gameable executor used both by the fixed harness scoring and by an
    agent's selection policy.
    """
    if not program or not program.strip():
        return {"passed": 0, "total": len(tests), "ok": False, "err": "empty-program"}
    if _EXECUTOR is None:
        start_executor()
    conn, _pid = _EXECUTOR
    try:
        conn.send((program, test_setup, list(tests), timeout))
        # allow a little slack over the job timeout for IPC
        if not conn.poll(timeout + 5.0):
            raise RuntimeError("executor IPC timed out")
        n_passed, n_total, ok, err = conn.recv()
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"executor IPC failed: {e}") from e
    if (
        type(n_passed) is not int
        or type(n_total) is not int
        or type(ok) is not bool
        or n_total != len(tests)
        or not 0 <= n_passed <= n_total
    ):
        raise RuntimeError("executor returned an inconsistent result")
    return {"passed": n_passed, "total": n_total, "ok": ok, "err": str(err)}


def passes_all(program: str, tests: list[str], test_setup: str = "",
               timeout: float = 8.0) -> bool:
    """True iff `program` passes ALL `tests` (safe isolated execution)."""
    return run_tests(program, tests, test_setup, timeout)["ok"]


# ---------------------------------------------------------------------------
# Generation engine (FIXED machinery; the POLICY chooses its parameters)
# ---------------------------------------------------------------------------

class GenConfig:
    """Sampling parameters the agent's policy returns for a problem.

    Attributes
    ----------
    prompt : str
        The full user message fed to the model (agent controls instruction text).
    n_samples : int
        How many candidate completions to draw. n_samples==1 with do_sample=False
        is deterministic greedy.
    do_sample : bool
        Whether to sample (True) or greedy-decode (False).
    temperature : float
        Sampling temperature (ignored if do_sample is False).
    top_p : float
        Nucleus sampling cutoff (ignored if do_sample is False).
    max_new_tokens : int
        Cap on generated tokens per candidate.
    """

    def __init__(
        self,
        prompt: str,
        n_samples: int = 1,
        do_sample: bool = False,
        temperature: float = 0.2,
        top_p: float = 0.95,
        max_new_tokens: int = 512,
    ) -> None:
        if not isinstance(prompt, str):
            raise TypeError("prompt must be text")
        if isinstance(n_samples, bool) or not isinstance(n_samples, numbers.Integral):
            raise TypeError("n_samples must be an integer")
        if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, numbers.Integral):
            raise TypeError("max_new_tokens must be an integer")
        if type(do_sample) is not bool:
            raise TypeError("do_sample must be a bool")
        if isinstance(temperature, bool) or not isinstance(temperature, numbers.Real):
            raise TypeError("temperature must be a real number")
        if isinstance(top_p, bool) or not isinstance(top_p, numbers.Real):
            raise TypeError("top_p must be a real number")
        self.prompt = prompt
        self.n_samples = int(n_samples)
        self.do_sample = do_sample
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.max_new_tokens = int(max_new_tokens)


# Hard caps keep all policies within the same inference-compute envelope.
_MAX_SAMPLES = 8
_MAX_NEW_TOKENS = 640


def _encode_prompt(tok, user_prompt: str):
    """Wrap the agent's prompt in the FIXED chat template and encode it."""
    text = tok.apply_chat_template(
        [{"role": "user", "content": user_prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    return tok(text, return_tensors="pt", add_special_tokens=False).input_ids


def generate(cfg: GenConfig, seed: int = 42) -> list[str]:
    """Generate candidate completions per `cfg`. FIXED machinery.

    Returns a list of RAW generated strings (length == cfg.n_samples, capped).
    Sampling temperature/top_p/n are the agent's choice; the model, chat
    template, and tokenizer are frozen. Greedy is deterministic; sampling is
    seeded so a run is reproducible.
    """
    if _POLICY_GENERATION_BLOCKED:
        raise RuntimeError("editable policy may not generate extra candidates")

    import torch

    set_seeds(seed)
    tok, model = load_model()
    device = next(model.parameters()).device

    if not isinstance(cfg, GenConfig):
        raise TypeError("generation policy must return common.GenConfig")
    if not math.isfinite(cfg.temperature) or not math.isfinite(cfg.top_p):
        raise ValueError("generation parameters must be finite")
    if not 1 <= cfg.n_samples <= _MAX_SAMPLES:
        raise ValueError(f"n_samples must be in [1,{_MAX_SAMPLES}]")
    if not 1 <= cfg.max_new_tokens <= _MAX_NEW_TOKENS:
        raise ValueError(f"max_new_tokens must be in [1,{_MAX_NEW_TOKENS}]")
    if not cfg.do_sample and cfg.n_samples != 1:
        raise ValueError("greedy decoding requires n_samples=1")
    if cfg.do_sample and not (0.0 < cfg.temperature <= 2.0 and 0.0 < cfg.top_p <= 1.0):
        raise ValueError("sampling temperature/top_p are outside the fixed envelope")
    if not isinstance(cfg.prompt, str) or not cfg.prompt.strip():
        raise ValueError("generation prompt must be non-empty text")

    n = cfg.n_samples
    max_new = cfg.max_new_tokens
    input_ids = _encode_prompt(tok, cfg.prompt).to(device)

    gen_kwargs = dict(
        max_new_tokens=max_new,
        num_return_sequences=n,
        pad_token_id=(tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id),
    )
    if cfg.do_sample:
        gen_kwargs.update(
            do_sample=True,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
        )
    else:
        # Greedy decoding has exactly one return sequence (validated above).
        gen_kwargs.update(do_sample=False)

    torch.manual_seed(seed)
    with torch.no_grad():
        out = model.generate(input_ids, **gen_kwargs)
    gen = out[:, input_ids.shape[1]:]
    texts = tok.batch_decode(gen, skip_special_tokens=True)
    if len(texts) != n or not all(isinstance(text, str) for text in texts):
        raise RuntimeError(f"generation returned {len(texts)} candidates, expected {n}")
    return list(texts)
