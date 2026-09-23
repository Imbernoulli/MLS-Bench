"""Parameter budget check for llm-pretrain-linear-attention (standalone).

Run by tools.py before training: python /workspace/_task/budget_check.py
Imports each baseline, instantiates GPT model, counts params, and
asserts the agent's model doesn't exceed 1.05x the largest baseline.
Then probes the agent's model on a GPU and rejects sequence mixing whose cost
grows quadratically with sequence length (see check_subquadratic).
"""
import importlib.util
import json
import os
import sys
import tempfile

# The complexity probe below runs the model eagerly under a TorchDispatchMode.
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import torch

TASK_DIR = "/workspace/_task"
WORKSPACE_FILE = "/workspace/nanoGPT/custom_pretrain.py"


def load_module(path, name=None):
    name = name or f"_mod_{hash(path)}"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def apply_ops(lines, ops, filename):
    result = list(lines)
    sorted_ops = sorted(
        [o for o in ops if o.get("file") == filename],
        key=lambda o: -o.get("start_line", o.get("after_line", 0)),
    )
    for op in sorted_ops:
        if op["op"] == "replace":
            s, e = op["start_line"] - 1, op["end_line"]
            result[s:e] = op["content"].splitlines()
        elif op["op"] == "insert":
            after = op["after_line"]
            result[after:after] = op["content"].splitlines()
        elif op["op"] == "delete":
            s, e = op["start_line"] - 1, op["end_line"]
            del result[s:e]
    return result


def count_params(module_path):
    """Import module, instantiate GPT with small config, return param count."""
    mod = load_module(module_path, f"_check_{id(module_path)}")
    config = mod.GPTConfig(
        block_size=1024,
        vocab_size=50304,
        n_layer=12,
        n_head=12,
        n_embd=768,
        dropout=0.0,
        bias=False,
    )
    model = mod.GPT(config)
    return sum(p.numel() for p in model.parameters())


# -- Get template content --
mid_edit = load_module(os.path.join(TASK_DIR, "edits", "mid_edit.py"), "_mid_edit")
config = json.loads(open(os.path.join(TASK_DIR, "config.json")).read())
editable_file = None
for f in config.get("files", []):
    if f.get("edit"):
        editable_file = f["filename"]
        break

template_content = None
for op in mid_edit.OPS:
    if op.get("op") == "create" and op.get("file") == editable_file:
        template_content = op["content"]
        break

assert template_content, f"No template found for {editable_file}"
template_lines = template_content.splitlines()

# -- Count params for each baseline --
baseline_params = {}
for bl_name, bl_cfg in config.get("baselines", {}).items():
    edit_path = os.path.join(TASK_DIR, bl_cfg["edit_ops"])
    if not os.path.exists(edit_path):
        continue
    bl_mod = load_module(edit_path, f"_bl_{bl_name}")
    ops = getattr(bl_mod, "OPS", [])
    modified_lines = apply_ops(template_lines, ops, editable_file)
    modified_code = "\n".join(modified_lines)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(modified_code)
        tmp_path = f.name
    try:
        params = count_params(tmp_path)
        baseline_params[bl_name] = params
        print(f"  baseline {bl_name}: {params} params")
    except Exception as e:
        print(f"  baseline {bl_name}: ERROR ({e})")
    finally:
        os.unlink(tmp_path)

if not baseline_params:
    print("WARNING: no baselines could be evaluated, skipping budget check")
    sys.exit(0)

max_baseline = max(baseline_params.values())
max_name = max(baseline_params, key=baseline_params.get)
budget = int(max_baseline * 1.05)

# -- Count params for agent's version --
agent_params = count_params(WORKSPACE_FILE)
print(f"\n  agent model: {agent_params} params")
print(f"  budget: {budget} (1.05 x {max_name}={max_baseline})")

if agent_params > budget:
    print(f"\nFAILED: {agent_params} > {budget}", file=sys.stderr)
    sys.exit(1)

# -- Sequence-mixing complexity probe --
# The task requires a linear / subquadratic mechanism, so the agent's model is
# also run forward (train mode, bf16 autocast, GPT-2 Medium shapes) at three
# sequence lengths with the same token count, and rejected if any Block
#   (a) materializes a tensor with two dimensions equal to the sequence length
#       (a T x T score / mask / decay matrix; chunk x chunk blocks are fine),
#   (b) spends matmul / SDPA FLOPs per token that grow linearly with the
#       sequence length (the signature of O(T^2) mixing, e.g. softmax attention
#       via scaled_dot_product_attention or chunked over queries), or whose
#       largest tensor grows with the sequence length at a fixed token count, or
#   (c) launches one of FLA's quadratic Triton kernels (softmax attention and
#       the O(T^2) "parallel" forms), which the two checks above cannot see.
PROBE_LENS = (192, 384, 768)   # batch 4 / 2 / 1: 768 tokens each
PROBE_TOKENS = 768
PROBE_TXT_MIN_T = 384          # the T x T check runs at the two longer lengths
GROWTH_RATIO = 1.7             # per-token cost growth 384->768 vs 192->384: 2 for O(T^2), 1 for O(T log T)
FLOP_TOL = 0.001               # ignore growth below 0.1 % of the per-token FLOPs
MAXNUMEL_RATIO = 1.7           # largest tensor 384->768: 2x when it is B x H x T x T
QUADRATIC_FLA_PREFIXES = (
    "fla.ops.attn", "fla.ops.forgetting_attn", "fla.ops.path_attn",
    "fla.ops.deltaformer", "fla.ops.nsa", "fla.ops.moba",
)


def _is_quadratic_kernel(module_name):
    if not module_name.startswith("fla.ops."):
        return False
    return (module_name.startswith(QUADRATIC_FLA_PREFIXES)
            or module_name.rsplit(".", 1)[-1] == "parallel")


def check_subquadratic(module_path):
    from torch.utils._python_dispatch import TorchDispatchMode
    from torch.utils._pytree import tree_flatten
    from torch.utils.flop_counter import flop_registry

    if not torch.cuda.is_available():
        return ["the complexity probe needs a CUDA device and none is visible"]

    state = {"block": None, "T": None, "active": False}
    stats = {}        # (scope, T) -> [flops, largest tensor numel]
    violations = []

    def note(msg):
        if msg not in violations:
            violations.append(msg)

    try:
        import triton.runtime.jit as _tjit
        _orig_run = _tjit.JITFunction.run

        def _run(self, *args, **kwargs):
            if state["active"]:
                fn = getattr(self, "fn", None)
                modname = getattr(fn, "__module__", "") or ""
                if _is_quadratic_kernel(modname):
                    note(f"quadratic FLA Triton kernel {modname}."
                         f"{getattr(fn, '__name__', '?')} launched")
            return _orig_run(self, *args, **kwargs)

        _tjit.JITFunction.run = _run
    except Exception as e:  # triton absent: nothing to hook
        print(f"  (triton hook unavailable: {e})")

    # The probe needs a kernel that runs, not the fastest one: take the first
    # autotune config that compiles, run once, and skip benchmarking the rest
    # (full FLA autotuning alone can take minutes).
    try:
        import triton.runtime.autotuner as _tat
        _orig_bench, _orig_tune = _tat.Autotuner._bench, _tat.Autotuner.run

        def _bench(self, *args, config, **meta):
            if getattr(self, "_probe_picked", False):
                return [float("inf")] * 3
            saved = self.do_bench
            self.do_bench = lambda kernel_call, quantiles: (kernel_call(), [0.0] * 3)[1]
            try:
                res = _orig_bench(self, *args, config=config, **meta)
            finally:
                self.do_bench = saved
            self._probe_picked = res[0] != float("inf")
            return res

        def _tune(self, *args, **kwargs):
            self._probe_picked = False
            return _orig_tune(self, *args, **kwargs)

        _tat.Autotuner._bench, _tat.Autotuner.run = _bench, _tune
    except Exception as e:
        print(f"  (autotune shortcut unavailable: {e})")

    class _Probe(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            out = func(*args, **kwargs)
            T = state["T"]
            scope = state["block"]
            if scope is None:
                return out
            flops = 0
            packet = func.overloadpacket
            if packet in flop_registry:
                try:
                    flops = int(flop_registry[packet](*args, **kwargs, out_val=out))
                except Exception:
                    flops = 0
            numel = 0
            for t in tree_flatten(out)[0]:
                if not isinstance(t, torch.Tensor):
                    continue
                numel = max(numel, t.numel())
                if T >= PROBE_TXT_MIN_T and sum(1 for s in t.shape if s == T) >= 2:
                    note(f"block {scope}: {func} produced a {tuple(t.shape)} tensor "
                         f"(two dims equal to the sequence length {T})")
            for key in (scope, "all"):
                acc = stats.setdefault((key, T), [0, 0])
                acc[0] += flops
                acc[1] = max(acc[1], numel)
            return out

    mod = load_module(module_path, "_probe_agent")
    torch.manual_seed(0)
    cfg = mod.GPTConfig(block_size=1024, vocab_size=50304, n_layer=24, n_head=16,
                        n_embd=1024, dropout=0.0, bias=False)
    device = "cuda"
    model = mod.GPT(cfg).to(device)
    model.train()
    for i, blk in enumerate(model.transformer.h):
        def _pre(_m, _a, i=i):
            state["block"] = i
        def _post(_m, _a, _o):
            state["block"] = None
        blk.register_forward_pre_hook(_pre)
        blk.register_forward_hook(_post)

    for T in PROBE_LENS:
        B = PROBE_TOKENS // T
        idx = torch.randint(0, cfg.vocab_size, (B, T), device=device)
        tgt = torch.randint(0, cfg.vocab_size, (B, T), device=device)
        state["T"] = T
        state["active"] = True
        try:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16), _Probe():
                _, loss = model(idx, tgt)
            torch.cuda.synchronize()
        finally:
            state["active"] = False
            state["block"] = None
        if loss is None or not torch.isfinite(loss).item():
            note(f"forward at T={T} returned a non-finite loss")
        del loss

    t1, t2, t3 = PROBE_LENS
    scopes = sorted({k for k, _ in stats}, key=lambda s: (s == "all", s))
    for scope in scopes:
        where = "model" if scope == "all" else f"block {scope}"
        c1, c2, c3 = (stats.get((scope, T), [0, 0])[0] / PROBE_TOKENS for T in PROBE_LENS)
        d12, d23 = c2 - c1, c3 - c2
        if c1 > 0 and d23 > FLOP_TOL * c1 and d23 > GROWTH_RATIO * max(d12, 0.0):
            note(f"{where}: matmul/SDPA FLOPs per token grow with sequence length "
                 f"({c1:.4g} @T={t1}, {c2:.4g} @T={t2}, {c3:.4g} @T={t3}) "
                 f"-- quadratic sequence mixing")
        m2, m3 = (stats.get((scope, T), [0, 0])[1] for T in (t2, t3))
        if m2 > 0 and m3 >= MAXNUMEL_RATIO * m2:
            note(f"{where}: largest tensor grows from {m2} (T={t2}) to {m3} (T={t3}) "
                 f"elements at a fixed token count -- quadratic sequence mixing")
    per_tok = [stats.get(("all", T), [0, 0]) for T in PROBE_LENS]
    print("  complexity probe (all blocks): "
          + ", ".join(f"T={T}: {f / PROBE_TOKENS:.4g} FLOPs/token, largest tensor {n}"
                      for T, (f, n) in zip(PROBE_LENS, per_tok)))
    return violations


try:
    complexity_violations = check_subquadratic(WORKSPACE_FILE)
except Exception as e:
    complexity_violations = [f"the complexity probe could not run the model: "
                             f"{type(e).__name__}: {e}"]
if complexity_violations:
    print("\nFAILED: the sequence-mixing mechanism is not subquadratic in sequence length:",
          file=sys.stderr)
    for v in complexity_violations[:20]:
        print(f"  - {v}", file=sys.stderr)
    sys.exit(1)
print("  complexity probe: subquadratic")

print("\nPASSED")
