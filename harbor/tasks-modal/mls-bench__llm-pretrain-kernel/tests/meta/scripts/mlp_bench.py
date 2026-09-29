"""Fixed MLP-kernel throughput benchmark for llm-pretrain-kernel.

Run by gpt_345m.sh after training. Times one MLP forward+backward at the
training shape (x: (32*1024, n_embd) fp32 activations, fp32 weights, bf16
autocast, torch.compile -- as in the training loop) for

  * the submission: `fused_mlp_forward` from the submitted custom_pretrain.py;
  * the reference: the unmodified template's MLP,
    `h = F.gelu(x @ w_fc.t()); h @ w_proj.t()`.

Each timing runs in its own fresh process on the same GPU (the reference
process never imports the submission, so submission code cannot slow it
down), alternating reference / submission processes for several rounds.

Prints `TEST_METRICS: mlp_speedup=<t_reference / t_submission>` (median over
rounds): > 1 means the submitted MLP is faster than the unfused PyTorch
reference. A ratio of two timings on the same device is comparable across GPU
types, unlike wall-clock elapsed time. A submission whose MLP fails to run, or
returns the wrong shape, reports mlp_speedup=0.
"""
import argparse
import importlib.util
import json
import os
import statistics
import subprocess
import sys


def time_impl(impl, source, tokens, n_embd, warmup, rounds, iters):
    import torch
    import torch.nn.functional as F

    def reference_mlp(x, w_fc, w_proj):
        h = F.gelu(x @ w_fc.t())
        return h @ w_proj.t()

    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dev = "cuda"
    M, C = tokens, n_embd
    # Fresh inputs cycled per call, so no output can be reused across calls.
    xs = [torch.randn(M, C, device=dev, requires_grad=True) for _ in range(4)]
    gs = [torch.randn(M, C, device=dev, dtype=torch.bfloat16) for _ in range(4)]
    w_fc = torch.nn.Parameter(torch.randn(4 * C, C, device=dev) * 0.02)
    w_proj = torch.nn.Parameter(torch.randn(C, 4 * C, device=dev) * 0.02)

    if impl == "reference":
        fn = reference_mlp
    else:
        spec = importlib.util.spec_from_file_location("custom_pretrain", source)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        fn = mod.fused_mlp_forward
    fn = torch.compile(fn)

    def step(i):
        x, g = xs[i % len(xs)], gs[i % len(gs)]
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = fn(x, w_fc, w_proj)
        if tuple(out.shape) != (M, C):
            raise RuntimeError(f"fused_mlp_forward returned shape {tuple(out.shape)}, expected {(M, C)}")
        out.backward(g.to(out.dtype))
        x.grad = None
        w_fc.grad = None
        w_proj.grad = None

    for i in range(warmup):
        step(i)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    per_round = []
    for _ in range(rounds):
        start.record()
        for i in range(iters):
            step(i)
        end.record()
        torch.cuda.synchronize()
        per_round.append(start.elapsed_time(end) / iters)
    return statistics.median(per_round), torch.cuda.get_device_name()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="custom_pretrain.py")
    ap.add_argument("--impl", choices=["reference", "submission"], default=None,
                    help="internal: time one implementation in this process")
    ap.add_argument("--tokens", type=int, default=32 * 1024)
    ap.add_argument("--n-embd", type=int, default=int(os.environ.get("N_EMBD", 1024)))
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--procs", type=int, default=3,
                    help="reference/submission process pairs")
    args = ap.parse_args()

    if args.impl is not None:
        ms, gpu = time_impl(args.impl, os.path.abspath(args.source), args.tokens,
                            args.n_embd, args.warmup, args.rounds, args.iters)
        print("MLP_BENCH_RESULT " + json.dumps({"ms": ms, "gpu": gpu}), flush=True)
        return 0

    env = dict(os.environ)
    # One GPU is enough; keep the first visible one.
    vis = env.get("CUDA_VISIBLE_DEVICES", "")
    if vis:
        env["CUDA_VISIBLE_DEVICES"] = vis.split(",")[0]
    times = {"reference": [], "submission": []}
    gpu = "?"
    for p in range(args.procs):
        order = ["reference", "submission"] if p % 2 == 0 else ["submission", "reference"]
        for impl in order:
            cmd = [sys.executable, os.path.abspath(__file__), "--impl", impl,
                   "--source", os.path.abspath(args.source),
                   "--tokens", str(args.tokens), "--n-embd", str(args.n_embd),
                   "--warmup", str(args.warmup), "--rounds", str(args.rounds),
                   "--iters", str(args.iters)]
            r = subprocess.run(cmd, env=env, capture_output=True, text=True)
            res = None
            for line in r.stdout.splitlines():
                if line.startswith("MLP_BENCH_RESULT "):
                    res = json.loads(line[len("MLP_BENCH_RESULT "):])
            if r.returncode != 0 or res is None:
                sys.stderr.write(r.stdout[-3000:] + r.stderr[-6000:])
                print(f"MLP_BENCH: {impl} failed (rc={r.returncode})", flush=True)
                if impl == "submission":
                    print("TEST_METRICS: mlp_speedup=0.0", flush=True)
                    return 0
                # Reference failure is an infrastructure problem, not the
                # submission's: report no metric rather than a wrong one.
                return 1
            times[impl].append(res["ms"])
            gpu = res["gpu"]

    t_ref = statistics.median(times["reference"])
    t_sub = statistics.median(times["submission"])
    print(
        f"MLP_BENCH: tokens={args.tokens} n_embd={args.n_embd} gpu={gpu} "
        f"reference_ms={[round(t, 3) for t in times['reference']]} "
        f"submission_ms={[round(t, 3) for t in times['submission']]}",
        flush=True,
    )
    print(f"TEST_METRICS: mlp_speedup={t_ref / t_sub:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
