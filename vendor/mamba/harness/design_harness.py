#!/usr/bin/env python3
"""Generic paper-scale selective-copy harness for Mamba design surfaces.

The SAME TinyMamba selective-copy pipeline as selective_copy_harness.py, but the
agent's design surface is selected by --hook. Every hook not named on the command
line stays at the standard-Mamba default (see vendor/mamba/common.py), so a task
freezes all-but-one surface and grades exactly the one the agent designed.

--hook chooses which trusted builder interprets the literal --solution plan and
where the resulting hook is wired into the block:
    couple_bc      couple_bc(block, B, C_lowrank, b, l)-> C (b, d_state, l)
    conv_act       conv_act(x)                         -> activated x
    finalize_dt    finalize_dt(block, dt)              -> positive/unchanged dt
    gate           gate(y, z)                          -> gated y
    compute_A      compute_A(A_log)                    -> A  (d_inner, d_state)
    make_norm      make_norm(d_model)                  -> nn.Module
    residual_step  residual_step(h, block_out)         -> h

Emits protocol-bound ``POOL_LOADED`` and ``MAMBA_TRAIN_COMPLETE`` proofs followed
by one ``MAMBA_COPY_METRICS`` record. All three bind the full run configuration
and parameter count.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from common import (
    format_metric_line,
    get_device,
    load_surface_hook_and_choice,
    train_and_eval,
)

_TASK_BY_HOOK = {
    "compute_A": "mamba-a-stability",
    "couple_bc": "mamba-bc-coupling",
    "conv_act": "mamba-conv-act",
    "finalize_dt": "mamba-delta-softplus",
    "gate": "mamba-gating",
    "make_norm": "mamba-normalization",
    "residual_step": "mamba-residual",
}
_HOOKS = set(_TASK_BY_HOOK)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--solution", required=True)
    ap.add_argument("--hook", required=True, choices=sorted(_HOOKS))
    ap.add_argument("--fn", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--label", default="paper_e1")
    ap.add_argument("--L", type=int, default=4096)
    ap.add_argument("--M", type=int, default=16)
    ap.add_argument("--A", type=int, default=16)
    ap.add_argument("--d_model", type=int, default=64)
    ap.add_argument("--d_state", type=int, default=16)
    ap.add_argument("--n_layer", type=int, default=2)
    ap.add_argument("--expand", type=int, default=2)
    ap.add_argument("--d_conv", type=int, default=4)
    ap.add_argument("--steps", type=int, default=400000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--protocol", default="mamba_selective_copy_paper_e1_v1")
    ap.add_argument("--optimizer", choices=["adam", "adamw"], default="adam")
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--eval-batches", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = get_device()
    if args.fn not in {None, args.hook}:
        raise ValueError("--fn must match the fixed task hook")
    fn, surface_choice = load_surface_hook_and_choice(
        Path(args.solution), args.hook
    )

    # The scan path is always full selectivity (so the hook under test, not a
    # crippled parameterization, decides the score); the agent's hook is threaded
    # in via the matching keyword.
    kwargs = {k: None for k in _HOOKS}
    kwargs[args.hook] = fn
    from common import default_parameterize
    kwargs["parameterize"] = default_parameterize

    out = train_and_eval(
        kwargs.pop("parameterize"), init_fn=None,
        steps=args.steps, L=args.L, M=args.M, A=args.A, d_model=args.d_model,
        d_state=args.d_state, n_layer=args.n_layer, batch=args.batch, lr=args.lr,
        seed=args.seed, device=device, expand=args.expand, d_conv=args.d_conv,
        proof_task=_TASK_BY_HOOK[args.hook], proof_label=args.label,
        surface_choice=surface_choice,
        protocol=args.protocol, optimizer_name=args.optimizer,
        weight_decay=args.weight_decay, grad_clip=args.grad_clip,
        eval_batches=args.eval_batches,
        **{k: v for k, v in kwargs.items()})

    print(f"COPY_DEBUG acc={out['acc']:.4f} final_loss={out['final_loss']:.4f} "
          f"wall_s={out['wall_s']:.1f}", flush=True)
    print(
        format_metric_line(
            "MAMBA_COPY_METRICS", protocol=args.protocol,
            task=_TASK_BY_HOOK[args.hook],
            label=args.label, surface_choice=surface_choice,
            L=args.L, M=args.M, A=args.A,
            d_model=args.d_model, d_state=args.d_state, n_layer=args.n_layer,
            steps=args.steps, batch=args.batch, lr=args.lr,
            optimizer=args.optimizer, weight_decay=args.weight_decay,
            grad_clip=args.grad_clip, eval_batches=args.eval_batches,
            seed=args.seed, result=out,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
