#!/usr/bin/env python3
"""State-space init harness for the mamba-state-init task.

The SSM parameterization is fixed to full selectivity (S6). The agent supplies a
literal initialization scheme for either A or Delta; trusted code applies it only
to the selected state tensors and keeps every other parameter and hook unchanged.

Emits protocol-bound ``POOL_LOADED`` and ``MAMBA_TRAIN_COMPLETE`` proofs followed
by one ``MAMBA_INIT_METRICS`` record. All three bind the full run configuration
and parameter count.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from common import (
    default_parameterize,
    format_metric_line,
    get_device,
    load_surface_hook_and_choice,
    train_and_eval,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--solution", required=True)
    ap.add_argument("--init-surface", required=True, choices=["delta", "state"])
    ap.add_argument("--label", default="paper_e1")
    ap.add_argument("--L", type=int, default=4096)
    ap.add_argument("--M", type=int, default=16)
    ap.add_argument("--A", type=int, default=16)
    ap.add_argument("--d_model", type=int, default=64)
    ap.add_argument("--d_state", type=int, default=16)
    ap.add_argument("--n_layer", type=int, default=2)
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
    function_name = "init_delta" if args.init_surface == "delta" else "init_state"
    init_ssm, surface_choice = load_surface_hook_and_choice(
        Path(args.solution), function_name
    )
    task = "mamba-dt-init" if args.init_surface == "delta" else "mamba-state-init"

    out = train_and_eval(
        default_parameterize, init_fn=init_ssm,
        steps=args.steps, L=args.L, M=args.M, A=args.A, d_model=args.d_model,
        d_state=args.d_state, n_layer=args.n_layer, batch=args.batch, lr=args.lr,
        seed=args.seed, device=device, init_allowed=args.init_surface,
        proof_task=task, proof_label=args.label, surface_choice=surface_choice,
        protocol=args.protocol,
        optimizer_name=args.optimizer, weight_decay=args.weight_decay,
        grad_clip=args.grad_clip, eval_batches=args.eval_batches)

    print(f"INIT_DEBUG acc={out['acc']:.4f} final_loss={out['final_loss']:.4f} "
          f"wall_s={out['wall_s']:.1f}", flush=True)
    print(
        format_metric_line(
            "MAMBA_INIT_METRICS", protocol=args.protocol,
            task=task, label=args.label, surface_choice=surface_choice,
            L=args.L, M=args.M, A=args.A, d_model=args.d_model,
            d_state=args.d_state, n_layer=args.n_layer, steps=args.steps,
            batch=args.batch, lr=args.lr, optimizer=args.optimizer,
            weight_decay=args.weight_decay, grad_clip=args.grad_clip,
            eval_batches=args.eval_batches, seed=args.seed, result=out,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
