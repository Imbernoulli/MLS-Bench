#!/usr/bin/env python3
"""Shared flow-* harness with task-specific, single-axis design surfaces.

The editable solution returns one scalar, enum, or mask sequence. This harness
validates that value and builds the complete normalizing flow in frozen code.
The target, data, all non-active architecture choices, optimizer, and training
budget therefore cannot be changed through an unrelated task surface.

The terminal protocol proof is intentionally redundant.  The parser validates
the selected surface, complete 20,000-step trace, frozen data identities,
layer accounting, parameter-count identity, metric consistency, and terminal
completion before exposing a scoreable metric.
"""
from __future__ import annotations

import argparse
import time

import torch

import common


PROTOCOL_VERSION = "flow-2d-community-20k-literal-ast-v3"

_SURFACES = (
    "architecture",
    "conditioner",
    "base_distribution",
    "batch_size",
    "conditioner_width",
    "coupling_transform",
    "depth",
    "learning_rate",
    "masking_pattern",
    "spline_bins",
)


def _require_choice(value, *, label: str, choices: set[str]) -> str:
    if not isinstance(value, str) or value not in choices:
        raise ValueError(f"{label} must be one of {sorted(choices)}")
    return value


def _require_int(value, *, label: str, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    if not low <= value <= high:
        raise ValueError(f"{label} must be in [{low}, {high}]")
    return value


def _require_float(value, *, label: str, low: float, high: float) -> float:
    import math

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not low <= result <= high:
        raise ValueError(f"{label} must be finite and in [{low}, {high}]")
    return result


def _append_canonical_recipe_layer(
    flows, fb, *, family: str, dim: int, index: int, spline_bins: int = 8,
) -> None:
    """Append one frozen recipe transform and its paired mixing layer."""
    if family == "affine":
        flows.append(fb.affine_coupling_layer(dim, hidden=64, n_hidden=2))
        flows.append(fb.swap_permute(dim))
    elif family == "maf":
        flows.append(fb.maf_layer(dim, hidden=64, num_blocks=2))
        flows.append(fb.lu_permute(dim))
    elif family == "spline":
        flows.append(fb.spline_coupling_layer(
            dim, hidden=64, num_bins=spline_bins, tail_bound=3.0,
            reverse_mask=bool(index % 2),
        ))
        flows.append(fb.lu_permute(dim))
    else:
        raise ValueError(f"unknown canonical recipe family {family!r}")


def _build_from_surface(solution: str, surface: str, dim: int):
    """Load exactly one task surface and build all remaining axes here."""
    import normflows as nf
    import flow_blocks as fb

    def call(name: str):
        return common.load_surface(solution, name)()

    flows = []
    q0 = fb.diag_gaussian(dim)
    lr = None
    batch_size = None
    choice_id = ""
    n_transforms = 0
    n_permutations = 0

    if surface == "architecture":
        choice = _require_choice(
            call("select_architecture"), label="architecture",
            choices={"affine", "maf", "spline"},
        )
        choice_id = choice
        n_transforms = 8
        n_permutations = 8
        for index in range(8):
            _append_canonical_recipe_layer(
                flows, fb, family=choice, dim=dim, index=index, spline_bins=8,
            )
    elif surface == "conditioner":
        choice = _require_choice(
            call("select_conditioner"), label="conditioner",
            choices={"affine", "maf", "spline"},
        )
        choice_id = choice
        n_transforms = 8
        n_permutations = 8
        for index in range(8):
            if choice == "affine":
                flows.append(fb.affine_coupling_layer(dim, hidden=64, n_hidden=2))
            elif choice == "maf":
                flows.append(fb.maf_layer(dim, hidden=64, num_blocks=2))
            else:
                flows.append(fb.spline_coupling_layer(
                    dim, hidden=64, num_bins=8, tail_bound=3.0,
                    reverse_mask=bool(index % 2),
                ))
            flows.append(fb.lu_permute(dim))
    elif surface == "base_distribution":
        choice = _require_choice(
            call("select_base_distribution"), label="base distribution",
            choices={"gaussian", "gaussian_trainable", "gmm"},
        )
        choice_id = choice
        n_transforms = 1
        n_permutations = 1
        if choice == "gaussian":
            q0 = fb.diag_gaussian(dim, trainable=False)
        elif choice == "gaussian_trainable":
            q0 = fb.diag_gaussian(dim, trainable=True)
        else:
            q0 = fb.gmm_base(dim, n_modes=8, trainable=True)
        flows = [
            fb.affine_coupling_layer(dim, hidden=64, n_hidden=2),
            fb.swap_permute(dim),
        ]
    elif surface == "batch_size":
        batch_size = _require_int(
            call("select_batch_size"), label="batch size", low=1, high=8192,
        )
        choice_id = str(batch_size)
        n_transforms = 8
        n_permutations = 8
        for index in range(8):
            flows.append(fb.spline_coupling_layer(
                dim, hidden=64, num_bins=8, tail_bound=3.0,
                reverse_mask=bool(index % 2),
            ))
            flows.append(fb.lu_permute(dim))
    elif surface == "conditioner_width":
        width = _require_int(
            call("select_conditioner_width"), label="conditioner width",
            low=2, high=512,
        )
        choice_id = str(width)
        n_transforms = 8
        n_permutations = 8
        for _ in range(8):
            flows.append(fb.affine_coupling_layer(dim, hidden=width, n_hidden=2))
            flows.append(fb.lu_permute(dim))
    elif surface == "coupling_transform":
        choice = _require_choice(
            call("select_coupling_transform"), label="coupling transform",
            choices={"affine", "spline4", "spline8"},
        )
        choice_id = choice
        n_transforms = 8
        n_permutations = 8
        depth = 8
        for index in range(depth):
            family = "affine" if choice == "affine" else "spline"
            bins = 4 if choice == "spline4" else 8
            _append_canonical_recipe_layer(
                flows, fb, family=family, dim=dim, index=index,
                spline_bins=bins,
            )
    elif surface == "depth":
        depth = _require_int(call("select_depth"), label="flow depth", low=1, high=32)
        choice_id = str(depth)
        n_transforms = depth
        n_permutations = depth
        for _ in range(depth):
            flows.append(fb.affine_coupling_layer(dim, hidden=64, n_hidden=2))
            flows.append(fb.swap_permute(dim))
    elif surface == "learning_rate":
        lr = _require_float(
            call("select_learning_rate"), label="learning rate", low=1e-6, high=1.0,
        )
        choice_id = format(lr, ".12g")
        n_transforms = 8
        n_permutations = 8
        for index in range(8):
            flows.append(fb.spline_coupling_layer(
                dim, hidden=64, num_bins=8, tail_bound=3.0,
                reverse_mask=bool(index % 2),
            ))
            flows.append(fb.lu_permute(dim))
    elif surface == "masking_pattern":
        masks = call("select_masks")
        if not isinstance(masks, (list, tuple)) or len(masks) != 8:
            raise ValueError("select_masks must return exactly eight masks")
        normalized = []
        for mask in masks:
            if not isinstance(mask, (list, tuple)) or len(mask) != dim:
                raise ValueError(f"each mask must contain exactly {dim} entries")
            if any(isinstance(v, bool) or not isinstance(v, int) or v not in (0, 1)
                   for v in mask):
                raise ValueError("mask entries must be integer 0 or 1")
            if sum(mask) in (0, dim):
                raise ValueError("each mask must transform and condition on a dimension")
            normalized.append(list(mask))
        for mask in normalized:
            flows.append(fb.masked_affine_layer(dim, mask, hidden=64, n_hidden=2))
        choice_id = "".join("".join(str(value) for value in mask) for mask in normalized)
        n_transforms = 8
        n_permutations = 0
    elif surface == "spline_bins":
        bins = _require_int(
            call("select_spline_bins"), label="spline bins", low=2, high=64,
        )
        choice_id = str(bins)
        n_transforms = 8
        n_permutations = 8
        for index in range(8):
            flows.append(fb.spline_coupling_layer(
                dim, hidden=64, num_bins=bins, tail_bound=3.0,
                reverse_mask=bool(index % 2),
            ))
            flows.append(fb.lu_permute(dim))
    else:  # argparse prevents this; keep the builder fail-closed for direct callers.
        raise ValueError(f"unknown flow surface {surface!r}")

    if not isinstance(q0, nf.distributions.BaseDistribution):
        raise TypeError("frozen flow builder produced an invalid base distribution")
    if not flows or not all(isinstance(flow, nf.flows.Flow) for flow in flows):
        raise TypeError("frozen flow builder produced an invalid flow sequence")
    if len(flows) != n_transforms + n_permutations:
        raise RuntimeError("frozen flow builder produced inconsistent layer accounting")
    return (
        q0,
        flows,
        lr,
        batch_size,
        choice_id,
        n_transforms,
        n_permutations,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--solution", required=True)
    ap.add_argument("--surface", required=True, choices=_SURFACES)
    ap.add_argument("--target", required=True, choices=list(common.TARGETS))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--n-train", type=int, default=30000)
    ap.add_argument("--n-test", type=int, default=30000)
    args = ap.parse_args()

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("FLOW_FAILED verifier requires exactly one visible CUDA GPU")

    common.set_seeds(args.seed)
    t0 = time.time()

    (
        q0,
        flows,
        selected_lr,
        selected_batch_size,
        choice_id,
        n_transforms,
        n_permutations,
    ) = _build_from_surface(args.solution, args.surface, common.DIM)
    lr = args.lr if selected_lr is None else selected_lr
    batch_size = args.batch_size if selected_batch_size is None else selected_batch_size

    import normflows as nf

    model = nf.NormalizingFlow(q0, flows)
    n_par = common.n_params(model)
    print(
        f"FLOW_PROTOCOL version={PROTOCOL_VERSION} surface={args.surface} "
        f"choice={choice_id} target={args.target} device=cuda device_count=1 "
        f"seed={args.seed} steps={args.steps} batch_size={batch_size} "
        f"lr={format(lr, '.12g')} optimizer=Adam objective=exact_nll",
        flush=True,
    )
    print(
        f"FLOW_DESIGN target={args.target} n_transforms={n_transforms} "
        f"n_permutations={n_permutations} total_layers={len(flows)} params={n_par}",
        flush=True,
    )

    x_tr, x_te = common.make_dataset(args.target, args.n_train, args.n_test, args.seed)
    print(
        f"FLOW_DATA target={args.target} seed={args.seed} "
        f"n_train={x_tr.shape[0]} n_test={x_te.shape[0]} "
        f"train_sha256={common.EXPECTED_TRAIN_FILE_SHA256[args.target]} "
        f"test_sha256={common.EXPECTED_TEST_FILE_SHA256[args.target]}",
        flush=True,
    )

    nll, bpd = common.train_and_eval(
        model, x_tr, x_te,
        steps=args.steps, batch_size=batch_size, lr=lr, seed=args.seed,
    )
    dt = time.time() - t0
    print(f"FLOW_METRICS nll={nll:.6f} bpd={bpd:.6f} params={n_par} elapsed={dt:.1f}",
          flush=True)
    print(
        f"FLOW_SETTING_COMPLETE version={PROTOCOL_VERSION} surface={args.surface} "
        f"choice={choice_id} target={args.target} seed={args.seed} "
        f"optimizer_steps={args.steps} samples_seen={args.steps * batch_size} "
        f"n_train={x_tr.shape[0]} n_test={x_te.shape[0]} params={n_par}",
        flush=True,
    )


if __name__ == "__main__":
    main()
