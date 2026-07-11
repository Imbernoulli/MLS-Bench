#!/usr/bin/env python3
"""Fixed-compute generation-length allocation over the full MBPP inventory."""
from __future__ import annotations

import argparse
import math
import time

import common


TASK_ID = "codegen-decode-length"
MIN_TOKEN_CAP = 64
MAX_TOKEN_CAP = 640
AVERAGE_TOKEN_CAP = 256
TOTAL_TOKEN_CAP = AVERAGE_TOKEN_CAP * common.EXPECTED_PROBLEMS


def _instruction(prob):
    return (
        "Complete the following Python function. Return ONLY the complete "
        "function inside a single ```python code block.\n\n" + prob["prompt"]
    )


def _compiles(src):
    try:
        compile(src, "<p>", "exec")
        return bool(src.strip())
    except Exception:
        return False


def allocate_token_caps(raw_weights, n):
    if type(raw_weights) not in {list, tuple} or len(raw_weights) != n:
        raise TypeError(f"token_cap_weights() must return {n} weights")
    weights = [common.require_real(value, "token-cap weight") for value in raw_weights]
    if any(value < 0.0 for value in weights) or not any(value > 0.0 for value in weights):
        raise ValueError("token-cap weights must be non-negative and not all zero")
    scale = max(weights)
    weights = [value / scale for value in weights]
    caps = [MIN_TOKEN_CAP] * n
    capacity = [MAX_TOKEN_CAP - MIN_TOKEN_CAP] * n
    remaining = TOTAL_TOKEN_CAP - MIN_TOKEN_CAP * n
    while remaining:
        active = [i for i in range(n) if capacity[i]]
        if not active:
            raise RuntimeError("token-cap allocator exhausted its capacity")
        active_weights = {i: weights[i] for i in active}
        if not any(value > 0.0 for value in active_weights.values()):
            active_weights = {i: 1.0 for i in active}
        total_weight = math.fsum(active_weights.values())
        quotas = {i: remaining * active_weights[i] / total_weight for i in active}
        floors = {i: min(capacity[i], int(math.floor(quotas[i]))) for i in active}
        floor_total = sum(floors.values())
        if floor_total:
            for i, amount in floors.items():
                caps[i] += amount
                capacity[i] -= amount
            remaining -= floor_total
            if not remaining:
                break
        candidates = [i for i in active if capacity[i]]
        candidates.sort(key=lambda i: (-(quotas[i] - math.floor(quotas[i])), i))
        take = min(remaining, len(candidates))
        if not take:
            continue
        for i in candidates[:take]:
            caps[i] += 1
            capacity[i] -= 1
        remaining -= take
    if sum(caps) != TOTAL_TOKEN_CAP or not all(
        MIN_TOKEN_CAP <= value <= MAX_TOKEN_CAP for value in caps
    ):
        raise RuntimeError("token-cap allocator produced an invalid budget")
    return caps


def _policy_weights(policy, problems):
    return common.call_policy_without_generation(policy, problems)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solution", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n", type=int, default=common.EXPECTED_PROBLEMS)
    args = ap.parse_args()
    common.set_seeds(args.seed)
    t0 = time.time()
    common.start_executor()
    _tok, _ = common.load_model()
    problems = common.load_problems(args.n)
    common.seal_private_data()
    token_cap_weights = common.load_surface(args.solution, "token_cap_weights")
    common.emit_protocol(TASK_ID, args.seed, problems)

    n = len(problems)
    policy_problems = [
        {"task_id": p["task_id"], "entry_point": p["entry_point"], "prompt": p["prompt"]}
        for p in problems
    ]
    token_caps = allocate_token_caps(
        _policy_weights(token_cap_weights, policy_problems), n
    )
    n_hidden = 0
    n_parsed = 0
    for i, (prob, token_cap) in enumerate(zip(problems, token_caps)):
        setup = prob.get("test_setup", "")
        raw = common.generate(common.GenConfig(
            prompt=_instruction(prob), n_samples=4, do_sample=True,
            temperature=0.6, top_p=0.95, max_new_tokens=token_cap), seed=args.seed)
        candidates = [common.extract_code(value) for value in raw]
        chosen = candidates[0]
        for candidate in candidates:
            if common.passes_all(candidate, prob["visible_tests"], setup):
                chosen = candidate
                break
        parse_ok = _compiles(chosen)
        hidden_ok = common.passes_all(chosen, prob["hidden_tests"], setup)
        n_parsed += int(parse_ok)
        n_hidden += int(hidden_ok)
        common.emit_item(i + 1, hidden_ok, parsed=parse_ok, tokens=token_cap)
        if i < 3:
            print(
                f"CG_SAMPLE i={i} id={prob['task_id']} token_cap={token_cap} "
                f"parsed={parse_ok} reserved_ok={hidden_ok}",
                flush=True,
            )
        if (i + 1) % 20 == 0:
            common.emit_progress(i + 1, n_hidden)
    if sum(token_caps) != TOTAL_TOKEN_CAP:
        raise RuntimeError("fixed token-cap budget was not consumed exactly")
    elapsed = time.time() - t0
    print(
        f"CG_METRICS task={TASK_ID} pass_at_1={n_hidden/n:.6f} "
        f"parse_rate={n_parsed/n:.6f} "
        f"avg_token_cap={sum(token_caps)/n:.6f} n={n} elapsed={elapsed:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
