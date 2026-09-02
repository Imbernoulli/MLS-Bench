#!/usr/bin/env python3
"""Create reduced-scale ("smoke") copies of rendered Harbor tasks.

A smoke copy keeps the image, workspace, edit guard, verifier and scorer of a
rendered task unchanged; only the verifier's eval scripts are shortened (fewer
epochs / steps / samples) so one Daytona run exercises

    build -> oracle -> guard -> train -> eval -> score

in minutes instead of hours.  It answers "does this environment work on this
provider?", nothing more: rewards from smoke copies are NOT benchmark scores
and must never be reported as such.

Usage::

    python scripts/make_smoke_copy.py <out_dir> <task-id> [<task-id> ...]
    harbor run -c run-daytona.yaml --path <out_dir>/mls-bench__<task-id> --agent oracle

Tasks without an entry in ``RULES`` are copied unchanged (a full-scale run).
The 30 MLS-Bench-Lite tasks that need shortening all have an entry; the
remaining Lite tasks finish at production scale within an hour on one GPU.
"""
from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

HARBOR_DIR = Path(__file__).resolve().parents[1]
SMOKE_ASSETS = Path(__file__).resolve().parent / "smoke"

# task-id -> [(script glob under tests/eval/scripts, [(regex, replacement), ...])]
RULES: dict[str, list[tuple[str, list[tuple[str, str]]]]] = {
    "cv-dbm-sampler": [
        ("run_*.sh", [(r"export num_samples=\d+", "export num_samples=256")]),
    ],
    "cv-vae-loss": [
        ("train_*.sh", [(r"export MAX_STEPS=\d+", "export MAX_STEPS=40"),
                        (r"export EVAL_INTERVAL=\d+", "export EVAL_INTERVAL=20")]),
    ],
    "dl-activation-function": [
        ("*.sh", [(r"--epochs \d+", "--epochs 1")]),
    ],
    "llm-pretrain-optimizer": [
        ("gpt_*.sh", [(r"MAX_ITERS=\$\{MAX_ITERS:-\d+\}", "MAX_ITERS=${MAX_ITERS:-20}"),
                      (r"EVAL_INTERVAL=\$\{EVAL_INTERVAL:-\d+\}", "EVAL_INTERVAL=${EVAL_INTERVAL:-10}")]),
    ],
    "llm-ptq-algorithm": [
        ("ptq_eval*.sh", [(r"--nsamples \d+", "--nsamples 16")]),
    ],
    "llm-rl-importance-sampling": [
        ("train.sh", [(r"\+trainer\.total_training_steps=\d+", "+trainer.total_training_steps=3"),
                      (r"trainer\.test_freq=\d+", "trainer.test_freq=3")]),
    ],
    "mlsys-sparse-attention-inference": [
        ("*.sh", [(r"\$\{MAX_CASES:-\d+\}", "${MAX_CASES:-5}")]),
    ],
    "rl-value-discrete": [
        # eval_freq defaults to 10000 timesteps; cut both so periodic evals fire
        ("*.sh", [(r"--total-timesteps \d+", "--total-timesteps 12000 --eval-freq 4000")]),
    ],
    "robo-diffusion-guidance": [
        ("train_*.sh", [(r"\+\+diffusion_gradient_steps=\d+", "++diffusion_gradient_steps=200"),
                        (r"\+\+classifier_gradient_steps=\d+", "++classifier_gradient_steps=200"),
                        (r"save_interval=\d+", "save_interval=200"),
                        (r"log_interval=\d+", "log_interval=100"),
                        (r"\+\+ckpt=\d+", "++ckpt=200"),
                        (r"num_episodes=\d+", "num_episodes=2")]),
    ],
    "robo-diffusion-policy": [
        ("train_*.sh", [(r"gradient_steps=\d+", "gradient_steps=200"),
                        (r"save_interval=\d+", "save_interval=200"),
                        (r"log_interval=\d+", "log_interval=100"),
                        (r"ckpt=\d+", "ckpt=200"),
                        (r"num_episodes=\d+", "num_episodes=1")]),
    ],
    "robomimic-bc-loss": [
        ("train.sh", [(r"config\['experiment'\]\['rollout'\]\['horizon'\] = \$\{HORIZON\}",
                       "config['experiment']['rollout']['horizon'] = ${HORIZON}\n"
                       "config['train']['num_epochs'] = 2\n"
                       "config['experiment']['rollout']['rate'] = 1\n"
                       "config['experiment']['rollout']['n'] = 2\n"
                       "config['experiment']['save']['every_n_epochs'] = 1")]),
    ],
    "security-membership-inference-defense": [
        ("*.sh", [(r"--epochs \d+", "--epochs 2")]),
    ],
    "ts-exogenous-forecast": [
        ("*.sh", [(r"--train_epochs \d+", "--train_epochs 1")]),
    ],
}

# Tasks whose training length is fixed inside the package config rather than
# on the command line ship alternative scripts under scripts/smoke/<task>/ and
# select them through the oracle command overrides (same token, new JSON).
OVERRIDE_SCRIPT_TASKS = {"robo-humanoid-sim2real-algo"}


def make(tasks_dir: Path, out_dir: Path, task_id: str) -> Path:
    src = tasks_dir / f"mls-bench__{task_id}"
    if not src.is_dir():
        raise SystemExit(f"{src} is not a rendered task")
    dst = out_dir / f"mls-bench__{task_id}"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, symlinks=True)
    scripts = dst / "tests" / "eval" / "scripts"
    changed = 0
    if task_id in OVERRIDE_SCRIPT_TASKS:
        asset_dir = SMOKE_ASSETS / task_id
        for script in sorted(asset_dir.glob("smoke_*.sh")):
            shutil.copy(script, scripts / script.name)
            changed += 1
        shutil.copy(asset_dir / "oracle_cmd_overrides.json",
                    dst / "solution" / "oracle_cmd_overrides.json")
    else:
        for pattern, subs in RULES.get(task_id, []):
            for script in sorted(scripts.glob(pattern)):
                text = script.read_text()
                new = text
                for rx, rep in subs:
                    new = re.sub(rx, rep, new)
                if new != text:
                    script.write_text(new)
                    changed += 1
        if task_id in RULES and changed == 0:
            raise SystemExit(f"{task_id}: no script changed; RULES are stale")
    (dst / "SMOKE_COPY.txt").write_text(
        f"Reduced-scale smoke copy of {task_id}; rewards are not benchmark scores.\n"
    )
    kind = "shortened" if changed else "unchanged (full scale)"
    print(f"{task_id}: {changed} script(s) {kind} -> {dst}")
    return dst


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("task_ids", nargs="+")
    parser.add_argument("--tasks-dir", type=Path, default=HARBOR_DIR / "tasks")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for task_id in args.task_ids:
        make(args.tasks_dir, args.out_dir, task_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
