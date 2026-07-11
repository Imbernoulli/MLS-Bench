#!/usr/bin/env python3
"""Generate the ten static full-protocol image-captioning task surfaces.

The generated solution files contain one literal ``CONFIG`` assignment.  The
verifier parses that assignment with ``ast.literal_eval``; no candidate code is
imported or executed.
"""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

MEASURED_SCORE_SPEC = '''"""Measured full-protocol Flickr8k calibration.

All ten research surfaces share the same official split, frozen CLIP/GPT-2
model, 7,500-step training protocol, and CIDEr/BLEU-4 evaluation.  The common
metric scale is calibrated by complete sample, greedy, and beam decoding runs.
Missing or invalid verifier metrics remain fail-closed at exact zero.
"""
from mlsbench.scoring.dsl import *

term(
    "cider_flickr",
    col("cider_flickr").higher().id().sigmoid(
        floor=const(0.245972),
        ref=const(0.586622),
    ),
)
term(
    "bleu4_flickr",
    col("bleu4_flickr").higher().id().sigmoid(
        floor=const(0.076101),
        ref=const(0.218874),
    ),
)
setting(
    "flickr",
    weighted_mean(
        ("cider_flickr", 1.0),
        ("bleu4_flickr", 1.0),
    ),
)
task(gmean("flickr"))
'''

LEADERBOARD_HEADER = (
    "timestamp,model,is_final,seed,cider_flickr,bleu4_flickr,elapsed_flickr\n"
)
MEASURED_ANCHORS = (
    (
        "2026-07-11T11:45:17.091422+00:00",
        "calibration:decoding_sample",
        0.245972,
        0.076101,
        199.94920420646667,
    ),
    (
        "2026-07-11T11:54:00.521367+00:00",
        "calibration:decoding_greedy",
        0.552901,
        0.198662,
        203.49038243293762,
    ),
    (
        "2026-07-11T11:57:47.549080+00:00",
        "calibration:decoding_beam5",
        0.586622,
        0.218874,
        204.5822253227234,
    ),
)
TASKS_ROOT = ROOT / "tasks"
SOLUTION_ROOT = ROOT / "vendor" / "image-captioning" / "solution"


SPECS = (
    {
        "task": "caption-decoding-strategy",
        "mode": "decoding",
        "file": "decoding.py",
        "title": "Caption Decoding Strategy",
        "question": "Investigate how autoregressive search should convert a trained visual prefix into a complete caption.",
        "contract": "Choose a greedy, beam, or nucleus-sampling strategy and provide every parameter required by that strategy. Lengths must be integral, probabilities and penalties finite, and decoding remains deterministic under the configured seed.",
        "native": {"strategy": "sample", "max_length": 24, "min_length": 3, "no_repeat_ngram": 0, "temperature": 1.0, "top_p": 1.0},
        "baselines": {
            "sample": {"strategy": "sample", "max_length": 24, "min_length": 3, "no_repeat_ngram": 0, "temperature": 1.0, "top_p": 1.0},
            "greedy": {"strategy": "greedy", "max_length": 24, "min_length": 3, "no_repeat_ngram": 2},
            "beam": {"strategy": "beam", "max_length": 24, "min_length": 3, "no_repeat_ngram": 2, "beam_size": 5, "length_penalty": 0.8},
        },
    },
    {
        "task": "caption-visual-mapping",
        "mode": "mapping",
        "file": "mapping.py",
        "title": "Visual Prefix Mapping",
        "question": "Investigate the trainable mapping from a frozen visual embedding to the prefix consumed by a frozen language model.",
        "contract": "Choose a linear, multilayer-perceptron, or transformer mapping. Supply the complete literal schema for the selected type; dimensions and the downstream prefix length remain fixed by the verifier.",
        "native": {"type": "linear"},
        "baselines": {
            "linear": {"type": "linear"},
            "mlp": {"type": "mlp", "hidden_ratio": 0.5, "activation": "tanh", "dropout": 0.0},
        },
    },
    {
        "task": "caption-training-objective",
        "mode": "objective",
        "file": "objective.py",
        "title": "Caption Training Objective",
        "question": "Investigate the amount of target-distribution smoothing used by next-token caption training.",
        "contract": "Set `label_smoothing` to a finite number from 0.0 through 0.3. Padding is always excluded and all other loss mechanics are fixed.",
        "native": {"label_smoothing": 0.0},
        "baselines": {
            "ce": {"label_smoothing": 0.0},
            "labelsmooth": {"label_smoothing": 0.1},
        },
    },
    {
        "task": "caption-feature-prep",
        "mode": "featureprep",
        "file": "feature_prep.py",
        "title": "Visual Feature Normalization",
        "question": "Investigate how frozen visual embeddings should be normalized before prefix-mapping training and inference.",
        "contract": "Set `normalization` to `none`, `l2`, or `standardize`. Standardization uses statistics from the training embeddings only and applies them unchanged during evaluation.",
        "native": {"normalization": "none"},
        "baselines": {
            "none": {"normalization": "none"},
            "l2": {"normalization": "l2"},
            "standardize": {"normalization": "standardize"},
        },
    },
    {
        "task": "caption-mapping-init",
        "mode": "init",
        "file": "init.py",
        "title": "Prefix Mapping Initialization",
        "question": "Investigate how the trainable visual-prefix mapping should be initialized before caption training.",
        "contract": "Set `scheme` to `pytorch_default`, `xavier_uniform`, `kaiming_uniform`, or `caption_mean`. The last option initializes the output prefix from frozen caption-token embeddings.",
        "native": {"scheme": "pytorch_default"},
        "baselines": {
            "default": {"scheme": "pytorch_default"},
            "xavier": {"scheme": "xavier_uniform"},
            "caption_mean": {"scheme": "caption_mean"},
        },
    },
    {
        "task": "caption-train-sampling",
        "mode": "sampling",
        "file": "sampling.py",
        "title": "Caption Batch Construction",
        "question": "Investigate how complete image-caption epochs should be assembled into optimization batches.",
        "contract": "Set `strategy` to `uniform` or `length_bucketed`. Both include every training pair exactly once per epoch. The latter groups similar target lengths within a batch, then randomizes batch order.",
        "native": {"strategy": "uniform"},
        "baselines": {
            "uniform": {"strategy": "uniform"},
            "bucketed": {"strategy": "length_bucketed"},
        },
    },
    {
        "task": "caption-optimizer",
        "mode": "optimizer",
        "file": "optimizer.py",
        "title": "Prefix Mapping Optimization",
        "question": "Investigate the optimizer and learning-rate trajectory used to fit the visual-prefix mapping.",
        "contract": "Choose AdamW or SGD and provide all required finite hyperparameters. Schedules may be constant, cosine, or cosine after warmup; warmup is zero for the first two choices.",
        "native": {"name": "sgd", "learning_rate": 0.001, "weight_decay": 0.0, "momentum": 0.9, "schedule": "constant", "warmup_steps": 0},
        "baselines": {
            "sgd": {"name": "sgd", "learning_rate": 0.001, "weight_decay": 0.0, "momentum": 0.9, "schedule": "constant", "warmup_steps": 0},
            "adamw": {"name": "adamw", "learning_rate": 0.00002, "weight_decay": 0.01, "schedule": "warmup_cosine", "warmup_steps": 500},
        },
    },
    {
        "task": "caption-prompt-format",
        "mode": "prompt",
        "file": "prompt.py",
        "title": "Caption Target Formatting",
        "question": "Investigate how caption targets should be normalized and whether they should include a short visual prefix phrase.",
        "contract": "Provide a complete literal choice of prefix, lowercasing, and terminal-period handling. The prefix is one of the three strings accepted by the fixed schema.",
        "native": {"prefix": "a photo of ", "lowercase": False, "strip_terminal_period": False},
        "baselines": {
            "photo_prefix": {"prefix": "a photo of ", "lowercase": False, "strip_terminal_period": False},
            "raw": {"prefix": "", "lowercase": False, "strip_terminal_period": False},
        },
    },
    {
        "task": "caption-feature-augment",
        "mode": "augment",
        "file": "augment.py",
        "title": "Visual Feature Regularization",
        "question": "Investigate train-only perturbations of frozen visual embeddings used to fit the prefix mapping.",
        "contract": "Set finite `gaussian_std` and `dropout_probability` values within the declared schema. Evaluation embeddings are never perturbed.",
        "native": {"gaussian_std": 0.0, "dropout_probability": 0.0},
        "baselines": {
            "none": {"gaussian_std": 0.0, "dropout_probability": 0.0},
            "regularized": {"gaussian_std": 0.01, "dropout_probability": 0.1},
        },
    },
    {
        "task": "caption-token-weighting",
        "mode": "weighting",
        "file": "weighting.py",
        "title": "Caption Token Weighting",
        "question": "Investigate whether next-token losses should weight all targets equally or emphasize rarer caption tokens.",
        "contract": "Set `scheme` to `uniform`, or use `idf` with finite `idf_power` and `idf_cap`. Corpus statistics are computed only from the complete training captions.",
        "native": {"scheme": "uniform"},
        "baselines": {
            "uniform": {"scheme": "uniform"},
            "idf": {"scheme": "idf", "idf_power": 0.5, "idf_cap": 3.0},
        },
    },
)


NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"


def _literal(value: dict) -> str:
    return "CONFIG = " + repr(value)


def _solution(spec: dict) -> tuple[str, int, int]:
    lines = [
        f'"""Agent-editable solution surface for {spec["title"].lower()}.',
        "",
        "The verifier statically parses CONFIG and never imports or executes this file.",
        '"""',
        "from __future__ import annotations",
        "",
        "# EDITABLE REGION",
        _literal(spec["native"]),
        "# END EDITABLE REGION",
        "",
    ]
    return "\n".join(lines), 8, 8


def _parser(mode: str) -> str:
    return f'''"""Strict completion parser for the official caption protocol."""
from __future__ import annotations

import math
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mlsbench.agent.parsers import OutputParser, ParseResult


EXPECTED_MODE = {mode!r}
EXPECTED_LABEL = "flickr"
NUMBER = r"{NUMBER}"
RESULT = re.compile(
    rf"CAPTION_RESULT protocol=flickr8k_official_v1 mode=(\\w+) "
    rf"train_images=(\\d+) train_pairs=(\\d+) eval_images=(\\d+) "
    rf"epochs=(\\d+) batch_size=(\\d+) steps=(\\d+) seed=(\\d+) "
    rf"split_sha256=([0-9a-f]{{{{64}}}}) manifest_sha256=([0-9a-f]{{{{64}}}}) "
    rf"predictions_sha256=([0-9a-f]{{{{64}}}}) cider=({{NUMBER}}) "
    rf"bleu4=({{NUMBER}}) status=ok"
)
FAILURE_MARKERS = (
    "Traceback (most recent call last)",
    "CAPTION_FAILURE",
    "_FALLBACK",
    "VERIFICATION_FAILED",
    "CUDA out of memory",
    "OutOfMemoryError",
    "Command exited with code",
    "Segmentation fault",
    "Killed",
    "TIMEOUT",
    "OUT_OF_MEMORY",
    "CANCELLED",
    "NODE_FAIL",
    "[ERROR]",
)


class Parser(OutputParser):
    def parse(self, cmd_label: str, raw_output: str) -> ParseResult:
        lines = [line.strip() for line in raw_output.splitlines() if line.strip()]
        records = [line for line in lines if line.startswith("CAPTION_RESULT")]
        if cmd_label != EXPECTED_LABEL:
            return ParseResult(feedback="caption command label is invalid", metrics={{}})
        if len(records) != 1 or not lines or lines[-1] != records[0]:
            return ParseResult(feedback="caption verification did not complete", metrics={{}})
        if any(marker in raw_output for marker in FAILURE_MARKERS):
            return ParseResult(feedback="caption verification reported a failure", metrics={{}})
        match = RESULT.fullmatch(records[0])
        if match is None:
            return ParseResult(feedback="caption completion proof is malformed", metrics={{}})
        mode, train_images, train_pairs, eval_images, epochs, batch_size, steps, seed, _, _, _, raw_cider, raw_bleu = match.groups()
        observed = tuple(map(int, (train_images, train_pairs, eval_images, epochs, batch_size, steps, seed)))
        if mode != EXPECTED_MODE or observed != (6000, 30000, 1000, 10, 40, 7500, 42):
            return ParseResult(feedback="caption completion proof does not bind the required protocol", metrics={{}})
        cider, bleu = float(raw_cider), float(raw_bleu)
        if not math.isfinite(cider) or not math.isfinite(bleu) or not 0.0 <= cider <= 10.0 or not 0.0 <= bleu <= 1.0:
            return ParseResult(feedback="caption metrics are invalid", metrics={{}})
        return ParseResult(
            feedback=f"complete caption evaluation: CIDEr={{cider:.6f}}, BLEU-4={{bleu:.6f}}",
            metrics={{"cider_flickr": cider, "bleu4_flickr": bleu}},
        )
'''


def _description(spec: dict) -> str:
    return f'''# Image Captioning: {spec["title"]}

## Objective

{spec["question"]} Modify only the declared editable file. No candidate ordering,
expected implementation, or evaluation category is supplied.

## Static Configuration Contract

The editable file contains one literal `CONFIG = {{...}}` assignment. {spec["contract"]}
Only literal strings, booleans, integers, and finite floating-point values are
accepted. Imports, calls, comprehensions, extra statements, missing keys, unknown
keys, malformed values, and incomplete configurations invalidate the run.

The verifier does not import or execute this file. It trains the same frozen-encoder,
frozen-decoder prefix captioner for ten complete epochs over every official training
image-caption pair, then evaluates every image in the fixed evaluation partition.
It reports CIDEr and BLEU-4 only after data hashes, training completion, prediction
count, and metric finiteness are proven.

Do not modify the harness, data, scripts, scorer, or unrelated files.
'''


def _script(spec: dict) -> str:
    return f'''#!/usr/bin/env bash
set -euo pipefail

cd /workspace/image-captioning
task_root="${{MLSBENCH_TASK_DIR:-${{TASK_DIR:-/tests/meta}}}}"
private_data="${{task_root}}/data/image-captioning"
runtime_data="${{OUTPUT_DIR:-/tmp}}/caption-official-{spec["mode"]}-${{SEED:-42}}"
rm -rf "${{runtime_data}}"
mkdir -p "${{runtime_data}}"
trap 'rc=$?; rm -rf "${{runtime_data}}"; if [[ ${{rc}} -ne 0 ]]; then printf "VERIFICATION_FAILED image-captioning rc=%s\\n" "${{rc}}" >&2; fi' EXIT
for name in source_manifest.json train_clip.pt train_refs.json eval_clip.pt eval_refs.json; do
    test -f "${{private_data}}/${{name}}"
    ln -s "${{private_data}}/${{name}}" "${{runtime_data}}/${{name}}"
done
test -f "${{CAPTION_GPT2:-/data/image-captioning/gpt2}}/config.json"

python harness.py \\
    --mode {spec["mode"]} \\
    --config solution/{spec["file"]} \\
    --data-root "${{runtime_data}}" \\
    --gpt-dir "${{CAPTION_GPT2:-/data/image-captioning/gpt2}}" \\
    --seed "${{SEED:-42}}"
'''


def _config(spec: dict, start: int, end: int) -> dict:
    private = []
    for name in (
        "source_manifest.json",
        "train_clip.pt",
        "train_refs.json",
        "eval_clip.pt",
        "eval_refs.json",
    ):
        private.append(
            {
                "name": f"caption_{name.replace('.', '_')}",
                "host_path": f"{{data_root}}/image-captioning/{name}",
                "dest": f"data/image-captioning/{name}",
                "required": True,
            }
        )
    return {
        "allow_create": False,
        "rigorous_codebase": False,
        "verifier_only_package_files": ["image-captioning/harness.py"],
        "agent_image_prune": ["/opt/mlsbench-caption"],
        "agent_data_prune": [
            f"/data/image-captioning/{name}"
            for name in (
                "source_manifest.json",
                "train_clip.pt",
                "train_refs.json",
                "eval_clip.pt",
                "eval_refs.json",
            )
        ],
        "verifier_data_deps": private,
        "seeds": [42],
        "test_cmds": [
            {
                "cmd": "scripts/run.sh",
                "label": "flickr",
                "group": 1,
                "compute": 1.0,
                "time": "4:00:00",
                "mem": 48,
                "package": "image-captioning",
            }
        ],
        "baselines": {
            name: {"edit_ops": f"edits/{name}.edit.py"}
            for name in spec["baselines"]
        },
        "files": [
            {
                "filename": f'image-captioning/solution/{spec["file"]}',
                "read": [{"start": -1, "end": -1}],
                "edit": [{"start": start, "end": end}],
            }
        ],
    }


def _edit(spec: dict, name: str, config: dict, start: int, end: int) -> str:
    target = f'image-captioning/solution/{spec["file"]}'
    content = _literal(config)
    return f'''"""Static reference configuration for offline calibration."""

OPS = [
    {{
        "op": "replace",
        "file": {target!r},
        "start_line": {start},
        "end_line": {end},
        "content": {content!r},
    }}
]
'''


def write_spec(spec: dict) -> None:
    task_dir = TASKS_ROOT / spec["task"]
    edits_dir = task_dir / "edits"
    scripts_dir = task_dir / "scripts"
    task_dir.mkdir(parents=True, exist_ok=True)
    edits_dir.mkdir(exist_ok=True)
    scripts_dir.mkdir(exist_ok=True)
    solution, start, end = _solution(spec)
    (SOLUTION_ROOT / spec["file"]).write_text(solution)
    for stale in edits_dir.glob("*.edit.py"):
        stale.unlink()
    for name, baseline in spec["baselines"].items():
        (edits_dir / f"{name}.edit.py").write_text(
            _edit(spec, name, baseline, start, end)
        )
    (task_dir / "config.json").write_text(
        json.dumps(_config(spec, start, end), indent=2) + "\n"
    )
    (task_dir / "parser.py").write_text(_parser(spec["mode"]))
    (task_dir / "task_description.md").write_text(_description(spec))
    for stale in scripts_dir.glob("*.sh"):
        stale.unlink()
    script = scripts_dir / "run.sh"
    script.write_text(_script(spec))
    script.chmod(0o755)
    (task_dir / "PENDING_FULL_OFFICIAL_ANCHORS").unlink(missing_ok=True)
    rows = [LEADERBOARD_HEADER]
    rows.extend(
        f"{timestamp},{model},true,42,{cider:.6f},{bleu:.6f},{elapsed:.9f}\n"
        for timestamp, model, cider, bleu, elapsed in MEASURED_ANCHORS
    )
    (task_dir / "leaderboard.csv").write_text("".join(rows))
    (task_dir / "score_spec.py").write_text(MEASURED_SCORE_SPEC)


def main() -> None:
    SOLUTION_ROOT.mkdir(parents=True, exist_ok=True)
    for spec in SPECS:
        write_spec(spec)
        print(f"generated {spec['task']}")


if __name__ == "__main__":
    main()
