from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import signal
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest


def _load_score_task():
    path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "mls_bench"
        / "task-template"
        / "tests"
        / "score_task.py"
    )
    spec = importlib.util.spec_from_file_location("score_task_under_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_eval_env_points_scripts_at_sanitized_verifier_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    score_task = _load_score_task()
    task_meta = tmp_path / "private-meta"
    eval_meta = tmp_path / "eval-meta"
    workspace = tmp_path / "workspace"
    package = workspace / "pkg"
    out_dir = tmp_path / "out"
    for path in (task_meta, eval_meta / "data", package, out_dir):
        path.mkdir(parents=True)
    (task_meta / "package").write_text("pkg\n")
    (task_meta / "task_id").write_text("task\n")
    monkeypatch.setenv("MLSBENCH_VERIFIER_DATA_ROOT", "/agent-controlled")

    env = score_task._eval_env(
        task_meta=task_meta,
        eval_task_meta=eval_meta,
        out_dir=out_dir,
        workspace_root=workspace,
        pkg_dir=package,
        tc={"label": "case", "package": "pkg"},
        seed=42,
    )

    assert env["MLSBENCH_VERIFIER_DATA_ROOT"] == str(eval_meta / "data")


def test_edit_guard_rejects_deleted_fixed_separator_with_duplicate_in_editable(tmp_path: Path):
    score_task = _load_score_task()
    pristine = tmp_path / "pristine.py"
    current = tmp_path / "current.py"

    pristine.write_text(
        "header\n"
        "editable before\n"
        "===\n"
        "editable after\n"
        "===\n"
        "second editable\n"
        "tail\n"
    )
    current.write_text(
        "header\n"
        "editable before\n"
        "===\n"
        "editable after\n"
        "second editable\n"
        "tail\n"
    )

    ranges = [score_task.EditRange(2, 4), score_task.EditRange(6, 6)]
    ok, reason = score_task._check_editable_only(pristine, current, ranges)

    assert not ok
    assert reason is not None
    assert "only the declared editable range" in reason


def test_guard_accepts_unchanged_workspace_for_failed_or_noop_agent(tmp_path: Path):
    score_task = _load_score_task()
    task_meta = tmp_path / "meta"
    pristine = task_meta / "pristine"
    workspace = tmp_path / "workspace"
    task_meta.mkdir()
    pristine.mkdir()
    workspace.mkdir()
    config = {
        "files": [{
            "filename": "pkg/solution.py",
            "edit": [{"start": 2, "end": 2}],
        }],
    }
    (task_meta / "config.json").write_text(json.dumps(config))
    (task_meta / "pristine_manifest.json").write_text(json.dumps({
        "pkg/solution.py": "unused-by-declared-file-check",
    }))
    (pristine / "pkg").mkdir()
    (workspace / "pkg").mkdir()
    source = "fixed\nreturn 'weak default'\nfixed\n"
    (pristine / "pkg" / "solution.py").write_text(source)
    (workspace / "pkg" / "solution.py").write_text(source)
    violation = tmp_path / "violation.txt"

    rc = score_task.cmd_guard(argparse.Namespace(
        task_meta=str(task_meta),
        pristine=str(pristine),
        workspace=str(workspace),
        violation_out=str(violation),
    ))

    assert rc == 0
    assert not violation.exists()


def test_guard_accepts_edit_within_declared_range(tmp_path: Path):
    score_task = _load_score_task()
    task_meta = tmp_path / "meta"
    pristine = task_meta / "pristine"
    workspace = tmp_path / "workspace"
    task_meta.mkdir()
    pristine.mkdir()
    workspace.mkdir()
    config = {
        "files": [{
            "filename": "pkg/solution.py",
            "edit": [{"start": 2, "end": 2}],
        }],
    }
    (task_meta / "config.json").write_text(json.dumps(config))
    (task_meta / "pristine_manifest.json").write_text(json.dumps({
        "pkg/solution.py": "unused-by-declared-file-check",
    }))
    (pristine / "pkg").mkdir()
    (workspace / "pkg").mkdir()
    (pristine / "pkg" / "solution.py").write_text("fixed\nreturn 'weak'\nfixed\n")
    (workspace / "pkg" / "solution.py").write_text("fixed\nreturn 'agent'\nfixed\n")
    violation = tmp_path / "violation.txt"

    rc = score_task.cmd_guard(argparse.Namespace(
        task_meta=str(task_meta),
        pristine=str(pristine),
        workspace=str(workspace),
        violation_out=str(violation),
    ))

    assert rc == 0
    assert not violation.exists()


def test_metric_aggregation_rejects_entire_matrix_on_nonfinite_value():
    score_task = _load_score_task()

    mean = score_task._aggregate_metrics([
        {"acc": "0.5", "loss": float("nan")},
        {"acc": 1.0, "loss": 7.0},
    ])

    assert mean == {}


def test_metric_aggregation_rejects_all_nan_values():
    score_task = _load_score_task()

    mean = score_task._aggregate_metrics([
        {"acc": float("nan")},
        {"acc": "nan"},
    ])

    assert mean == {}


@pytest.mark.parametrize(
    "metrics",
    [
        {"score": "not-a-number"},
        {"score": float("nan")},
        {"score": float("inf")},
        {"score": True},
        {"score": 1.0, "ignored_std": float("nan")},
        {"score": 1.0, "elapsed_eval": float("inf")},
        [("score", 1.0)],
    ],
)
def test_parser_metric_validation_rejects_every_invalid_value(metrics):
    score_task = _load_score_task()

    assert score_task._parser_metrics_error(metrics) is not None


_SELECTED20_METRIC_LOGS = [
    (
        "cd-numeric-answer",
        "gsm8k",
        [
            "CD_MODEL params=494032768 device=cuda:0 dtype=torch.float16",
            "CD_DATA dataset=gsm8k n=1319 seed=42",
            "CD_METRICS valid_rate=0.5 accuracy=0.1 n=1319 elapsed=1",
            "CD_COMPLETE dataset=gsm8k n=1319 seed=42",
        ],
    ),
    (
        "summ-source-policy",
        "summ",
        [
            "SUMM_METRICS setting=xsum rougeL=0.1 rouge1=0.2 rouge2=0.05 plen=10",
            "SUMM_METRICS setting=cnndm rougeL=0.2 rouge1=0.3 rouge2=0.1 plen=20",
            "SUMM_METRICS setting=samsum rougeL=0.3 rouge1=0.4 rouge2=0.2 plen=15",
        ],
    ),
    (
        "nli-finetune",
        "snli",
        ["NLI_METRICS acc=0.5 n_train=6000 n_test=3000 elapsed=1"],
    ),
    (
        "simp-source-policy",
        "simplify",
        [
            "SIMP_METRICS setting=asset sari=40 bleu=20 n_sents=359 plen=10 lenratio=0.8",
            "SIMP_METRICS setting=turk sari=41 bleu=21 n_sents=359 plen=11 lenratio=0.9",
            "SIMP_METRICS setting=wiki sari=42 bleu=22 n_sents=720 plen=12 lenratio=1.0",
            "SIMP_DONE policy=empty elapsed=1.0",
        ],
    ),
    (
        "kge-training-epochs",
        "FB15k237",
        [
            "KGE_PROTOCOL_RESULT "
            + json.dumps(
                {
                    "protocol_version": "kge-fullscale-v2",
                    "status": "success",
                    "pykeen_version": "1.11.1",
                    "dataset": "FB15k237",
                    "seed": 42,
                    "model_random_seed": 42,
                    "device_type": "cuda",
                    "gpu_name": "test-gpu",
                    "entities": 14541,
                    "relations": 237,
                    "train_triples": 272115,
                    "validation_triples": 17535,
                    "test_triples": 20466,
                    "split_sha256": {
                        "train": "6e4c2782169af21e9743f3b1d200886f5d595bf6bc504ec1351720949c5cdfae",
                        "valid": "cf6309010852f6a8d47a45df830a426415d1ee6f7a3970a8376ff1fb81db4a5c",
                        "test": "5711cf41623ceb4eacc50eb6108a3ca6565c7492e3caaf82a3e355cc660d1574",
                    },
                    "training_loop": "sLCWA",
                    "entity_dim": 400,
                    "relation_dim": 400,
                    "model_parameters": 5911200,
                    "interaction": "packed-real-ComplEx",
                    "loss": "SoftplusLoss",
                    "optimizer": "Adam",
                    "learning_rate": 1e-3,
                    "epochs_completed": 1,
                    "batch_size": 1024,
                    "training_instances_per_epoch": 272115,
                    "steps_per_epoch": 266,
                    "optimizer_steps_completed": 266,
                    "optimizer_steps_source": "torch_optimizer_step_return",
                    "negative_sampler": "bernoulli",
                    "num_negs_per_pos": 64,
                    "final_loss": 1.0,
                    "evaluation": "filtered_both_realistic_full_test",
                    "evaluation_batch_size": 256,
                    "automatic_memory_optimization": False,
                    "evaluation_fallback": False,
                    "stopper": "nop",
                    "create_inverse_triples": False,
                    "evaluated_test_triples": 20466,
                    "train_seconds": 0.4,
                    "evaluate_seconds": 0.5,
                    "elapsed_seconds": 1.0,
                    "metrics": {
                        "MRR": 50.0,
                        "Hits@10": 80.0,
                        "Hits@3": 60.0,
                        "Hits@1": 40.0,
                    },
                },
                sort_keys=True,
            )
        ],
    ),
    (
        "molgen-learning-rate",
        "n1000",
        [
            "MOLGEN_METRICS n_train=1000 vun=0.1 frac_valid=0.5 "
            "frac_unique=0.5 frac_novel=0.4 n_valid=500 n_unique=250 "
            "n_novel=100 n_requested=1000"
        ],
    ),
    (
        "cv-count-normalization",
        "medium",
        ["COUNT_METRICS surface=norm setting=medium mae=10 rmse=12 nae=0.2"],
    ),
    (
        "cv-pointcloud-ball-query-radius",
        "clean",
        [
            "PN2_METRICS mode=group regime=clean test_acc=0.7 "
            "class_acc=0.6 n_train=800 n_test=200"
        ],
    ),
    (
        "cv-inpaint-loss-design",
        "small",
        [
            "INPAINT_METRICS surface=loss setting=small hole_l1=0.1 "
            "hole_psnr=20 full_l1=0.05 hole_frac=0.25"
        ],
    ),
    (
        "openclip-contrastive-loss",
        "all",
        [
            "OPENCLIP_INVENTORY protocol=openclip_full_canonical_finetune_v1 "
            f"manifest_sha256={'a' * 64} files=28 datasets=3",
            "OPENCLIP_PROTOCOL protocol=openclip_full_canonical_finetune_v1 "
            f"model=ViT-B-32 checkpoint_sha256={'b' * 64} epochs=10 "
            "batch_size=128 seed=42 settings=3 device_count=1",
            "OPENCLIP_METRICS protocol=openclip_full_canonical_finetune_v1 mode=loss "
            "dataset=cifar10 zeroshot_acc=0.8 pool_n=50000 train_n=50000 "
            "eval_n=10000 epochs=10 steps=3910 samples_seen=500000 "
            "train_elapsed=90 elapsed=100",
            "OPENCLIP_METRICS protocol=openclip_full_canonical_finetune_v1 mode=loss "
            "dataset=cifar100 zeroshot_acc=0.5 pool_n=50000 train_n=50000 "
            "eval_n=10000 epochs=10 steps=3910 samples_seen=500000 "
            "train_elapsed=91 elapsed=101",
            "OPENCLIP_METRICS protocol=openclip_full_canonical_finetune_v1 mode=loss "
            "dataset=stl10 zeroshot_acc=0.7 pool_n=5000 train_n=5000 "
            "eval_n=8000 epochs=10 steps=400 samples_seen=50000 "
            "train_elapsed=15 elapsed=20",
            "OPENCLIP_COMPLETE protocol=openclip_full_canonical_finetune_v1 "
            "settings=3 metric_lines=3 optimizer_steps=8220 samples_seen=1050000 "
            "eval_images=28000 elapsed=221",
        ],
    ),
    (
        "caption-decoding-strategy",
        "flickr",
        [
            "CAPTION_RESULT protocol=flickr8k_official_v1 mode=decoding "
            "train_images=6000 train_pairs=30000 eval_images=1000 "
            "epochs=10 batch_size=40 steps=7500 seed=42 "
            f"split_sha256={'a' * 64} manifest_sha256={'b' * 64} "
            f"predictions_sha256={'c' * 64} cider=0.4 bleu4=0.2 status=ok"
        ],
    ),
    (
        "reid-spatial-pooling",
        "market",
        [
            "REID_PROTOCOL schema=1 seed=42 model=resnet50 epochs=60 batch=64 "
            "instances=4 train_images=12936 query_images=3368 gallery_images=19732 "
            "train_sha=4f1a5416bad595a67a45652568919252e56a54e99c49fd74f1fd29492123f3d3 "
            "query_sha=d34ff6d094521111a10a16f7879f01bb210abdeab24efba4b950fe1f3b9e90f7 "
            "gallery_sha=7900c8355955f1ca7e2ad5d6844f4be03dddfc3ded1f7a21cf43e55441075c4e "
            "weights_sha=0676ba61b6795bbe1773cffd859882e5e297624d384b6993f7c9e683e722fb8a",
            *[
                f"REID_EPOCH epoch={epoch} steps=160 total_steps={(epoch + 1) * 160} "
                "loss=1.25 lr=0.0003"
                for epoch in range(60)
            ],
            "REID_TRAIN_COMPLETE epochs=60 total_steps=9600 train_samples=614400",
            "REID_METRICS setting=easy map=0.4 rank1=0.5 rank5=0.7 "
            "num_query=1122 num_gallery=19732 elapsed=1000",
            "REID_METRICS setting=medium map=0.3 rank1=0.4 rank5=0.6 "
            "num_query=1123 num_gallery=19732 elapsed=1000",
            "REID_METRICS setting=hard map=0.2 rank1=0.3 rank5=0.5 "
            "num_query=1123 num_gallery=19732 elapsed=1000",
            "REID_EVAL_COMPLETE settings=easy,medium,hard query_total=3368 gallery=19732",
        ],
    ),
    (
        "flow-coupling-transform",
        "checkerboard",
        [
            "FLOW_DESIGN target=checkerboard n_layers=16 params=1000",
            "FLOW_DATA target=checkerboard seed=42 n_train=30000 n_test=30000",
            "FLOW_TRAIN step=19999 train_nll=3.1",
            "FLOW_METRICS nll=3.0 bpd=2.164043 params=1000 elapsed=1",
            "FLOW_SETTING_COMPLETE target=checkerboard",
        ],
    ),
    (
        "ood-logit-score",
        "ood_logit_full_protocol",
        [
            "OOD_METRICS protocol=openood_cifar10_resnet18_full_v1 "
            "setting=ood_logit_svhn_full ood=svhn auroc=0.93 fpr95=0.44 "
            "id_acc=0.9514 n_fit=50000 n_id=10000 n_ood=26032 "
            "forward_batches=204 inference_seconds=1.1 status=ok",
            "OOD_METRICS protocol=openood_cifar10_resnet18_full_v1 "
            "setting=ood_logit_cifar100_full ood=cifar100 auroc=0.89 fpr95=0.52 "
            "id_acc=0.9514 n_fit=50000 n_id=10000 n_ood=10000 "
            "forward_batches=79 inference_seconds=0.5 status=ok",
            "OOD_METRICS protocol=openood_cifar10_resnet18_full_v1 "
            "setting=ood_logit_tin_full ood=tin auroc=0.89 fpr95=0.50 "
            "id_acc=0.9514 n_fit=50000 n_id=10000 n_ood=10000 "
            "forward_batches=79 inference_seconds=0.5 status=ok",
            "OOD_COMPLETE protocol=openood_cifar10_resnet18_full_v1 "
            "data_sha256=796799c9a1c073784b02a3f42a00d0fc8b902387b1f3345b5b3d1f2631e0722d "
            "checkpoint_sha256=8859e0ff484ab029fcd5bd0f85052c5679d82b8e36a7c066e922e2fdff62b7dc "
            "n_fit=50000 n_id=10000 n_svhn=26032 n_cifar100=10000 n_tin=10000 "
            "total_forward_images=106032 total_forward_batches=832 status=ok",
        ],
    ),
    (
        "mamba-selective-scan",
        "paper_e1",
        [
            "POOL_LOADED protocol=mamba_selective_copy_paper_e1_v1 "
            "task=mamba-selective-scan label=paper_e1 surface=mode.selective "
            "L=4096 M=16 A=16 d_model=64 d_state=16 n_layer=2 steps=400000 "
            "batch=64 lr=0.0001 optimizer=adam weight_decay=0 grad_clip=1 "
            "eval_batches=16 seed=42 n_params=1000 train_examples=25600000 "
            "train_tokens=104857600000 eval_examples=1024 eval_tokens=4194304 "
            "device=cuda",
            "MAMBA_TRAIN_COMPLETE protocol=mamba_selective_copy_paper_e1_v1 "
            "task=mamba-selective-scan label=paper_e1 surface=mode.selective "
            "L=4096 M=16 A=16 d_model=64 d_state=16 n_layer=2 steps=400000 "
            "batch=64 lr=0.0001 optimizer=adam weight_decay=0 grad_clip=1 "
            "eval_batches=16 seed=42 n_params=1000 train_examples=25600000 "
            "train_tokens=104857600000 eval_examples=1024 eval_tokens=4194304 "
            "final_loss=0.100000",
            "MAMBA_EVAL_COMPLETE protocol=mamba_selective_copy_paper_e1_v1 "
            "task=mamba-selective-scan label=paper_e1 surface=mode.selective "
            "L=4096 M=16 A=16 d_model=64 d_state=16 n_layer=2 steps=400000 "
            "batch=64 lr=0.0001 optimizer=adam weight_decay=0 grad_clip=1 "
            "eval_batches=16 seed=42 n_params=1000 train_examples=25600000 "
            "train_tokens=104857600000 eval_examples=1024 eval_tokens=4194304 "
            "eval_correct=12288 copy_acc=0.750000",
            "MAMBA_COPY_METRICS protocol=mamba_selective_copy_paper_e1_v1 "
            "task=mamba-selective-scan label=paper_e1 surface=mode.selective "
            "L=4096 M=16 A=16 d_model=64 d_state=16 n_layer=2 steps=400000 "
            "batch=64 lr=0.0001 optimizer=adam weight_decay=0 grad_clip=1 "
            "eval_batches=16 seed=42 n_params=1000 train_examples=25600000 "
            "train_tokens=104857600000 eval_examples=1024 eval_tokens=4194304 "
            "copy_acc=0.750000 final_loss=0.100000 wall_s=1.0 "
            "eval_correct=12288",
        ],
    ),
    (
        "mdn-density-bench",
        "inverse_sine",
        [
            "MDN_COMPLETE target=inverse_sine seed=42 steps=4000 "
            "final_step=3999 batch_size=512 n_train=20000 n_test=20000",
            "MDN_METRICS nll=-1.0 params=1000 elapsed=1",
        ],
    ),
    (
        "compress-activation",
        "low",
        ["COMPRESS_METRICS psnr=25 bpp=1 rd=17 target=1 elapsed=1"],
    ),
    (
        "gp-kernel-design",
        "concrete",
        [
            "GP_METRICS protocol=openml_full_v1 dataset=concrete "
            "n_train=927 n_test=103 budget_kind=iterations budget=200 "
            "nll=3.1 rmse=0.5 elapsed=1"
        ],
    ),
    (
        "spkverif-temporal-pooling",
        "libri",
        ["SV_METRICS setting=libri eer=0.2 mindcf=0.3 chance=0.5 n_pairs=1000 elapsed=1"],
    ),
    (
        "inr-fourier-frequency",
        "low",
        [
            "DATA_INFO signal=low res=256 n_coords=65536 dev=cuda",
            "STEP_METRICS label=frequency step=2000/2000 loss=0.01 psnr=34",
            "INR_METRICS signal=low psnr=35 res=256 elapsed=1",
            "INR_DONE signal=low n_coords=65536 steps=2000 seed=0",
        ],
    ),
]


@pytest.mark.parametrize(
    ("task_id", "cmd_label", "metric_lines"),
    _SELECTED20_METRIC_LOGS,
    ids=[case[0] for case in _SELECTED20_METRIC_LOGS],
)
def test_selected20_parsers_have_one_authoritative_line_per_scored_metric(
    task_id: str,
    cmd_label: str,
    metric_lines: list[str],
):
    score_task = _load_score_task()
    repo_root = Path(__file__).resolve().parents[2]
    task_dir = repo_root / "tasks" / task_id

    parser_spec = importlib.util.spec_from_file_location(
        f"authority_parser_{task_id.replace('-', '_')}",
        task_dir / "parser.py",
    )
    parser_module = importlib.util.module_from_spec(parser_spec)
    assert parser_spec.loader is not None
    parser_spec.loader.exec_module(parser_module)

    from mlsbench.scoring.spec import load_score_spec

    score_spec = load_score_spec(task_dir)
    assert score_spec is not None
    score_metric_keys = {
        term.metric
        for term in score_spec.terms.values()
        if term.role != "drop"
    }
    normal_log = "TRAIN diagnostic=1\n" + "\n".join(metric_lines) + "\n"
    normal_metrics = parser_module.Parser().parse(cmd_label, normal_log).metrics
    normal_scored_keys = set(normal_metrics) & score_metric_keys
    assert normal_scored_keys, task_id
    assert score_task._duplicate_authoritative_metric_lines(
        parser_module.Parser,
        cmd_label,
        normal_log,
        score_metric_keys,
    ) == {}

    isolated = [
        (
            line,
            set(parser_module.Parser().parse(cmd_label, line).metrics)
            & score_metric_keys,
        )
        for line in metric_lines
    ]
    independently_authoritative = [entry for entry in isolated if entry[1]]
    if independently_authoritative:
        # Parsers that accept an authoritative record in isolation are also
        # protected by score_task's generic per-line duplicate detector.
        forged_line, duplicated_keys = independently_authoritative[0]
        duplicates = score_task._duplicate_authoritative_metric_lines(
            parser_module.Parser,
            cmd_label,
            normal_log + forged_line + "\n",
            score_metric_keys,
        )
        assert set(duplicates) == duplicated_keys
        assert all(len(lines) == 2 for lines in duplicates.values())
    else:
        # Completion-proof parsers intentionally reject every protocol line
        # in isolation. They must reject a duplicated line in the complete
        # protocol themselves; the generic per-line detector is inapplicable.
        rejected_duplicates = []
        for forged_line in metric_lines:
            forged_metrics = parser_module.Parser().parse(
                cmd_label,
                normal_log + forged_line + "\n",
            ).metrics
            if not (set(forged_metrics) & score_metric_keys):
                rejected_duplicates.append(forged_line)
        assert rejected_duplicates == metric_lines, task_id


@pytest.mark.parametrize(
    ("task_id", "cmd_label", "_metric_lines"),
    _SELECTED20_METRIC_LOGS,
    ids=[case[0] for case in _SELECTED20_METRIC_LOGS],
)
@pytest.mark.parametrize(
    "failed_log",
    [
        "",
        "training failed\nTraceback (most recent call last):\nRuntimeError: boom\n",
        "loss=nan\n",
        "accuracy=0.9\n",
    ],
    ids=("empty", "traceback", "nonfinite-noise", "generic-noise"),
)
def test_selected20_parsers_do_not_invent_scored_metrics_on_failed_logs(
    task_id: str,
    cmd_label: str,
    _metric_lines: list[str],
    failed_log: str,
):
    repo_root = Path(__file__).resolve().parents[2]
    task_dir = repo_root / "tasks" / task_id
    parser_spec = importlib.util.spec_from_file_location(
        f"failed_log_parser_{task_id.replace('-', '_')}",
        task_dir / "parser.py",
    )
    parser_module = importlib.util.module_from_spec(parser_spec)
    assert parser_spec.loader is not None
    parser_spec.loader.exec_module(parser_module)

    from mlsbench.scoring.spec import load_score_spec

    score_spec = load_score_spec(task_dir)
    assert score_spec is not None
    score_metric_keys = {
        term.metric
        for term in score_spec.terms.values()
        if term.role != "drop"
    }
    parsed = parser_module.Parser().parse(cmd_label, failed_log)
    metrics = getattr(parsed, "metrics", None) or {}

    assert not (set(metrics) & score_metric_keys), (task_id, failed_log, metrics)


def test_openclip_compact_eval_reconstruction_is_authenticated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    import numpy as np
    import torch

    harness_path = (
        Path(__file__).resolve().parents[2] / "vendor" / "open_clip" / "harness.py"
    )
    harness_spec = importlib.util.spec_from_file_location(
        "openclip_compact_harness_under_test",
        harness_path,
    )
    harness = importlib.util.module_from_spec(harness_spec)
    assert harness_spec.loader is not None
    harness_spec.loader.exec_module(harness)

    raw = np.arange(2 * 2 * 2 * 3, dtype=np.uint8).reshape(2, 2, 2, 3)
    labels = torch.tensor([0, 1], dtype=torch.int64)
    classnames = ["zero", "one"]

    def preprocess(image):
        pixels = torch.from_numpy(np.array(image, dtype=np.float32, copy=True))
        channel_means = pixels.mean(dim=(0, 1)) / 255.0
        return channel_means[:, None, None].expand(3, 224, 224).clone()

    processed = torch.empty((2, 3, 224, 224), dtype=torch.float16)
    for index, image in enumerate(raw):
        processed[index].copy_(preprocess(image))
    classnames_text = json.dumps(
        classnames,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode()
    monkeypatch.setitem(
        harness.EXPECTED_EVAL,
        "tiny",
        {
            "raw_shape": tuple(raw.shape),
            "classes": 2,
            "raw_sha256": harness._array_sha256(raw),
            "labels_sha256": harness._tensor_sha256(labels),
            "classnames_sha256": hashlib.sha256(classnames_text).hexdigest(),
            "processed_sha256": harness._tensor_sha256(processed),
        },
    )

    np.save(tmp_path / "eval_tiny_raw_images.npy", raw, allow_pickle=False)
    torch.save(labels, tmp_path / "eval_tiny_labels.pt")
    (tmp_path / "classnames_tiny.json").write_text(json.dumps(classnames))

    images_out, labels_out, names_out = harness.load_eval(
        tmp_path,
        "tiny",
        preprocess,
    )

    assert torch.equal(images_out, processed)
    assert torch.equal(labels_out, labels)
    assert names_out == classnames

    raw[0, 0, 0, 0] += 1
    np.save(tmp_path / "eval_tiny_raw_images.npy", raw, allow_pickle=False)
    with pytest.raises(RuntimeError, match="raw eval digest mismatch"):
        harness.load_eval(tmp_path, "tiny", preprocess)


def test_sparse_seed_filter_drops_empty_and_elapsed_only_records():
    score_task = _load_score_task()

    valid = score_task._valid_seed_metric_records({
        1: {},
        2: {"elapsed_eval": 0.1},
        3: {"acc": "0.5", "elapsed_eval": 0.2},
    })

    assert valid == [{"acc": "0.5", "elapsed_eval": 0.2}]


def test_run_evals_records_elapsed_time(tmp_path: Path):
    score_task = _load_score_task()
    task_meta = tmp_path / "meta"
    eval_root = tmp_path / "eval"
    workspace = tmp_path / "workspace"
    package = workspace / "pkg"
    scripts = eval_root / "scripts"
    task_meta.mkdir()
    scripts.mkdir(parents=True)
    package.mkdir(parents=True)
    (task_meta / "config.json").write_text(json.dumps({
        "test_cmds": [
            {
                "cmd": "scripts/eval.sh",
                "label": "eval",
                "package": "pkg",
                "time": "0:01:00",
                "compute": 1.0,
                "hidden": True,
            }
        ],
        "seeds": [123],
    }))
    (task_meta / "package").write_text("pkg\n")
    (task_meta / "task_id").write_text("elapsed-task\n")
    script = scripts / "eval.sh"
    script.write_text("printf 'acc=0.5\\n'\n")

    rc = score_task.cmd_run_evals(argparse.Namespace(
        task_meta=str(task_meta),
        workspace=str(workspace),
        eval_root=str(eval_root),
        out_dir=str(tmp_path / "out"),
    ))
    summary = json.loads((tmp_path / "out" / "eval_summary.json").read_text())

    assert rc == 0
    assert set(summary[0]) == {"label", "logs"}
    assert summary[0]["logs"][0]["seed"] == 123
    assert isinstance(summary[0]["logs"][0]["elapsed"], float)
    assert summary[0]["logs"][0]["elapsed"] >= 0.0


def test_run_evals_fails_closed_when_command_prints_metric_then_exits_nonzero(tmp_path: Path):
    score_task = _load_score_task()
    task_meta = tmp_path / "meta"
    eval_root = tmp_path / "eval"
    workspace = tmp_path / "workspace"
    package = workspace / "pkg"
    scripts = eval_root / "scripts"
    task_meta.mkdir()
    scripts.mkdir(parents=True)
    package.mkdir(parents=True)
    (task_meta / "config.json").write_text(json.dumps({
        "test_cmds": [{
            "cmd": "scripts/eval.sh",
            "label": "eval",
            "package": "pkg",
            "time": "0:01:00",
            "compute": 1.0,
        }],
        "seeds": [123],
    }))
    (task_meta / "package").write_text("pkg\n")
    (task_meta / "task_id").write_text("failed-eval-task\n")
    (scripts / "eval.sh").write_text("printf 'acc=0.9\\n'\nexit 1\n")
    out_dir = tmp_path / "out"

    rc = score_task.cmd_run_evals(argparse.Namespace(
        task_meta=str(task_meta),
        workspace=str(workspace),
        eval_root=str(eval_root),
        out_dir=str(out_dir),
        oracle_cmd_overrides=None,
    ))

    assert rc == 1
    assert json.loads((out_dir / "eval_summary.json").read_text())[0]["logs"][0]["rc"] == 1
    assert "reward forced to 0" in (out_dir / "score_error.txt").read_text()


def test_run_evals_removes_stale_success_proof_before_failure(tmp_path: Path):
    score_task = _load_score_task()
    task_meta = tmp_path / "meta"
    eval_root = tmp_path / "eval"
    workspace = tmp_path / "workspace"
    package = workspace / "pkg"
    scripts = eval_root / "scripts"
    out_dir = tmp_path / "out"
    task_meta.mkdir()
    scripts.mkdir(parents=True)
    package.mkdir(parents=True)
    out_dir.mkdir()
    (task_meta / "config.json").write_text(json.dumps({
        "test_cmds": [{
            "cmd": "scripts/eval.sh",
            "label": "eval",
            "package": "pkg",
            "time": "0:01:00",
            "compute": 1.0,
        }],
        "seeds": [123],
    }))
    (task_meta / "package").write_text("pkg\n")
    (task_meta / "task_id").write_text("stale-proof-task\n")
    (scripts / "eval.sh").write_text("exit 7\n")
    (out_dir / "metrics.json").write_text('{"reward": 0.9}\n')
    (out_dir / "verification_result.json").write_text(
        '{"status": "passed", "reward": 0.9}\n'
    )
    (out_dir / "stale.log").write_text("acc=0.9\n")

    rc = score_task.cmd_run_evals(argparse.Namespace(
        task_meta=str(task_meta),
        workspace=str(workspace),
        eval_root=str(eval_root),
        out_dir=str(out_dir),
        oracle_cmd_overrides=None,
    ))

    assert rc == 1
    assert not (out_dir / "metrics.json").exists()
    assert not (out_dir / "verification_result.json").exists()
    assert not (out_dir / "stale.log").exists()
    assert "rc=7" in (out_dir / "score_error.txt").read_text()


def test_validate_eval_summary_rejects_partial_or_failed_matrix(tmp_path: Path):
    score_task = _load_score_task()
    ok_log = tmp_path / "ok.log"
    failed_log = tmp_path / "failed.log"
    ok_log.write_text("acc=0.5\n")
    failed_log.write_text("acc=0.9\n")
    config = {
        "test_cmds": [
            {"cmd": "scripts/visible.sh", "label": "visible"},
            {"cmd": "scripts/hidden.sh", "label": "hidden"},
        ],
        "seeds": [42, 43],
    }
    summary = [
        {"label": "visible", "logs": [
            {"seed": 42, "rc": 0, "log": str(ok_log)},
            {"seed": 43, "rc": 1, "log": str(failed_log)},
        ]},
        {"label": "hidden", "logs": [
            {"seed": 42, "rc": 0, "log": str(ok_log)},
        ]},
    ]

    error = score_task._validate_eval_summary(summary, config)

    assert error is not None
    assert "visible seed 43: eval exited with rc=1" in error
    assert "hidden seed 43: expected exactly one log, found 0" in error


def test_validate_eval_summary_requires_exact_labels_and_seeds(tmp_path: Path):
    score_task = _load_score_task()
    log = tmp_path / "ok.log"
    log.write_text("acc=0.5\n")
    config = {
        "test_cmds": [{"cmd": "scripts/eval.sh", "label": "expected"}],
        "seeds": [42],
    }
    summary = [
        {
            "label": "expected",
            "logs": [
                {"seed": 42, "rc": 0, "log": str(log)},
                {"seed": 43, "rc": 0, "log": str(log)},
            ],
        },
        {
            "label": "stale",
            "logs": [{"seed": 42, "rc": 0, "log": str(log)}],
        },
    ]

    error = score_task._validate_eval_summary(summary, config)

    assert error is not None
    assert "unexpected summary labels: stale" in error
    assert "expected: unexpected seeds 43" in error


def test_validate_eval_summary_rejects_empty_success_log(tmp_path: Path):
    score_task = _load_score_task()
    log = tmp_path / "empty.log"
    log.write_text("")
    config = {
        "test_cmds": [{"cmd": "scripts/eval.sh", "label": "eval"}],
        "seeds": [42],
    }
    summary = [{
        "label": "eval",
        "logs": [{"seed": 42, "rc": 0, "log": str(log)}],
    }]

    error = score_task._validate_eval_summary(summary, config)

    assert error is not None
    assert "eval log is empty" in error


@pytest.mark.parametrize(
    "config",
    [
        {
            "test_cmds": [
                {"cmd": "scripts/a.sh", "label": "duplicate"},
                {"cmd": "scripts/b.sh", "label": "duplicate"},
            ],
            "seeds": [42],
        },
        {
            "test_cmds": [{"cmd": "scripts/a.sh", "label": "eval"}],
            "seeds": [42, 42],
        },
    ],
)
def test_validate_eval_summary_rejects_duplicate_config_matrix(config: dict):
    score_task = _load_score_task()

    error = score_task._validate_eval_summary([], config)

    assert error is not None
    assert "duplicate" in error


def test_validate_eval_summary_rejects_harness_fallback_marker(tmp_path: Path):
    score_task = _load_score_task()
    log = tmp_path / "eval.log"
    log.write_text("AGENT_LOAD_FALLBACK RuntimeError('broken')\nacc=0.9\n")
    config = {
        "test_cmds": [{"cmd": "scripts/eval.sh", "label": "eval"}],
        "seeds": [42],
    }
    summary = [{
        "label": "eval",
        "logs": [{"seed": 42, "rc": 0, "log": str(log)}],
    }]

    error = score_task._validate_eval_summary(summary, config)

    assert error is not None
    assert "harness failure marker AGENT_LOAD_FALLBACK" in error


def test_validate_eval_summary_rejects_surface_error_marker(tmp_path: Path):
    score_task = _load_score_task()
    log = tmp_path / "eval.log"
    log.write_text("SURFACE_ERROR: bad solution; using random output\nacc=0.9\n")
    config = {
        "test_cmds": [{"cmd": "scripts/eval.sh", "label": "eval"}],
        "seeds": [42],
    }
    summary = [{
        "label": "eval",
        "logs": [{"seed": 42, "rc": 0, "log": str(log)}],
    }]

    error = score_task._validate_eval_summary(summary, config)

    assert error is not None
    assert "harness failure marker SURFACE_ERROR" in error


@pytest.mark.parametrize(
    ("line", "marker"),
    [
        ("TRAIN_ERROR -> reporting untrained model", "TRAIN_ERROR"),
        ("EVAL_FAILED reason=RuntimeError", "EVAL_FAILED"),
        (
            "PROMPT_CFG build_prompt failed (bad); using object-name fallback",
            "PROMPT_CFG build_prompt failed",
        ),
        ("TOKENSTRAT_CFG set_failed (bad)", "TOKENSTRAT_CFG set_failed"),
        ("LAYER_CFG surgery_failed (bad); running full depth", "LAYER_CFG surgery_failed"),
        ("PROMPT_TEMPLATE_ERROR contrastive-decoding: bad template", "PROMPT_TEMPLATE_ERROR"),
        ("TOKEN_SURFACE_ERROR token_id=7: bad token", "TOKEN_SURFACE_ERROR"),
        ("DETECTOR_ERROR fixed watermark detector: short text", "DETECTOR_ERROR"),
    ],
)
def test_validate_eval_summary_rejects_other_failure_markers(
    tmp_path: Path,
    line: str,
    marker: str,
):
    score_task = _load_score_task()
    log = tmp_path / "eval.log"
    log.write_text(f"{line}\nacc=0.9\n")
    config = {
        "test_cmds": [{"cmd": "scripts/eval.sh", "label": "eval"}],
        "seeds": [42],
    }
    summary = [{
        "label": "eval",
        "logs": [{"seed": 42, "rc": 0, "log": str(log)}],
    }]

    error = score_task._validate_eval_summary(summary, config)

    assert error is not None
    assert f"harness failure marker {marker}" in error


def test_package_dir_matches_case_and_separators(tmp_path: Path):
    score_task = _load_score_task()
    workspace = tmp_path / "workspace"
    actual = workspace / "Nano-GPT"
    actual.mkdir(parents=True)

    resolved = score_task._package_dir(
        workspace,
        "fallback",
        {"package": "nano_gpt"},
    )

    assert resolved == actual


def _write_score_fixture(tmp_path: Path, a_log: str, b_log: str) -> tuple[Path, Path]:
    task_meta = tmp_path / "meta"
    out_dir = tmp_path / "out"
    task_meta.mkdir()
    out_dir.mkdir()
    (task_meta / "config.json").write_text(json.dumps({
        "test_cmds": [
            {"cmd": "scripts/a.sh", "label": "a"},
            {"cmd": "scripts/b.sh", "label": "b"},
        ],
        "seeds": [42],
    }))
    (task_meta / "parser.py").write_text(
        "from mlsbench.agent.parsers import OutputParser, ParseResult\n"
        "class Parser(OutputParser):\n"
        "    def parse(self, cmd_label, raw_output):\n"
        "        metrics = {}\n"
        "        for token in raw_output.split():\n"
        "            parsed = self.parse_metric_assignment(token)\n"
        "            if parsed is not None:\n"
        "                metrics[parsed[0]] = parsed[1]\n"
        "        return ParseResult(feedback=raw_output, metrics=metrics)\n"
    )
    (task_meta / "score_spec.py").write_text(
        "from mlsbench.scoring.dsl import *\n"
        "term('a', col('a').sigmoid(ref=const(0.5), scale=0.1))\n"
        "term('b', col('b').sigmoid(ref=const(0.5), scale=0.1))\n"
        "setting('a', weighted_mean(('a', 1.0)))\n"
        "setting('b', weighted_mean(('b', 1.0)))\n"
        "task(gmean('a', 'b'))\n"
    )
    (task_meta / "leaderboard.csv").write_text(
        "timestamp,model,is_final,seed,a,b\n"
    )
    logs = []
    for label, content in (("a", a_log), ("b", b_log)):
        path = out_dir / f"{label}.log"
        path.write_text(content)
        logs.append({
            "label": label,
            "logs": [{"seed": 42, "rc": 0, "log": str(path), "elapsed": 1.0}],
        })
    (out_dir / "eval_summary.json").write_text(json.dumps(logs))
    return task_meta, out_dir


def test_score_rejects_cross_setting_metric_fill(tmp_path: Path):
    score_task = _load_score_task()
    task_meta, out_dir = _write_score_fixture(
        tmp_path,
        "a=0.8 b=0.8\n",
        "junk=1\n",
    )
    reward = out_dir / "reward.txt"
    reward.write_text("0.91\n")
    (out_dir / "metrics.json").write_text('{"reward": 0.91}\n')
    (out_dir / "verification_result.json").write_text(
        '{"status": "passed", "reward": 0.91}\n'
    )

    rc = score_task.cmd_score(argparse.Namespace(
        task_meta=str(task_meta),
        out_dir=str(out_dir),
        reward_out=str(reward),
    ))

    assert rc == 0
    assert reward.read_text().strip() == "0"
    assert "exactly one score setting" in (out_dir / "score_error.txt").read_text()
    assert not (out_dir / "metrics.json").exists()
    assert not (out_dir / "verification_result.json").exists()


def test_score_rejects_agent_atexit_metric_override(tmp_path: Path):
    score_task = _load_score_task()
    task_meta, out_dir = _write_score_fixture(
        tmp_path,
        # The first record represents the trusted harness result. An editable
        # solution can register an atexit callback that prints the second record
        # after main() returns; the fixture parser is intentionally last-wins.
        "a=0.1\na=0.99\n",
        "b=0.8\n",
    )
    reward = out_dir / "reward.txt"

    rc = score_task.cmd_score(argparse.Namespace(
        task_meta=str(task_meta),
        out_dir=str(out_dir),
        reward_out=str(reward),
    ))

    assert rc == 0
    assert reward.read_text().strip() == "0"
    assert "duplicate authoritative metric" in (
        out_dir / "score_error.txt"
    ).read_text()
    assert not (out_dir / "metrics.json").exists()
    assert not (out_dir / "verification_result.json").exists()


def test_score_rejects_invalid_extra_parser_metric_before_filtering(tmp_path: Path):
    score_task = _load_score_task()
    task_meta, out_dir = _write_score_fixture(
        tmp_path,
        "a=0.8\n",
        "b=0.8\n",
    )
    (task_meta / "parser.py").write_text(
        "from mlsbench.agent.parsers import OutputParser, ParseResult\n"
        "class Parser(OutputParser):\n"
        "    def parse(self, cmd_label, raw_output):\n"
        "        key = cmd_label\n"
        "        metrics = {key: 0.8}\n"
        "        if cmd_label == 'a':\n"
        "            metrics['ignored_std'] = float('nan')\n"
        "        return ParseResult(feedback=raw_output, metrics=metrics)\n"
    )
    reward = out_dir / "reward.txt"

    rc = score_task.cmd_score(argparse.Namespace(
        task_meta=str(task_meta),
        out_dir=str(out_dir),
        reward_out=str(reward),
    ))

    assert rc == 0
    assert reward.read_text().strip() == "0"
    assert "invalid parser metric" in (out_dir / "score_error.txt").read_text()
    assert not (out_dir / "metrics.json").exists()
    assert not (out_dir / "verification_result.json").exists()


def test_score_validation_error_publishes_zero(tmp_path: Path):
    score_task = _load_score_task()
    task_meta, out_dir = _write_score_fixture(
        tmp_path,
        "a=0.8\n",
        "b=0.8\n",
    )
    (task_meta / "score_spec.py").write_text(
        "from mlsbench.scoring.dsl import *\n"
        "term('a', col('a').sigmoid(ref=const(0.5), scale=0.1))\n"
        "term('b', col('b').sigmoid(ref=const(0.5), scale=0.1))\n"
        "setting('a', weighted_mean(('a', 0.0)))\n"
        "setting('b', weighted_mean(('b', 1.0)))\n"
        "task(gmean('a', 'b'))\n"
    )
    reward = out_dir / "reward.txt"

    rc = score_task.cmd_score(argparse.Namespace(
        task_meta=str(task_meta),
        out_dir=str(out_dir),
        reward_out=str(reward),
    ))

    assert rc == 0
    assert reward.read_text().strip() == "0"
    assert "invalid score specification" in (out_dir / "score_error.txt").read_text()
    assert not (out_dir / "metrics.json").exists()
    assert not (out_dir / "verification_result.json").exists()


def test_score_hash_survives_mangrove_json_transport(tmp_path: Path):
    score_task = _load_score_task()
    task_meta, out_dir = _write_score_fixture(
        tmp_path,
        "a=0.8\n",
        "b=0.8\n",
    )
    reward = out_dir / "reward.txt"

    rc = score_task.cmd_score(argparse.Namespace(
        task_meta=str(task_meta),
        out_dir=str(out_dir),
        reward_out=str(reward),
    ))

    assert rc == 0
    assert float(reward.read_text()) > 0.0
    proof = json.loads((out_dir / "verification_result.json").read_text())
    assert proof["status"] == "passed"
    metrics_text = (out_dir / "metrics.json").read_text()
    assert proof["metrics_sha256"] == hashlib.sha256(metrics_text.encode()).hexdigest()
    mangrove_transport_text = json.dumps(
        json.loads(metrics_text),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    assert mangrove_transport_text == metrics_text
    assert proof["metrics_sha256"] == hashlib.sha256(
        mangrove_transport_text.encode()
    ).hexdigest()


def test_test_sh_wires_sanitized_meta_only_to_run_evals():
    script = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "mls_bench"
        / "task-template"
        / "tests"
        / "test.sh"
    ).read_text()
    guard_block = script.split(" guard \\\n", 1)[1].split("guard_rc=$?", 1)[0]
    eval_block = script.split(" run-evals \\\n", 1)[1].split("|| _RUN_EVALS_RC", 1)[0]

    assert "--eval-task-meta" not in guard_block
    assert '--eval-task-meta "${EVAL_META}"' in eval_block


def test_test_sh_preserves_zero_until_success_proof_is_committed():
    script = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "mls_bench"
        / "task-template"
        / "tests"
        / "test.sh"
    ).read_text()

    zero_init = script.index("printf '0\\n' > /logs/verifier/reward.txt")
    run_evals = script.index(' run-evals \\\n')
    score = script.index(' score \\\n')
    candidate = script.index('_CANDIDATE_REWARD="/logs/verifier/.reward.candidate"')
    proof = script.index('/logs/verifier/verification_result.json', score)
    publish = script.index(
        'mv -f -- "${_CANDIDATE_REWARD}" /logs/verifier/reward.txt'
    )
    commit = script.index("_VERIFICATION_COMMITTED=1", publish)
    assert zero_init < candidate < run_evals < score < proof < publish < commit
    score_block = script.split(' score \\\n', 1)[1].split("_PROOF_RC=0", 1)[0]
    assert '--reward-out "${_CANDIDATE_REWARD}"' in score_block
    assert "--reward-out /logs/verifier/reward.txt" not in score_block
    assert "canonical_proof_text" in script
    assert '_remove_reward_candidate' in script
    assert 'if [ "${_VERIFICATION_COMMITTED:-0}" -ne 1 ]; then' in script
    assert "trap _abort_verifier HUP INT TERM" in script
    assert "metrics_sha256" in script


def test_eval_preexec_drops_root_privileges(monkeypatch):
    if os.geteuid() != 0:
        pytest.skip("privilege-drop assertion requires a root test process")
    score_task = _load_score_task()
    monkeypatch.setenv("MLSBENCH_EVAL_UID", "65534")
    monkeypatch.setenv("MLSBENCH_EVAL_GID", "65534")

    proc = subprocess.run(
        ["id", "-u"],
        check=True,
        capture_output=True,
        text=True,
        preexec_fn=score_task._eval_preexec_fn(),
    )

    assert proc.stdout.strip() == "65534"


def _stage_verifier_shell_fixture(
    tmp_path: Path,
    *,
    pause_before_proof: bool = False,
) -> tuple[Path, Path, Path, Path, Path | None]:
    repo_root = Path(__file__).resolve().parents[2]
    fixture_root = tmp_path / "verifier-smoke"
    tests_root = fixture_root / "tests"
    logs_root = fixture_root / "logs"
    workspace = fixture_root / "workspace"
    fake_root = fixture_root / "root"
    solution_root = fixture_root / "solution"
    task_meta = tests_root / "meta"
    eval_scripts = tests_root / "eval" / "scripts"
    package = workspace / "pkg"
    pristine_package = task_meta / "pristine" / "pkg"
    for directory in (
        logs_root,
        package,
        pristine_package,
        eval_scripts,
        fake_root / ".cache",
        solution_root,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    # pytest's per-run parents are normally mode 0700. The verifier's nobody
    # process must be able to traverse to the staged eval script and workspace.
    for parent in [fixture_root, *fixture_root.parents]:
        if parent == Path("/"):
            break
        parent.chmod(parent.stat().st_mode | 0o055)
        if parent == Path("/tmp"):
            break

    source = "def native_solution():\n    return 1\n"
    (package / "solution.py").write_text(source)
    (pristine_package / "solution.py").write_text(source)
    source_sha = hashlib.sha256(source.encode()).hexdigest()
    config = {
        "use_cuda": False,
        "files": [{
            "filename": "pkg/solution.py",
            "edit": [{"start": -1, "end": -1}],
        }],
        "test_cmds": [{
            "cmd": "scripts/eval.sh",
            "label": "eval",
            "package": "pkg",
            "compute": 0,
            "time": "0:00:10",
        }],
        "seeds": [42],
    }
    (task_meta / "config.json").write_text(json.dumps(config))
    (task_meta / "pristine_manifest.json").write_text(json.dumps({
        "pkg/solution.py": source_sha,
    }))
    (task_meta / "task_id").write_text("smoke-task\n")
    (task_meta / "package").write_text("pkg\n")
    (task_meta / "workdir").write_text(f"{workspace}\n")
    (task_meta / "parser.py").write_text(
        "import re\n"
        "from mlsbench.agent.parsers import OutputParser, ParseResult\n"
        "class Parser(OutputParser):\n"
        "    def parse(self, cmd_label, raw_output):\n"
        "        match = re.search(r'score=([0-9.]+)', raw_output)\n"
        "        metrics = {'score': float(match.group(1))} if match else {}\n"
        "        return ParseResult(feedback=raw_output, metrics=metrics)\n"
    )
    (task_meta / "score_spec.py").write_text(
        "from mlsbench.scoring.dsl import *\n"
        "term('score', col('score').sigmoid(ref=const(0.0), scale=1.0))\n"
        "setting('eval', weighted_mean(('score', 1.0)))\n"
        "task(gmean('eval'))\n"
    )
    (task_meta / "leaderboard.csv").write_text(
        "timestamp,model,is_final,seed,score\n"
    )
    eval_script = eval_scripts / "eval.sh"
    eval_script.write_text(
        "#!/bin/bash\n"
        "set -euo pipefail\n"
        "test \"$(id -u)\" = 65534\n"
        "test -r \"${TASK_DIR}/config.json\"\n"
        "test ! -r \"${TASK_DIR}/parser.py\"\n"
        "test ! -r \"${TASK_DIR}/score_spec.py\"\n"
        "test ! -r \"${TASK_DIR}/leaderboard.csv\"\n"
        "test ! -w solution.py\n"
        "mkdir -p \"${XDG_CACHE_HOME}\"\n"
        "printf 'home-ok\\n' > \"${HOME}/home-artifact.txt\"\n"
        "printf 'cache-ok\\n' > \"${XDG_CACHE_HOME}/cache-artifact.txt\"\n"
        "mkdir -p \"${OUTPUT_DIR}\"\n"
        "printf 'artifact-ok\\n' > \"${OUTPUT_DIR}/artifact.txt\"\n"
        "printf 'artifact_write=ok\\nscore=1.0\\n'\n"
    )
    eval_script.chmod(0o755)

    shutil.copytree(
        repo_root / "src" / "mlsbench",
        tests_root / "mlsbench_src" / "mlsbench",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    score_source = (
        repo_root
        / "harbor_adapter"
        / "src"
        / "mls_bench"
        / "task-template"
        / "tests"
        / "score_task.py"
    ).read_text()
    score_source = score_source.replace("/tests", str(tests_root))
    score_source = score_source.replace("/workspace", str(workspace))
    (tests_root / "score_task.py").write_text(score_source)

    test_source = (
        repo_root
        / "harbor_adapter"
        / "src"
        / "mls_bench"
        / "task-template"
        / "tests"
        / "test.sh"
    ).read_text()
    for original, replacement in (
        ("/tests", str(tests_root)),
        ("/logs", str(logs_root)),
        ("/workspace", str(workspace)),
        ("/solution", str(solution_root)),
        ("/root", str(fake_root)),
    ):
        test_source = test_source.replace(original, replacement)

    pause_marker = None
    if pause_before_proof:
        pause_marker = fixture_root / "score-stage-complete"
        pause_release = fixture_root / "release-proof-check"
        proof_boundary = "\n_PROOF_RC=1\n"
        assert test_source.count(proof_boundary) == 1
        test_source = test_source.replace(
            proof_boundary,
            (
                f'\nprintf "ready\\n" > "{pause_marker}"\n'
                f'while [ ! -e "{pause_release}" ]; do sleep 0.05; done\n'
                "\n_PROOF_RC=1\n"
            ),
            1,
        )
    test_script = tests_root / "test.sh"
    test_script.write_text(test_source)
    test_script.chmod(0o755)

    return (
        fixture_root,
        test_script,
        logs_root / "verifier",
        package / "solution.py",
        pause_marker,
    )


def _verifier_env() -> dict[str, str]:
    return {
        **os.environ,
        "MLSBENCH_VERIFIER_LOG_INTERVAL_SEC": "9999",
    }


def _wait_for_path(path: Path, proc: subprocess.Popen, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return
        returncode = proc.poll()
        if returncode is not None:
            stdout, stderr = proc.communicate()
            pytest.fail(
                f"verifier exited with rc={returncode} before {path} appeared\n"
                f"stdout:\n{stdout}\nstderr:\n{stderr}"
            )
        time.sleep(0.01)
    proc.kill()
    stdout, stderr = proc.communicate()
    pytest.fail(
        f"timed out waiting for {path}\nstdout:\n{stdout}\nstderr:\n{stderr}"
    )


def test_verifier_shell_end_to_end_with_unchanged_native_solution(tmp_path: Path):
    if os.geteuid() != 0:
        pytest.skip("full verifier smoke requires root to exercise uid drop")

    fixture_root, test_script, verifier_logs, solution_file, _ = (
        _stage_verifier_shell_fixture(tmp_path)
    )

    result = subprocess.run(
        ["bash", str(test_script)],
        cwd=str(fixture_root),
        env=_verifier_env(),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert float((verifier_logs / "reward.txt").read_text()) > 0.0, (
        result.stdout + result.stderr
    )
    assert solution_file.read_text() == "def native_solution():\n    return 1\n"
    assert "artifact_write=ok" in (verifier_logs / "eval__seed42.log").read_text()
    proof = json.loads((verifier_logs / "verification_result.json").read_text())
    assert proof["status"] == "passed"
    assert proof["strict_fail_closed"] is True
    assert not (verifier_logs / ".reward.candidate").exists()


def test_verifier_sigkill_before_proof_keeps_public_reward_exact_zero(tmp_path: Path):
    if os.geteuid() != 0:
        pytest.skip("full verifier smoke requires root to exercise uid drop")

    fixture_root, test_script, verifier_logs, _solution_file, pause_marker = (
        _stage_verifier_shell_fixture(tmp_path, pause_before_proof=True)
    )
    assert pause_marker is not None
    proc = subprocess.Popen(
        ["bash", str(test_script)],
        cwd=str(fixture_root),
        env=_verifier_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    _wait_for_path(pause_marker, proc)
    candidate = verifier_logs / ".reward.candidate"
    assert float(candidate.read_text()) > 0.0
    assert (verifier_logs / "reward.txt").read_text() == "0\n"

    os.kill(proc.pid, signal.SIGKILL)
    proc.communicate(timeout=5)

    assert proc.returncode == -signal.SIGKILL
    assert (verifier_logs / "reward.txt").read_text() == "0\n"


def test_verifier_caught_signal_removes_reward_candidate(tmp_path: Path):
    if os.geteuid() != 0:
        pytest.skip("full verifier smoke requires root to exercise uid drop")

    fixture_root, test_script, verifier_logs, _solution_file, pause_marker = (
        _stage_verifier_shell_fixture(tmp_path, pause_before_proof=True)
    )
    assert pause_marker is not None
    proc = subprocess.Popen(
        ["bash", str(test_script)],
        cwd=str(fixture_root),
        env=_verifier_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    _wait_for_path(pause_marker, proc)
    candidate = verifier_logs / ".reward.candidate"
    assert float(candidate.read_text()) > 0.0

    os.kill(proc.pid, signal.SIGTERM)
    stdout, stderr = proc.communicate(timeout=5)

    assert proc.returncode == 0, stdout + stderr
    assert (verifier_logs / "reward.txt").read_text() == "0\n"
    assert not candidate.exists()


def test_verifier_invalid_proof_removes_candidate_and_keeps_zero(tmp_path: Path):
    if os.geteuid() != 0:
        pytest.skip("full verifier smoke requires root to exercise uid drop")

    fixture_root, test_script, verifier_logs, _solution_file, pause_marker = (
        _stage_verifier_shell_fixture(tmp_path, pause_before_proof=True)
    )
    assert pause_marker is not None
    proc = subprocess.Popen(
        ["bash", str(test_script)],
        cwd=str(fixture_root),
        env=_verifier_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    _wait_for_path(pause_marker, proc)
    candidate = verifier_logs / ".reward.candidate"
    assert float(candidate.read_text()) > 0.0
    proof_path = verifier_logs / "verification_result.json"
    proof_path.write_text(proof_path.read_text() + "\n")
    (fixture_root / "release-proof-check").touch()

    stdout, stderr = proc.communicate(timeout=5)

    assert proc.returncode == 0, stdout + stderr
    assert (verifier_logs / "reward.txt").read_text() == "0\n"
    assert not candidate.exists()
    assert not proof_path.exists()
    assert not (verifier_logs / "metrics.json").exists()
    assert "invalid success proof" in (verifier_logs / "score_error.txt").read_text()
