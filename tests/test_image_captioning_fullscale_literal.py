from __future__ import annotations

import ast
import importlib.util
import json
import runpy
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
TASKS = tuple(sorted(path.parent for path in (ROOT / "tasks").glob("caption-*/config.json")))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def harness():
    return _load("caption_fullscale_harness", ROOT / "vendor/image-captioning/harness.py")


def _materialize(source: str, operation: dict) -> str:
    lines = source.splitlines()
    start, end = int(operation["start_line"]), int(operation["end_line"])
    return "\n".join(
        lines[: start - 1] + str(operation["content"]).splitlines() + lines[end:]
    ) + "\n"


def test_official_protocol_scale_is_fixed(harness):
    assert harness.PROTOCOL == "flickr8k_official_v1"
    assert (harness.TRAIN_IMAGES, harness.EVAL_IMAGES) == (6000, 1000)
    assert harness.REFS_PER_IMAGE == 5
    assert harness.TRAIN_PAIRS == 30000
    assert (harness.EPOCHS, harness.BATCH_SIZE, harness.EXPECTED_STEPS) == (10, 40, 7500)

    prep = _load(
        "caption_fullscale_prepare",
        ROOT / "vendor/data_scripts/image-captioning/prepare_data.py",
    )
    assert (prep.TRAIN_SPLIT, prep.EVAL_SPLIT) == ("train", "test")
    assert (prep.N_TRAIN, prep.N_EVAL, prep.REFS_PER_IMAGE) == (6000, 1000, 5)


def test_canonical_filename_proof_is_pinned_and_recomputed(harness):
    prep = _load(
        "caption_fullscale_prepare_proof",
        ROOT / "vendor/data_scripts/image-captioning/prepare_data.py",
    )
    assert prep.MANIFEST_SCHEMA_VERSION == harness.MANIFEST_SCHEMA_VERSION == 3
    assert prep.CANONICAL_ARCHIVE_SHA256 == harness.CANONICAL_ARCHIVE_SHA256
    assert prep.CANONICAL_JSON_SHA256 == harness.CANONICAL_JSON_SHA256
    assert (
        prep.CANONICAL_FILENAME_SET_SHA256
        == harness.CANONICAL_FILENAME_SET_SHA256
        == {
            "train": "fbb334d8b4d4bab05a65950cb0b8123079c40ba8d1c38d8aa360fa27459e8cf4",
            "test": "25d2fec0836bb4728d4672c46a5694dfbdb953a2ff5ba146f5ffaa7062512489",
        }
    )

    captions = [
        "A cafe scene .",
        "Two people sit .",
        "A table outside .",
        "A street view .",
        "People share a meal .",
    ]
    source = {
        "source_filename": "123_example.jpg",
        "decoded_rgb_sha256": "a" * 64,
        "captions_sha256": prep._captions_sha256(captions),
        "canonical_captions_sha256": prep._caption_signature_sha256(captions),
    }
    assert source["captions_sha256"] == harness._json_sha256(captions)
    assert source["canonical_captions_sha256"] == (
        harness._caption_signature_sha256(captions)
    )
    assert prep._split_sha256([source]) == harness._source_split_sha256([source])
    assert prep._filename_set_sha256(["b.jpg", "a.jpg"]) == (
        prep._filename_set_sha256(["a.jpg", "b.jpg"])
    )


def test_model_provenance_is_bound_to_fixed_checkpoint_hashes(harness):
    prep = _load(
        "caption_fullscale_prepare_models",
        ROOT / "vendor/data_scripts/image-captioning/prepare_data.py",
    )
    expected_gpt2 = {
        "model.safetensors": "c7d00560d8910fbed77ffad4065dee5011c41ba401b1064e749c498ba9e20373",
        "config.json": "50fda00afcbf90d2a7655c764fd8879f6ce8bed5624ff8231cae8889a7983cd4",
        "tokenizer.json": "1fe93b6152957cf9cfd6d89002467f789ce8b3f3e000b3a2edf27c808ddd0b9e",
    }
    assert prep.GPT2_FILE_SHA256 == harness.GPT2_FILE_SHA256 == expected_gpt2
    assert (
        prep.CLIP_CHECKPOINT_SHA256
        == harness.CLIP_CHECKPOINT_SHA256
        == "1bd3c7172de5b207ceac554f5ab5266166f3b9baccc9af5989bc801016d080ad"
    )
    metadata = prep._manifest_metadata()
    assert metadata["gpt2_file_sha256"] == expected_gpt2
    assert metadata["clip_pretrained_tag"] == "laion2b_s34b_b79k"
    assert metadata["clip_repo"] == "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
    assert metadata["clip_checkpoint"] == "open_clip_pytorch_model.bin"
    assert metadata["clip_checkpoint_sha256"] == prep.CLIP_CHECKPOINT_SHA256


def test_training_loss_supervises_real_eos_but_not_padding(harness):
    # GPT-2 uses EOS as its padding id here. The attention mask, rather than
    # token identity, must distinguish the real EOS target from padded EOS.
    logits = torch.tensor(
        [[[0.0, 2.0, -1.0], [0.0, -2.0, 3.0], [10.0, -10.0, -10.0]]],
        requires_grad=True,
    )
    targets = torch.tensor([[1, 2, 2]])
    attention = torch.tensor([[1, 1, 0]])
    loss = harness._caption_loss(logits, targets, attention, None, None)
    expected = torch.nn.functional.cross_entropy(logits[0, :2], targets[0, :2])
    assert float(loss.detach()) == pytest.approx(float(expected.detach()), rel=1e-7)
    loss.backward()
    assert logits.grad is not None
    assert torch.count_nonzero(logits.grad[0, 1]).item() > 0
    assert torch.count_nonzero(logits.grad[0, 2]).item() == 0


def test_smoothed_idf_keeps_ubiquitous_eos_supervised(harness):
    sequences = [[7] for _ in range(harness.TRAIN_PAIRS)]
    sequences[0].append(8)
    weights = harness._idf_lut(sequences, vocab_size=10, device=torch.device("cpu"))
    assert float(weights[7]) == pytest.approx(1.0)
    assert float(weights[8]) > 1.0
    assert torch.isfinite(weights).all()


def test_every_sampling_strategy_covers_each_pair_once_per_epoch(harness):
    sequences = [[0] * (1 + index % 20) for index in range(harness.TRAIN_PAIRS)]
    expected = torch.arange(harness.TRAIN_PAIRS)
    orders = {}
    for strategy in ("uniform", "length_bucketed"):
        order = harness._epoch_order({"strategy": strategy}, sequences, 42, 0)
        assert torch.equal(torch.sort(order).values, expected)
        assert torch.equal(
            order,
            harness._epoch_order({"strategy": strategy}, sequences, 42, 0),
        )
        orders[strategy] = order

    def mean_batch_span(order):
        spans = []
        for start in range(0, harness.TRAIN_PAIRS, harness.BATCH_SIZE):
            lengths = [len(sequences[index]) for index in order[start : start + harness.BATCH_SIZE]]
            spans.append(max(lengths) - min(lengths))
        return sum(spans) / len(spans)

    assert mean_batch_span(orders["length_bucketed"]) < mean_batch_span(orders["uniform"])


def test_ten_siblings_use_static_literal_surfaces(harness, tmp_path: Path):
    assert len(TASKS) == 10
    modes = set()
    files = set()
    for task_dir in TASKS:
        config = json.loads((task_dir / "config.json").read_text())
        assert config["rigorous_codebase"] is False
        assert config["seeds"] == [42]
        assert config["verifier_only_package_files"] == ["image-captioning/harness.py"]
        assert config["agent_image_prune"] == ["/opt/mlsbench-caption"]
        assert len(config["test_cmds"]) == 1
        assert config["test_cmds"][0]["label"] == "flickr"
        assert len(config["files"]) == 1

        declared = config["files"][0]
        relative = Path(declared["filename"]).relative_to("image-captioning")
        solution = ROOT / "vendor/image-captioning" / relative
        tree = ast.parse(solution.read_text())
        assert not any(
            isinstance(node, (ast.Call, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            for node in ast.walk(tree)
        )
        script = (task_dir / "scripts/run.sh").read_text()
        mode = script.split("--mode ", 1)[1].split()[0]
        modes.add(mode)
        files.add(solution.name)
        native = harness.load_literal_config(solution, mode)
        assert native

        for baseline in config["baselines"].values():
            operations = runpy.run_path(str(task_dir / baseline["edit_ops"]))["OPS"]
            assert len(operations) == 1
            operation = operations[0]
            assert operation["op"] == "replace"
            assert operation["file"] == declared["filename"]
            candidate = tmp_path / f"{task_dir.name}-{Path(baseline['edit_ops']).stem}.py"
            candidate.write_text(_materialize(solution.read_text(), operation))
            assert harness.load_literal_config(candidate, mode)

    assert modes == {
        "mapping",
        "decoding",
        "objective",
        "featureprep",
        "init",
        "sampling",
        "optimizer",
        "prompt",
        "augment",
        "weighting",
    }
    assert len(files) == 10


def test_literal_loader_never_executes_candidate_code(harness, tmp_path: Path):
    marker = tmp_path / "executed"
    candidate = tmp_path / "candidate.py"
    candidate.write_text(
        '"""candidate"""\n'
        "from __future__ import annotations\n"
        "CONFIG = {'label_smoothing': 0.1}\n"
        f"Path({str(marker)!r}).write_text('bad')\n"
    )
    with pytest.raises(ValueError, match="may contain only"):
        harness.load_literal_config(candidate, "objective")
    assert not marker.exists()

    candidate.write_text("CONFIG = {'label_smoothing': float('nan')}\n")
    with pytest.raises(ValueError, match="literals only"):
        harness.load_literal_config(candidate, "objective")
    assert not marker.exists()


def test_every_data_artifact_is_verifier_only():
    names = {
        "source_manifest.json",
        "train_clip.pt",
        "train_refs.json",
        "eval_clip.pt",
        "eval_refs.json",
    }
    for task_dir in TASKS:
        config = json.loads((task_dir / "config.json").read_text())
        assert config["agent_image_prune"] == ["/opt/mlsbench-caption"]
        pruned = {Path(path).name for path in config["agent_data_prune"]}
        staged = {Path(entry["host_path"]).name for entry in config["verifier_data_deps"]}
        assert pruned == names
        assert staged == names
        assert all(entry["required"] is True for entry in config["verifier_data_deps"])

    package = json.loads(
        (ROOT / "vendor/pkg_configs/image-captioning/config.json").read_text()
    )
    assert "baselines" in package["agent_pruned_files"]
    assert "2.7.0-cuda12.8" in package["base_image"]
    assert package["mangrove_base_image"].endswith(
        "@sha256:2bc773cf6e838e9defe1b06e20efde3d93a3690b1b09e7cd2ea29217c120e7c7"
    )


def _result(mode: str, **overrides) -> str:
    fields = {
        "protocol": "flickr8k_official_v1",
        "mode": mode,
        "train_images": 6000,
        "train_pairs": 30000,
        "eval_images": 1000,
        "epochs": 10,
        "batch_size": 40,
        "steps": 7500,
        "seed": 42,
        "split_sha256": "a" * 64,
        "manifest_sha256": "b" * 64,
        "predictions_sha256": "c" * 64,
        "cider": "0.725000",
        "bleu4": "0.180000",
        "status": "ok",
    }
    fields.update(overrides)
    order = (
        "protocol",
        "mode",
        "train_images",
        "train_pairs",
        "eval_images",
        "epochs",
        "batch_size",
        "steps",
        "seed",
        "split_sha256",
        "manifest_sha256",
        "predictions_sha256",
        "cider",
        "bleu4",
        "status",
    )
    return "CAPTION_RESULT " + " ".join(f"{key}={fields[key]}" for key in order)


def test_parsers_require_one_final_full_protocol_completion_record():
    for task_dir in TASKS:
        parser_module = _load(f"parser_{task_dir.name}", task_dir / "parser.py")
        parser = parser_module.Parser()
        valid = _result(parser_module.EXPECTED_MODE)
        assert parser.parse("flickr", "progress\n" + valid + "\n").metrics == {
            "cider_flickr": 0.725,
            "bleu4_flickr": 0.18,
        }
        assert parser.parse("wrong-label", valid).metrics == {}
        invalid_logs = (
            "",
            "verification failed\nscore=0.9",
            _result(parser_module.EXPECTED_MODE, train_images=2000),
            valid + "\ntrailing output",
            "Traceback (most recent call last)\n" + valid,
            valid + "\n" + valid,
            _result(parser_module.EXPECTED_MODE, cider="nan"),
        )
        for raw_output in invalid_logs:
            assert parser.parse("flickr", raw_output).metrics == {}
        for marker in parser_module.FAILURE_MARKERS:
            assert parser.parse("flickr", marker + "\n" + valid).metrics == {}


def test_full_official_anchors_replace_old_subset_measurements():
    from mlsbench.scoring.spec import load_score_spec, validate_score_spec

    for task_dir in TASKS:
        assert not (task_dir / "PENDING_FULL_OFFICIAL_ANCHORS").exists()
        score_source = (task_dir / "score_spec.py").read_text()
        assert "Measured full-protocol Flickr8k calibration" in score_source
        assert "floor=const(0.245972)" in score_source
        assert "ref=const(0.218874)" in score_source
        leaderboard = (task_dir / "leaderboard.csv").read_text().splitlines()
        assert len(leaderboard) == 4
        assert "calibration:decoding_sample" in leaderboard[1]
        assert "calibration:decoding_greedy" in leaderboard[2]
        assert "calibration:decoding_beam5" in leaderboard[3]
        spec = load_score_spec(task_dir)
        assert spec is not None
        assert validate_score_spec(spec, ["cider_flickr", "bleu4_flickr"]) == []


def test_measured_caption_calibration_is_ordered_and_fail_closed():
    from mlsbench.scoring.anchors import BaselineAnchors
    from mlsbench.scoring.evaluate import score_record_details
    from mlsbench.scoring.spec import load_score_spec

    measured_records = (
        {"cider_flickr": 0.245972, "bleu4_flickr": 0.076101},
        {"cider_flickr": 0.552901, "bleu4_flickr": 0.198662},
        {"cider_flickr": 0.586622, "bleu4_flickr": 0.218874},
    )
    invalid_records = (
        {},
        {"cider_flickr": 1.25},
        {"cider_flickr": float("nan"), "bleu4_flickr": 0.25},
    )
    for task_dir in TASKS:
        spec = load_score_spec(task_dir)
        assert spec is not None
        anchors = BaselineAnchors(task_dir)
        measured_scores = []
        for record in measured_records:
            score, settings, valid = score_record_details(spec, record, anchors)
            assert valid is True
            assert len(settings) == 1
            assert settings[0].score == pytest.approx(score)
            measured_scores.append(score)
        assert measured_scores[0] == 0.0
        assert 0.0 < measured_scores[1] < measured_scores[2]
        assert measured_scores[2] == pytest.approx(0.5)
        for record in invalid_records:
            score, _, valid = score_record_details(spec, record, anchors)
            assert valid is False
            assert score == 0.0


def test_metrics_match_coco_caption_reference_fixture(harness):
    references = [
        [
            "a dog runs in the grass",
            "a brown dog runs outside",
            "dog running on grass",
            "a dog is running",
            "brown dog in a field",
        ],
        [
            "two people ride bicycles",
            "a pair rides bikes",
            "people on bicycles",
            "two cyclists ride",
            "riders on bikes",
        ],
        [
            "a cat sits near a window",
            "cat by the window",
            "a sitting cat",
            "cat is sitting indoors",
            "a feline near glass",
        ],
    ]
    hypotheses = [
        "a dog runs on grass",
        "two people ride bikes",
        "a cat sits by the window",
    ]
    # Values independently computed with pycocoevalcap 1.2 after the same
    # deterministic punctuation/lowercase tokenization.
    cider, bleu = harness.caption_metrics(hypotheses, references)
    assert cider == pytest.approx(1.8865850385016827, rel=1e-12)
    assert bleu == pytest.approx(8.633400212205652e-05, rel=1e-12)
    assert harness._ptb_tokenize(["A dark-haired man's t-shirt."]) == [
        "a dark-haired man 's t-shirt"
    ]


def test_data_prep_proof_uses_the_claimed_dynamic_run_directory():
    script = (ROOT / "scripts/prepare_caption_full_v2_worker.sh").read_text()
    assert 'stage=${CAPTION_STAGE:-/mnt/moonfs/lvbohan-b0/image-captioning-full-v1}' in script
    assert 'run_id="${CAPTION_PREP_RUN_ID:-official-v2-streaming-canonical}"' in script
    assert 'python - "${stage}" "${run}" <<\'PY\'' in script
    assert "root = Path(sys.argv[1])" in script
    assert "run = Path(sys.argv[2])" in script
    assert 'run = root / "data-prep/official-v2-streaming-canonical"' not in script
    assert '"${run}/data-proof.json"' in script
    assert 'rc=70' in script
    assert "trap 'exit 130' INT" in script
    assert 'exec >> "${run}/worker.log" 2>&1' in script
    assert "exec > >(tee" not in script
    assert '(cd "${repo}" && sha256sum -c "${stage}/repo.sha256")' in script

    image_builder = (ROOT / "scripts/build_caption_full_image.sh").read_text()
    assert '"${BUILD}/image.ref"' in image_builder
    assert '"${BUILD}/image.digest"' in image_builder
    assert '"${BUILD}/IMAGE_READY"' in image_builder
    assert "caption image success gate is missing" in image_builder
    assert "trap 'exit 130' INT" in image_builder
    assert 'exec >> "${BUILD}/build.log" 2>&1' in image_builder
    assert "exec > >(tee" not in image_builder
    assert (
        '(cd "${STAGED_REPO}" && sha256sum -c "${STAGE}/repo.sha256")'
        in image_builder
    )
    assert "rsync " not in image_builder
    assert 'tar -C "${DATA_ROOT}" -cf - .' in image_builder

    anchor_launcher = (ROOT / "scripts/mlaunch_caption_full_anchor_cell.sh").read_text()
    assert 'STAGE_MOUNT_ROOT=${CAPTION_STAGE_MOUNT_ROOT:-$(dirname "${STAGE_ROOT}")}' in anchor_launcher
    assert '--volume "${STAGE_MOUNT_ROOT}:${STAGE_MOUNT_ROOT}"' in anchor_launcher
    assert "--volume /mnt/moonfs/lvbohan-b0:/mnt/moonfs/lvbohan-b0" not in anchor_launcher
    assert "RC_GROUP=${CAPTION_RC_GROUP:-}" in anchor_launcher
    assert 'rc_group_args=(-g "${RC_GROUP}")' in anchor_launcher
    assert '"${rc_group_args[@]}"' in anchor_launcher
    assert 'priority_args=(--preemptible=yes)' in anchor_launcher
    assert '"${priority_args[@]}"' in anchor_launcher
    assert "cd '${RUN}' || exit 111" in anchor_launcher

    wave_launcher = (ROOT / "scripts/mlaunch_caption_full_anchor_wave.sh").read_text()
    assert '--gpu=8' in wave_launcher
    assert '--preemptible=yes' in wave_launcher
    assert 'PRIORITY=${CAPTION_WAVE_PRIORITY:-599}' in wave_launcher
    assert 'gpu_per_cell=1' in wave_launcher
    assert 'CUDA_VISIBLE_DEVICES=${gpu} python' in wave_launcher
    assert 'torch.cuda.device_count() == 8' in wave_launcher
    assert 'optimizer_steps_per_cell=7500' in wave_launcher
    assert 'WAVE_PROFILE=${CAPTION_WAVE_PROFILE:-decoding3}' in wave_launcher
    assert "caption-visual-mapping/linear:0" in wave_launcher
    assert "caption-feature-prep/l2:4" in wave_launcher
    assert 'cd "${CAPTION_RUN}" || exit 111' in wave_launcher
    assert 'pip install' not in wave_launcher

    decode_source = (ROOT / "vendor/image-captioning/harness.py").read_text()
    assert 'batch_size = 20 if config["strategy"] == "beam" else 64' in decode_source
