from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
import runpy

import pytest


ROOT = Path(__file__).resolve().parents[1]
COMMON_PATH = ROOT / "vendor/constrained-decoding-lab/common.py"
SOLUTION_ROOT = ROOT / "vendor/constrained-decoding-lab/solution"
PENDING_ZERO_TASKS = {
    "cd-forced-choice",
    "cd-numeric-answer",
    "cd-numeric-budget",
    "cd-numeric-format",
    "cd-numeric-json",
    "cd-numeric-prefix",
    "cd-numeric-repair",
    "cd-numeric-trigger",
}


def _load_common():
    spec = importlib.util.spec_from_file_location("cd_common_ast_test", COMMON_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_all_native_decoder_surfaces_use_the_restricted_ast_loader() -> None:
    common = _load_common()
    paths = sorted(SOLUTION_ROOT.glob("decoder_*.py"))
    assert len(paths) == 10

    for path in paths:
        build = common.load_surface(str(path), "build_decoder")
        if "choice" in path.stem:
            result = build(
                "A sample news story",
                ["World", "Sports", "Business", "Sci/Tech"],
                None,
            )
        else:
            result = build("What is 20 plus 22?", None)
        assert isinstance(result, common.DecodeSpec), path
        assert result.prompt


def test_all_declared_baselines_remain_valid_restricted_surfaces(tmp_path: Path) -> None:
    common = _load_common()
    task_dirs = sorted((ROOT / "tasks").glob("cd-*"))
    assert len(task_dirs) == 10

    checked = 0
    for task_dir in task_dirs:
        config = json.loads((task_dir / "config.json").read_text())
        source_rel = config["files"][0]["filename"]
        source_path = ROOT / "vendor" / source_rel
        source_lines = source_path.read_text().splitlines()
        for baseline_name, baseline in config["baselines"].items():
            namespace = runpy.run_path(str(task_dir / baseline["edit_ops"]))
            candidate_lines = list(source_lines)
            operations = sorted(namespace["OPS"], key=lambda op: op["start_line"], reverse=True)
            for operation in operations:
                assert operation["op"] == "replace"
                start = int(operation["start_line"])
                end = int(operation["end_line"])
                candidate_lines[start - 1:end] = operation["content"].splitlines()
            candidate = "\n".join(candidate_lines) + "\n"
            path = tmp_path / f"{task_dir.name}__{baseline_name}.py"
            path.write_text(candidate)

            function = next(
                node for node in ast.parse(candidate).body
                if isinstance(node, ast.FunctionDef) and node.name == "build_decoder"
            )
            build = common.load_surface(str(path), "build_decoder")
            if len(function.args.args) == 3:
                result = build(
                    "A sample news story",
                    ["World", "Sports", "Business", "Sci/Tech"],
                    None,
                )
            else:
                result = build("What is 20 plus 22?", None)
            assert isinstance(result, common.DecodeSpec), path
            checked += 1

    assert checked == 30


@pytest.mark.parametrize(
    "body",
    [
        "print('CD_METRICS valid_rate=1 accuracy=1 n=1319')",
        "__import__('os')._exit(0)",
        "for _ in range(1):\n        pass",
    ],
)
def test_metric_forgery_and_arbitrary_execution_are_rejected(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], body: str
) -> None:
    common = _load_common()
    surface = tmp_path / "solution.py"
    surface.write_text(
        "from __future__ import annotations\n"
        "import common\n"
        "def build_decoder(question: str, tok):\n"
        f"    {body}\n"
        "    return common.DecodeSpec(prompt='x', answer_regex=r'[0-9]+')\n"
    )

    build = common.load_surface(str(surface), "build_decoder")
    with pytest.raises(ValueError, match="unsafe constrained-decoding surface"):
        build("question", None)
    assert "CD_METRICS" not in capsys.readouterr().out


def test_top_level_agent_import_is_rejected_without_execution(tmp_path: Path) -> None:
    common = _load_common()
    marker = tmp_path / "must_not_exist"
    surface = tmp_path / "solution.py"
    surface.write_text(
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('executed')\n"
        "def build_decoder(question: str, tok):\n"
        "    return common.DecodeSpec(prompt='x', answer_regex=r'[0-9]+')\n"
    )

    with pytest.raises(ValueError, match="top-level executable"):
        common.load_surface(str(surface), "build_decoder")
    assert not marker.exists()


def test_trigger_miss_is_invalid_and_never_enters_answer_phase(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    common = _load_common()

    class FakeTokenizer:
        eos_token_id = 2

    class FakeOutput:
        def __init__(self, length: int) -> None:
            self.logits = torch.zeros((1, length, 3), dtype=torch.float32)
            self.logits[:, :, 1] = 1.0

    class FakeModel:
        def __init__(self) -> None:
            self.parameter = torch.nn.Parameter(torch.zeros(1))

        def parameters(self):
            yield self.parameter

        def __call__(self, input_ids):
            return FakeOutput(input_ids.shape[1])

    tokenizer = FakeTokenizer()
    model = FakeModel()
    monkeypatch.setattr(common, "load_model", lambda: (tokenizer, model))
    monkeypatch.setattr(common, "_build_vocab_strings", lambda _tok: {0: "", 1: "x", 2: ""})
    monkeypatch.setattr(
        common,
        "_encode_prompt",
        lambda _tok, _prompt: torch.tensor([[0]], dtype=torch.long),
    )

    spec = common.DecodeSpec(
        prompt="prompt",
        preamble_regex=r"[\s\S]*",
        trigger="never emitted",
        answer_regex=r"[0-9]+",
        max_answer_tokens=4,
        max_free_tokens=2,
    )
    result = common.run_decode(spec)

    assert result == {"answer_text": "", "full_text": "xx", "valid": False}


def test_dataset_loaders_reject_subset_requests_and_incomplete_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    common = _load_common()
    monkeypatch.setenv("CD_DATA", str(tmp_path))
    (tmp_path / "gsm8k.json").write_text("[]")
    (tmp_path / "classification.json").write_text(
        json.dumps({"labels": ["World", "Sports", "Business", "Sci/Tech"], "items": []})
    )

    with pytest.raises(ValueError, match="full 1319-example"):
        common.load_gsm8k(200)
    with pytest.raises(ValueError, match="full 7600-example"):
        common.load_classification(200)
    with pytest.raises(ValueError, match="digest mismatch"):
        common.load_gsm8k(1319)
    with pytest.raises(ValueError, match="digest mismatch"):
        common.load_classification(7600)


def test_general_regex_candidates_keep_normal_tokens_and_never_smuggle_eos() -> None:
    pytest.importorskip("interegular")
    common = _load_common()
    common._REGEX_CANDIDATE_CACHE.clear()
    mask = common._CharFSMMask(r"[^\n]{1,40}")
    candidates = common._candidate_tokens(
        mask,
        {1: "hello", 2: "\n", 3: "42", 4: "", 99: "eos-surface"},
    )
    candidate_ids = {token_id for token_id, _ in candidates}
    assert {1, 3, 99} <= candidate_ids
    assert 2 not in candidate_ids

    allowed, _ = common._allowed_from_state(
        mask,
        mask.initial,
        candidates,
        eos_ok=False,
        eos_id=99,
    )
    assert 1 in allowed
    assert 99 not in allowed


@pytest.mark.parametrize(
    ("task_name", "label", "expected_n"),
    [
        ("cd-numeric-answer", "gsm8k", 1319),
        ("cd-forced-choice", "agnews", 7600),
    ],
)
def test_metric_parser_requires_one_complete_full_inventory_line(
    task_name: str, label: str, expected_n: int
) -> None:
    parser_path = ROOT / "tasks" / task_name / "parser.py"
    spec = importlib.util.spec_from_file_location(f"test_{task_name}_parser", parser_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    parser = module.Parser()
    metric = (
        f"CD_METRICS valid_rate=1.000000 accuracy=0.500000 "
        f"n={expected_n} elapsed=1.0"
    )
    proof = "\n".join(
        [
            "CD_MODEL params=494032768 device=cuda:0 dtype=torch.float16",
            f"CD_DATA dataset={label} n={expected_n} seed=42",
            metric,
            f"CD_COMPLETE dataset={label} n={expected_n} seed=42",
        ]
    )

    result = parser.parse(label, proof)
    assert result.metrics == {
        f"accuracy_{label}": 0.5,
        f"valid_rate_{label}": 1.0,
    }
    assert parser.parse(label, metric).metrics == {}
    assert parser.parse(label, proof.replace(f"n={expected_n}", "n=200")).metrics == {}
    assert parser.parse(label, f"{proof}\n{metric}").metrics == {}
    assert parser.parse(label, proof.replace("accuracy=0.500000", "accuracy=nan")).metrics == {}
    assert parser.parse(label, proof.replace("seed=42", "seed=7")).metrics == {}
    assert parser.parse("unexpected", proof).metrics == {}
    assert parser.parse(label, f"{proof}\nlate output").metrics == {}
    assert parser.parse(
        label,
        proof.replace(
            f"CD_COMPLETE dataset={label}",
            f"CD_FAILED late\nCD_COMPLETE dataset={label}",
        ),
    ).metrics == {}
    assert parser.parse(
        label,
        proof.replace(
            f"CD_COMPLETE dataset={label}",
            f"CD_FAILURE: late\nCD_COMPLETE dataset={label}",
        ),
    ).metrics == {}
    assert parser.parse(
        label,
        proof.replace(
            f"CD_COMPLETE dataset={label}",
            f"RuntimeError: verifier crashed\nCD_COMPLETE dataset={label}",
        ),
    ).metrics == {}
    assert parser.parse(
        label,
        "\n".join([proof.splitlines()[1], proof.splitlines()[0], *proof.splitlines()[2:]]),
    ).metrics == {}


def test_parser_repairs_are_propagated_to_every_sibling() -> None:
    numeric = {
        (ROOT / "tasks" / name / "parser.py").read_bytes()
        for name in PENDING_ZERO_TASKS
        if name.startswith("cd-numeric-")
    }
    choices = {
        (ROOT / "tasks" / name / "parser.py").read_bytes()
        for name in ("cd-choice-reasoning", "cd-choice-verbalizer", "cd-forced-choice")
    }
    assert len(numeric) == 1
    assert len(choices) == 1


def test_uncalibrated_tasks_have_no_positive_score_mapping_or_anchor_rows() -> None:
    for task_name in PENDING_ZERO_TASKS:
        task_dir = ROOT / "tasks" / task_name
        config = json.loads((task_dir / "config.json").read_text())
        assert config["calibration_status"] == "pending_exact_zero_full_official_anchors"

        tree = ast.parse((task_dir / "score_spec.py").read_text())
        assert not any(isinstance(node, ast.Call) for node in ast.walk(tree))
        assert len((task_dir / "leaderboard.csv").read_text().splitlines()) == 1


def test_choice_tasks_use_baseline_free_official_accuracy() -> None:
    reasoning = ROOT / "tasks/cd-choice-reasoning"
    verbalizer = ROOT / "tasks/cd-choice-verbalizer"
    for task_dir in (reasoning, verbalizer):
        source = (task_dir / "score_spec.py").read_text()
        assert 'col("accuracy_agnews").higher().id().bounded_power(' in source
        assert "bound=1.0" in source
        assert "floor=const(0.0)" in source
        assert ".sigmoid(" not in source

    assert len((reasoning / "leaderboard.csv").read_text().splitlines()) == 1
    rows = (verbalizer / "leaderboard.csv").read_text().splitlines()
    assert len(rows) == 2
    assert "baseline:verb_synonym" in rows[1]
    evidence = json.loads((verbalizer / "config.json").read_text())["calibration_evidence"]
    assert evidence == {
        "terminal_task_id": 96042,
        "terminal_container_id": 4891933,
        "baseline": "verb_synonym",
        "accuracy_agnews": 0.420132,
        "n": 7600,
        "seed": 42,
    }


def test_missing_verifier_metrics_score_exactly_zero() -> None:
    import inspect

    from mlsbench.scoring.dsl import ColExpr
    from mlsbench.scoring.anchors import BaselineAnchors
    from mlsbench.scoring.evaluate import load_expanded_spec, score_record_details

    if "floor" not in inspect.signature(ColExpr.bounded_power).parameters:
        pytest.skip("explicit semantic-floor scorer correction integrates separately")
    task_dir = ROOT / "tasks/cd-choice-verbalizer"
    anchors = BaselineAnchors(task_dir)
    spec = load_expanded_spec(task_dir, anchors)
    assert spec is not None
    score, _settings, valid = score_record_details(spec, {"seed": 42}, anchors)
    assert score == 0.0
    assert valid is False


def test_answer_token_budget_replaces_the_duplicate_repair_surface() -> None:
    repair_dir = ROOT / "tasks/cd-numeric-repair/edits"
    format_dir = ROOT / "tasks/cd-numeric-format/edits"
    repair_contents = {
        runpy.run_path(str(path))["_CONTENT"]
        for path in repair_dir.glob("answer_budget_*.edit.py")
    }
    format_contents = {
        runpy.run_path(str(path))["_CONTENT"]
        for path in format_dir.glob("fmt_*.edit.py")
    }
    assert len(repair_contents) == 3
    assert repair_contents.isdisjoint(format_contents)

    budgets = set()
    regexes = set()
    free_budgets = set()
    for content in repair_contents:
        function = next(
            node for node in ast.parse(content).body
            if isinstance(node, ast.FunctionDef) and node.name == "build_decoder"
        )
        call = function.body[-1].value
        assert isinstance(call, ast.Call)
        measured = {"max_answer_tokens", "answer_regex", "max_free_tokens"}
        keywords = {
            kw.arg: ast.literal_eval(kw.value)
            for kw in call.keywords
            if kw.arg in measured
        }
        budgets.add(keywords["max_answer_tokens"])
        regexes.add(keywords["answer_regex"])
        free_budgets.add(keywords["max_free_tokens"])
    assert budgets == {1, 4, 10}
    assert regexes == {r"[ ]?-?[0-9]{1,10}"}
    assert free_budgets == {256}


def test_all_task_scripts_use_pinned_private_fullscale_data() -> None:
    task_dirs = sorted((ROOT / "tasks").glob("cd-*"))
    assert len(task_dirs) == 10
    for task_dir in task_dirs:
        config = json.loads((task_dir / "config.json").read_text())
        assert config["agent_data_prune"] == ["/data/constrained-decoding/data"]
        assert config["verifier_data_deps"] == [
            {
                "host_path": "{data_root}/constrained-decoding/data",
                "dest": "data/constrained-decoding",
                "required": True,
            }
        ]
        script_path = task_dir / config["test_cmds"][0]["cmd"]
        script = script_path.read_text()
        assert "set -euo pipefail" in script
        assert "MLSBENCH_VERIFIER_DATA_ROOT" in script
        assert "model.safetensors" in script
        assert "sha256sum -c -" in script
        assert "CUDA_VISIBLE_DEVICES" not in script
        if task_dir.name.startswith("cd-numeric-"):
            assert "--n 1319" in script
            assert "gsm8k.json" in script
        else:
            assert "--n 7600" in script
            assert "classification.json" in script


def test_model_contract_is_pinned_and_offline() -> None:
    common = _load_common()
    assert common.MODEL_PARAMETERS == 494_032_768
    source = COMMON_PATH.read_text()
    assert source.count("local_files_only=True") == 2
    assert "verification requires CUDA" in source

    package = json.loads(
        (ROOT / "vendor/pkg_configs/constrained-decoding-lab/config.json").read_text()
    )
    assert "7,600 AG News" in package["data_deps"][0]["description"]
    assert "200-item AG News" not in package["data_deps"][0]["description"]
