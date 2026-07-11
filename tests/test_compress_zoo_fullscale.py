from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "tasks" / "compress-entropy-model"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _sha(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _proof(family: str = "factorized") -> str:
    parser_module = _load("compress_parser_fixture", TASK / "parser.py")
    lines = [
        "COMPRESS_PROTOCOL "
        f"protocol={parser_module.PROTOCOL} family={family} images=24 qualities=8 "
        f"cases=192 protocol_sha={'a' * 64} network=blocked"
    ]
    models = []
    for quality in parser_module.QUALITIES:
        model = {
            "quality": quality,
            "checkpoint_sha": hashlib.sha256(f"checkpoint-{quality}".encode()).hexdigest(),
            "cdf_sha": hashlib.sha256(f"cdf-{quality}".encode()).hexdigest(),
            "params": 1_000_000 + quality,
            "updated": True,
        }
        models.append(model)
        lines.append(
            f"COMPRESS_MODEL quality={quality} checkpoint_sha={model['checkpoint_sha']} "
            f"cdf_sha={model['cdf_sha']} params={model['params']} updated=1"
        )

    image_order = [f"kodim{i:02d}.png" for i in range(1, 25)]
    group_by_image = {
        image: group
        for group, images in parser_module.GROUPS.items()
        for image in images
    }
    cases = []
    for quality in parser_module.QUALITIES:
        for index, image in enumerate(image_order):
            pixels = 512 * 768
            nbytes = 1000 + quality * 100 + index
            bpp = float(f"{8.0 * nbytes / pixels:.12f}")
            psnr = float(f"{25.0 + quality + index / 1000:.9f}")
            recon_sha = hashlib.sha256(f"{quality}-{image}".encode()).hexdigest()
            case = {
                "quality": quality,
                "image": image,
                "group": group_by_image[image],
                "pixels": pixels,
                "bytes": nbytes,
                "bpp": bpp,
                "psnr": psnr,
                "recon_sha": recon_sha,
            }
            cases.append(case)
            lines.append(
                f"COMPRESS_CASE quality={quality} image={image} group={case['group']} "
                f"pixels={pixels} bytes={nbytes} bpp={bpp:.12f} psnr={psnr:.9f} "
                f"recon_sha={recon_sha}"
            )

    rates = {}
    settings = []
    for setting in parser_module.SETTINGS:
        names = (
            parser_module.ALL_IMAGES
            if setting == "full"
            else parser_module.GROUPS[setting]
        )
        for quality in parser_module.QUALITIES:
            selected = [
                case
                for case in cases
                if case["quality"] == quality and case["image"] in names
            ]
            total_pixels = sum(case["pixels"] for case in selected)
            total_bytes = sum(case["bytes"] for case in selected)
            bpp = float(f"{8.0 * total_bytes / total_pixels:.12f}")
            psnr = float(f"{sum(case['psnr'] for case in selected) / len(selected):.9f}")
            utility = float(f"{psnr - parser_module.RD_BETA * bpp:.9f}")
            rates[(setting, quality)] = {
                "count": len(selected),
                "pixels": total_pixels,
                "bytes": total_bytes,
                "bpp": bpp,
                "psnr": psnr,
                "utility": utility,
            }
            lines.append(
                f"COMPRESS_RATE setting={setting} quality={quality} count={len(selected)} "
                f"pixels={total_pixels} bytes={total_bytes} bpp={bpp:.12f} "
                f"psnr={psnr:.9f} utility={utility:.9f}"
            )
        selected_rates = [rates[(setting, quality)] for quality in parser_module.QUALITIES]
        record = {
            "setting": setting,
            "qualities": 8,
            "images": len(names),
            "cases": len(names) * 8,
            "mean_rd_utility": float(
                f"{sum(rate['utility'] for rate in selected_rates) / 8:.9f}"
            ),
            "mean_psnr": float(
                f"{sum(rate['psnr'] for rate in selected_rates) / 8:.9f}"
            ),
            "mean_bpp": float(
                f"{sum(rate['bpp'] for rate in selected_rates) / 8:.12f}"
            ),
        }
        settings.append(record)
        lines.append(
            f"COMPRESS_SETTING setting={setting} qualities=8 images={len(names)} "
            f"cases={len(names) * 8} mean_rd_utility={record['mean_rd_utility']:.9f} "
            f"mean_psnr={record['mean_psnr']:.9f} mean_bpp={record['mean_bpp']:.12f}"
        )

    lines.append(
        "COMPRESS_FINAL "
        f"protocol={parser_module.PROTOCOL} family={family} images=24 qualities=8 "
        f"cases=192 streams=192 cases_sha={_sha(cases)} checkpoints_sha={_sha(models)} "
        f"settings_sha={_sha(settings)} elapsed=123.456"
    )
    return "\n".join(lines)


def test_full_proof_parses_all_required_settings():
    module = _load("compress_parser_valid", TASK / "parser.py")
    result = module.Parser().parse("kodak24_q1q8", _proof())
    assert set(result.metrics) == {
        f"{metric}_{setting}"
        for setting in module.SETTINGS
        for metric in ("mean_rd_utility", "psnr", "bpp")
    }
    assert all(value == value for value in result.metrics.values())


def test_incomplete_nonfinite_duplicate_and_trailing_proofs_are_empty():
    module = _load("compress_parser_fail_closed", TASK / "parser.py")
    parser = module.Parser()
    valid = _proof()
    lines = valid.splitlines()
    case = next(line for line in lines if line.startswith("COMPRESS_CASE"))
    rate = next(line for line in lines if line.startswith("COMPRESS_RATE"))
    setting = next(line for line in lines if line.startswith("COMPRESS_SETTING"))
    model = next(line for line in lines if line.startswith("COMPRESS_MODEL"))
    final = lines[-1]
    invalid = (
        "",
        "\n".join(line for line in lines if line != case),
        valid.replace(case, case + "\n" + case),
        valid.replace("updated=1", "updated=0", 1),
        valid.replace("bpp=0.022379557292", "bpp=nan", 1),
        valid.replace("bytes=1100", "bytes=1101", 1),
        "\n".join(line for line in lines if line != rate),
        "\n".join(line for line in lines if line != setting),
        "\n".join(line for line in lines if line != model),
        "\n".join([lines[0], lines[2], lines[1], *lines[3:]]),
        "\n".join(lines[:-1]),
        valid + "\n" + final,
        valid + "\ntrailing output",
        valid.replace(final, "unrecognized output\n" + final),
        "Traceback (most recent call last)\n" + valid,
    )
    for raw in invalid:
        assert parser.parse("kodak24_q1q8", raw).metrics == {}


def test_measured_calibration_scores_valid_families_and_missing_metrics_zero():
    from mlsbench.scoring.anchors import BaselineAnchors
    from mlsbench.scoring.evaluate import score_record_details
    from mlsbench.scoring.spec import load_score_spec

    spec = load_score_spec(TASK)
    assert spec is not None
    anchors = BaselineAnchors(TASK)
    expected = {
        "factorized": 0.2,
        "hyperprior_scale": 0.5,
        "meanscale": 0.5942195237451419,
    }
    with (TASK / "leaderboard.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["model"] for row in rows} == {
        f"baseline:{family}" for family in expected
    }

    records = {}
    for row in rows:
        family = row["model"].removeprefix("baseline:")
        record = {
            key: float(value)
            for key, value in row.items()
            if key.startswith(("mean_rd_utility_", "psnr_", "bpp_"))
        }
        records[family] = record
        score, settings, valid = score_record_details(spec, record, anchors)
        assert valid is True
        assert len(settings) == 4
        assert math.isclose(score, expected[family], rel_tol=0, abs_tol=1e-12)

    missing = dict(records["meanscale"])
    del missing["mean_rd_utility_high"]
    score, _settings, valid = score_record_details(spec, missing, anchors)
    assert valid is False
    assert score == 0.0

    score, _settings, valid = score_record_details(spec, {}, anchors)
    assert valid is False
    assert score == 0.0


def test_surface_is_literal_and_never_executed(tmp_path: Path):
    harness = _load("compress_harness_surface", ROOT / "vendor/compressai/harness_zoo_entropy.py")
    good = tmp_path / "good.py"
    good.write_text(
        '"""surface"""\nfrom __future__ import annotations\n\n\ndef entropy_model() -> str:\n    return "meanscale"\n'
    )
    assert harness._load_surface(good) == "meanscale"

    for source in (
        "def entropy_model():\n    raise RuntimeError('executed')\n",
        "def entropy_model():\n    return choose_model()\n",
        "def entropy_model(x):\n    return 'factorized'\n",
        "def entropy_model():\n    return 'factorized'\n    return 'meanscale'\n",
        "def entropy_model():\n    return 'unknown'\n",
    ):
        bad = tmp_path / f"bad-{hashlib.sha256(source.encode()).hexdigest()}.py"
        bad.write_text(source)
        try:
            harness._load_surface(bad)
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError("non-literal surface was accepted")
