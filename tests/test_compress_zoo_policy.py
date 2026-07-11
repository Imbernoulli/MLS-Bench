from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor" / "compressai"
POLICY_TASKS = {
    "compress-content-dispatch": ("content", "content_policy", "rd12"),
    "compress-quality-dispatch": ("quality", "quality_policy", "rd12"),
    "compress-low-rate-policy": ("quality", "low_rate_policy", "lowq_rd12"),
    "compress-parameter-budget": ("global", "parameter_budget_policy", "rd12"),
    "compress-objective-policy": ("quality", "objective_policy", "rd6"),
    "compress-robust-policy": ("content", "robust_policy", "rd18"),
    "compress-bitrate-policy": ("quality", "bitrate_policy", "target_utility"),
    "compress-stream-budget": ("global", "stream_budget_policy", "rd12"),
    "compress-high-rate-policy": ("quality", "high_rate_policy", "highq_rd12"),
}


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _family_for(module, mode: str, choice, quality: int, group: str) -> str:
    if mode == "global":
        return choice
    if mode == "content":
        return choice[module.GROUP_ORDER.index(group)]
    return choice[module.GROUP_ORDER.index(module._quality_band(quality))]


def _proof(
    mode: str = "content",
    surface: str = "content_policy",
    choice=("factorized", "hyperprior_scale", "meanscale"),
) -> str:
    module = _load("policy_parser_fixture", VENDOR / "policy_parser.py")
    if mode == "global":
        choice = "hyperprior_scale"
    images = [f"kodim{index:02d}.png" for index in range(1, 25)]
    group_by_image = {
        image: group
        for group, names in module.GROUPS.items()
        for image in names
    }
    dispatch = [
        {
            "quality": quality,
            "image": image,
            "group": group_by_image[image],
            "family": _family_for(module, mode, choice, quality, group_by_image[image]),
        }
        for quality in module.QUALITIES
        for image in images
    ]
    choice_sha = module._canonical_sha(
        {"mode": mode, "surface": surface, "choice": choice}
    )
    dispatch_sha = module._canonical_sha(dispatch)
    lines = [
        "COMPRESS_PROTOCOL "
        f"protocol={module.PROTOCOL} asset_protocol={module.ASSET_PROTOCOL} "
        f"mode={mode} surface={surface} images=24 qualities=8 cases=192 "
        f"protocol_sha={module.ASSET_PROTOCOL_SHA} choice_sha={choice_sha} "
        f"dispatch_sha={dispatch_sha} target_sha={module._canonical_sha(module.TARGET_BPP)} "
        "network=blocked"
    ]
    family_rank = {family: rank + 1 for rank, family in enumerate(module.FAMILIES)}
    models = []
    model_by_key = {}
    for quality in module.QUALITIES:
        selected = {
            record["family"]
            for record in dispatch
            if record["quality"] == quality
        }
        for family in module.FAMILIES:
            if family not in selected:
                continue
            checkpoint_sha = hashlib.sha256(
                f"checkpoint-{family}-{quality}".encode()
            ).hexdigest()
            record = {
                "family": family,
                "quality": quality,
                "checkpoint_sha": checkpoint_sha,
                "cdf_sha": hashlib.sha256(f"cdf-{family}-{quality}".encode()).hexdigest(),
                "params": family_rank[family] * 1_000_000 + quality,
                "updated": True,
            }
            models.append(record)
            model_by_key[(family, quality)] = record
            lines.append(
                f"COMPRESS_MODEL family={family} quality={quality} "
                f"checkpoint_sha={record['checkpoint_sha']} cdf_sha={record['cdf_sha']} "
                f"params={record['params']} updated=1"
            )

    cases = []
    for quality in module.QUALITIES:
        for index, image in enumerate(images):
            group = group_by_image[image]
            family = _family_for(module, mode, choice, quality, group)
            model = model_by_key[(family, quality)]
            pixels = 512 * 768
            nbytes = 1000 + quality * 100 + index + family_rank[family] * 50
            bpp = float(f"{8.0 * nbytes / pixels:.12f}")
            psnr = float(f"{24.0 + quality + family_rank[family] / 10 + index / 1000:.9f}")
            record = {
                "family": family,
                "quality": quality,
                "image": image,
                "group": group,
                "checkpoint_sha": model["checkpoint_sha"],
                "params": model["params"],
                "streams": module.EXPECTED_STREAMS[family],
                "pixels": pixels,
                "bytes": nbytes,
                "bpp": bpp,
                "psnr": psnr,
                "recon_sha": hashlib.sha256(
                    f"recon-{family}-{quality}-{image}".encode()
                ).hexdigest(),
            }
            cases.append(record)
            lines.append(
                f"COMPRESS_CASE family={family} quality={quality} image={image} "
                f"group={group} checkpoint_sha={record['checkpoint_sha']} "
                f"params={record['params']} streams={record['streams']} pixels={pixels} "
                f"bytes={nbytes} bpp={bpp:.12f} psnr={psnr:.9f} "
                f"recon_sha={record['recon_sha']}"
            )

    rates = []
    settings = []
    for setting in module.SETTINGS:
        names = set(images) if setting == "full" else module.GROUPS[setting]
        per_quality = []
        for quality in module.QUALITIES:
            selected = [
                case
                for case in cases
                if case["quality"] == quality and case["image"] in names
            ]
            total_pixels = sum(case["pixels"] for case in selected)
            total_bytes = sum(case["bytes"] for case in selected)
            bpp = 8.0 * total_bytes / total_pixels
            psnr = sum(case["psnr"] for case in selected) / len(selected)
            record = {
                "setting": setting,
                "quality": quality,
                "count": len(selected),
                "pixels": total_pixels,
                "bytes": total_bytes,
                "bpp": float(f"{bpp:.12f}"),
                "psnr": float(f"{psnr:.9f}"),
                "rd6": float(f"{psnr - 6.0 * bpp:.9f}"),
                "rd12": float(f"{psnr - 12.0 * bpp:.9f}"),
                "rd18": float(f"{psnr - 18.0 * bpp:.9f}"),
                "target_utility": float(
                    f"{psnr - module.TARGET_PENALTY * abs(bpp - module.TARGET_BPP[quality]):.9f}"
                ),
                "mean_params": float(
                    f"{sum(case['params'] for case in selected) / len(selected):.3f}"
                ),
                "mean_streams": float(
                    f"{sum(case['streams'] for case in selected) / len(selected):.6f}"
                ),
            }
            rates.append(record)
            per_quality.append(record)
            lines.append(
                f"COMPRESS_RATE setting={setting} quality={quality} count={record['count']} "
                f"pixels={total_pixels} bytes={total_bytes} bpp={record['bpp']:.12f} "
                f"psnr={record['psnr']:.9f} rd6={record['rd6']:.9f} "
                f"rd12={record['rd12']:.9f} rd18={record['rd18']:.9f} "
                f"target_utility={record['target_utility']:.9f} "
                f"mean_params={record['mean_params']:.3f} "
                f"mean_streams={record['mean_streams']:.6f}"
            )

        def mean(key: str, records=per_quality) -> float:
            return sum(record[key] for record in records) / len(records)

        def band_mean(band: str) -> float:
            records = [
                record
                for record in per_quality
                if record["quality"] in module.QUALITY_BANDS[band]
            ]
            return sum(record["rd12"] for record in records) / len(records)

        setting_record = {
            "setting": setting,
            "qualities": 8,
            "images": len(names),
            "cases": len(names) * 8,
            "rd6": float(f"{mean('rd6'):.9f}"),
            "rd12": float(f"{mean('rd12'):.9f}"),
            "rd18": float(f"{mean('rd18'):.9f}"),
            "lowq_rd12": float(f"{band_mean('low'):.9f}"),
            "midq_rd12": float(f"{band_mean('mid'):.9f}"),
            "highq_rd12": float(f"{band_mean('high'):.9f}"),
            "target_utility": float(f"{mean('target_utility'):.9f}"),
            "mean_psnr": float(f"{mean('psnr'):.9f}"),
            "mean_bpp": float(f"{mean('bpp'):.12f}"),
            "mean_params": float(f"{mean('mean_params'):.3f}"),
            "mean_streams": float(f"{mean('mean_streams'):.6f}"),
        }
        settings.append(setting_record)
        lines.append(
            f"COMPRESS_SETTING setting={setting} qualities=8 images={len(names)} "
            f"cases={len(names) * 8} rd6={setting_record['rd6']:.9f} "
            f"rd12={setting_record['rd12']:.9f} rd18={setting_record['rd18']:.9f} "
            f"lowq_rd12={setting_record['lowq_rd12']:.9f} "
            f"midq_rd12={setting_record['midq_rd12']:.9f} "
            f"highq_rd12={setting_record['highq_rd12']:.9f} "
            f"target_utility={setting_record['target_utility']:.9f} "
            f"mean_psnr={setting_record['mean_psnr']:.9f} "
            f"mean_bpp={setting_record['mean_bpp']:.12f} "
            f"mean_params={setting_record['mean_params']:.3f} "
            f"mean_streams={setting_record['mean_streams']:.6f}"
        )
    lines.append(
        f"COMPRESS_FINAL protocol={module.PROTOCOL} mode={mode} surface={surface} "
        f"images=24 qualities=8 cases=192 models={len(models)} "
        f"streams={sum(case['streams'] for case in cases)} choice_sha={choice_sha} "
        f"dispatch_sha={dispatch_sha} cases_sha={module._canonical_sha(cases)} "
        f"models_sha={module._canonical_sha(models)} rates_sha={module._canonical_sha(rates)} "
        f"settings_sha={module._canonical_sha(settings)} elapsed=123.456"
    )
    return "\n".join(lines)


def test_policy_task_inventory_is_real_scale_and_distinct():
    surfaces = set()
    for task_id, (mode, surface, objective) in POLICY_TASKS.items():
        task = ROOT / "tasks" / task_id
        config = json.loads((task / "config.json").read_text())
        assert config["calibration_protocol"] == "compressai_zoo_kodak24_q1q8_policy_v1"
        assert config["seeds"] == [42]
        assert len(config["test_cmds"]) == 1
        command = config["test_cmds"][0]
        assert command["cmd"] == "scripts/full.sh"
        assert command["score_settings"] == ["full", "low", "mid", "high"]
        assert "--steps" not in (task / "scripts" / "full.sh").read_text()
        assert "harness_zoo_policy.py" in (task / "scripts" / "full.sh").read_text()
        assert "24 Kodak" in (task / "task_description.md").read_text()
        assert "qualities 1 through 8" in (task / "task_description.md").read_text()
        assert not any(
            word in (task / "task_description.md").read_text().lower().split()
            for word in ("hidden", "public")
        )
        parser_module = _load(f"parser_{task_id}", task / "parser.py")
        parser = parser_module.Parser()
        assert parser.expected_mode == mode
        assert parser.expected_surface == surface
        assert objective in (task / "score_spec.py").read_text()
        surfaces.add((mode, surface, objective))
    assert len(surfaces) == 9


def test_policy_surface_loader_accepts_only_literal_dispatch(tmp_path: Path):
    harness = _load("policy_harness_surface", VENDOR / "harness_zoo_policy.py")
    valid = tmp_path / "valid.py"
    valid.write_text(
        '"""surface"""\nfrom __future__ import annotations\n\n\n'
        'def content_policy():\n    return ("factorized", "hyperprior_scale", "meanscale")\n'
    )
    assert harness._load_surface(valid, "content_policy", "content") == (
        "factorized",
        "hyperprior_scale",
        "meanscale",
    )
    for source in (
        "def content_policy():\n    raise RuntimeError('executed')\n",
        "def content_policy():\n    return choose()\n",
        "def content_policy(x):\n    return ('factorized',) * 3\n",
        "def content_policy():\n    return ('factorized', 'unknown', 'meanscale')\n",
        "def wrong_name():\n    return ('factorized', 'factorized', 'factorized')\n",
    ):
        path = tmp_path / hashlib.sha256(source.encode()).hexdigest()
        path.write_text(source)
        try:
            harness._load_surface(path, "content_policy", "content")
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError("invalid policy surface was accepted")


def test_complete_policy_proofs_parse_all_four_settings():
    module = _load("policy_parser_valid", VENDOR / "policy_parser.py")
    for mode, surface in (
        ("global", "parameter_budget_policy"),
        ("content", "content_policy"),
        ("quality", "quality_policy"),
    ):
        parser = module.make_parser(mode, surface)()
        result = parser.parse("kodak24_q1q8", _proof(mode, surface))
        assert len(result.metrics) == len(module.SETTING_METRICS) * 4, result.feedback
        assert all(math.isfinite(value) for value in result.metrics.values())


def test_policy_parser_rejects_incomplete_tampered_and_trailing_proofs():
    module = _load("policy_parser_invalid", VENDOR / "policy_parser.py")
    parser = module.make_parser("content", "content_policy")()
    valid = _proof()
    lines = valid.splitlines()
    case = next(line for line in lines if line.startswith("COMPRESS_CASE"))
    final = lines[-1]
    invalid = (
        "",
        "\n".join(line for line in lines if line != case),
        valid.replace(case, case + "\n" + case),
        valid.replace("streams=1", "streams=2", 1),
        valid.replace("params=1000001", "params=1000002", 1),
        valid.replace("bpp=", "bpp=nan", 1),
        valid.replace(module.ASSET_PROTOCOL_SHA, "0" * 64, 1),
        "\n".join(lines[:-1]),
        valid + "\n" + final,
        valid + "\ntrailing output",
        "Traceback (most recent call last)\n" + valid,
    )
    for raw in invalid:
        assert parser.parse("kodak24_q1q8", raw).metrics == {}
