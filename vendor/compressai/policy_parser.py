"""Shared strict parser for pinned CompressAI policy evaluations."""

from __future__ import annotations

import hashlib
import json
import math
import re

from mlsbench.agent.parsers import OutputParser, ParseResult


PROTOCOL = "compressai_zoo_kodak24_q1q8_policy_v1"
ASSET_PROTOCOL = "compressai_zoo_kodak24_q1q8_bitstream_v1"
ASSET_PROTOCOL_SHA = "4b84d6ac0f8af07206b674824608ddbf1ff6e05037f363048521c5869bc525c9"
QUALITIES = tuple(range(1, 9))
SETTINGS = ("full", "low", "mid", "high")
GROUP_ORDER = ("low", "mid", "high")
FAMILIES = ("factorized", "hyperprior_scale", "meanscale")
POLICY_MODES = ("global", "content", "quality")
QUALITY_BANDS = {
    "low": (1, 2, 3),
    "mid": (4, 5, 6),
    "high": (7, 8),
}
TARGET_BPP = {
    1: 0.15,
    2: 0.25,
    3: 0.35,
    4: 0.50,
    5: 0.70,
    6: 0.90,
    7: 1.20,
    8: 1.50,
}
TARGET_PENALTY = 12.0
EXPECTED_STREAMS = {
    "factorized": 1,
    "hyperprior_scale": 2,
    "meanscale": 2,
}
GROUPS = {
    "low": {
        "kodim02.png", "kodim03.png", "kodim09.png", "kodim10.png",
        "kodim12.png", "kodim15.png", "kodim20.png", "kodim23.png",
    },
    "mid": {
        "kodim04.png", "kodim07.png", "kodim11.png", "kodim16.png",
        "kodim17.png", "kodim19.png", "kodim21.png", "kodim22.png",
    },
    "high": {
        "kodim01.png", "kodim05.png", "kodim06.png", "kodim08.png",
        "kodim13.png", "kodim14.png", "kodim18.png", "kodim24.png",
    },
}
ALL_IMAGES = set().union(*GROUPS.values())
HEX64 = r"[0-9a-f]{64}"
NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
SETTING_METRICS = (
    "rd6",
    "rd12",
    "rd18",
    "lowq_rd12",
    "midq_rd12",
    "highq_rd12",
    "target_utility",
    "mean_psnr",
    "mean_bpp",
    "mean_params",
    "mean_streams",
)


def _canonical_sha(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _close(left: float, right: float, tolerance: float = 5e-8) -> bool:
    return math.isclose(left, right, rel_tol=tolerance, abs_tol=tolerance)


def _mean(records: list[dict], key: str) -> float:
    if not records:
        raise ValueError(f"empty aggregate for {key}")
    return sum(float(record[key]) for record in records) / len(records)


def _quality_band(quality: int) -> str:
    for band, qualities in QUALITY_BANDS.items():
        if quality in qualities:
            return band
    raise ValueError("quality outside fixed inventory")


def _derive_choice(mode: str, cases: list[dict]):
    if mode == "global":
        values = {case["family"] for case in cases}
        if len(values) != 1:
            raise ValueError("global policy dispatched more than one family")
        return next(iter(values))
    if mode == "content":
        choice = []
        for group in GROUP_ORDER:
            values = {case["family"] for case in cases if case["group"] == group}
            if len(values) != 1:
                raise ValueError(f"content policy is not constant for {group}")
            choice.append(next(iter(values)))
        return tuple(choice)
    if mode == "quality":
        choice = []
        for band in GROUP_ORDER:
            values = {
                case["family"]
                for case in cases
                if case["quality"] in QUALITY_BANDS[band]
            }
            if len(values) != 1:
                raise ValueError(f"quality policy is not constant for {band}")
            choice.append(next(iter(values)))
        return tuple(choice)
    raise ValueError("unknown policy mode")


class PolicyParser(OutputParser):
    expected_mode = ""
    expected_surface = ""

    def parse(self, cmd_label: str, raw_output: str) -> ParseResult:
        del cmd_label
        try:
            metrics, feedback = self._parse_strict(raw_output)
            return ParseResult(feedback=feedback, metrics=metrics)
        except Exception as exc:
            tail = raw_output[-3000:]
            return ParseResult(
                feedback=f"full policy protocol rejected: {exc}\n{tail}",
                metrics={},
            )

    def _parse_strict(self, raw_output: str) -> tuple[dict, str]:
        if self.expected_mode not in POLICY_MODES:
            raise ValueError("parser has no valid expected mode")
        if not self.expected_surface:
            raise ValueError("parser has no expected surface")
        lines = [line.strip() for line in raw_output.splitlines() if line.strip()]
        if not lines:
            raise ValueError("empty verifier output")
        if any(
            marker in raw_output
            for marker in (
                "Traceback (most recent call last)",
                "SURFACE_ERROR",
                "COMPRESS_FAILED",
                "COMPRESS_FALLBACK",
            )
        ):
            raise ValueError("failure marker present")

        protocol_re = re.compile(
            rf"^COMPRESS_PROTOCOL protocol={PROTOCOL} asset_protocol={ASSET_PROTOCOL} "
            rf"mode=(global|content|quality) surface=([A-Za-z_][A-Za-z0-9_]*) "
            rf"images=(\d+) qualities=(\d+) cases=(\d+) protocol_sha=({HEX64}) "
            rf"choice_sha=({HEX64}) dispatch_sha=({HEX64}) target_sha=({HEX64}) "
            rf"network=blocked$"
        )
        model_re = re.compile(
            rf"^COMPRESS_MODEL family=(factorized|hyperprior_scale|meanscale) "
            rf"quality=(\d+) checkpoint_sha=({HEX64}) cdf_sha=({HEX64}) "
            rf"params=(\d+) updated=([01])$"
        )
        case_re = re.compile(
            rf"^COMPRESS_CASE family=(factorized|hyperprior_scale|meanscale) "
            rf"quality=(\d+) image=(kodim\d{{2}}\.png) group=(low|mid|high) "
            rf"checkpoint_sha=({HEX64}) params=(\d+) streams=(\d+) "
            rf"pixels=(\d+) bytes=(\d+) bpp=({NUMBER}) psnr=({NUMBER}) "
            rf"recon_sha=({HEX64})$"
        )
        rate_re = re.compile(
            rf"^COMPRESS_RATE setting=(full|low|mid|high) quality=(\d+) "
            rf"count=(\d+) pixels=(\d+) bytes=(\d+) bpp=({NUMBER}) "
            rf"psnr=({NUMBER}) rd6=({NUMBER}) rd12=({NUMBER}) rd18=({NUMBER}) "
            rf"target_utility=({NUMBER}) mean_params=({NUMBER}) "
            rf"mean_streams=({NUMBER})$"
        )
        setting_re = re.compile(
            rf"^COMPRESS_SETTING setting=(full|low|mid|high) qualities=(\d+) "
            rf"images=(\d+) cases=(\d+) rd6=({NUMBER}) rd12=({NUMBER}) "
            rf"rd18=({NUMBER}) lowq_rd12=({NUMBER}) midq_rd12=({NUMBER}) "
            rf"highq_rd12=({NUMBER}) target_utility=({NUMBER}) "
            rf"mean_psnr=({NUMBER}) mean_bpp=({NUMBER}) "
            rf"mean_params=({NUMBER}) mean_streams=({NUMBER})$"
        )
        final_re = re.compile(
            rf"^COMPRESS_FINAL protocol={PROTOCOL} mode=(global|content|quality) "
            rf"surface=([A-Za-z_][A-Za-z0-9_]*) images=(\d+) qualities=(\d+) "
            rf"cases=(\d+) models=(\d+) streams=(\d+) choice_sha=({HEX64}) "
            rf"dispatch_sha=({HEX64}) cases_sha=({HEX64}) models_sha=({HEX64}) "
            rf"rates_sha=({HEX64}) settings_sha=({HEX64}) elapsed=({NUMBER})$"
        )

        protocol_matches = [
            protocol_re.fullmatch(line)
            for line in lines
            if line.startswith("COMPRESS_PROTOCOL")
        ]
        if len(protocol_matches) != 1 or protocol_matches[0] is None:
            raise ValueError("missing or duplicate policy protocol proof")
        protocol_match = protocol_matches[0]
        mode, surface = protocol_match.group(1, 2)
        if (mode, surface) != (self.expected_mode, self.expected_surface):
            raise ValueError("policy mode or surface mismatch")
        if tuple(map(int, protocol_match.group(3, 4, 5))) != (24, 8, 192):
            raise ValueError("protocol inventory mismatch")
        if protocol_match.group(6) != ASSET_PROTOCOL_SHA:
            raise ValueError("asset protocol digest mismatch")
        protocol_choice_sha = protocol_match.group(7)
        protocol_dispatch_sha = protocol_match.group(8)
        if protocol_match.group(9) != _canonical_sha(TARGET_BPP):
            raise ValueError("target schedule digest mismatch")

        models = []
        seen_models = set()
        for line in lines:
            if not line.startswith("COMPRESS_MODEL"):
                continue
            match = model_re.fullmatch(line)
            if match is None:
                raise ValueError("malformed model proof")
            record = {
                "family": match.group(1),
                "quality": int(match.group(2)),
                "checkpoint_sha": match.group(3),
                "cdf_sha": match.group(4),
                "params": int(match.group(5)),
                "updated": bool(int(match.group(6))),
            }
            key = (record["family"], record["quality"])
            if key in seen_models:
                raise ValueError("duplicate model proof")
            seen_models.add(key)
            if record["quality"] not in QUALITIES or record["params"] <= 0:
                raise ValueError("invalid model inventory")
            if not record["updated"]:
                raise ValueError("entropy CDF update proof is missing")
            models.append(record)
        model_by_key = {
            (record["family"], record["quality"]): record
            for record in models
        }

        cases = []
        seen_cases = set()
        for line in lines:
            if not line.startswith("COMPRESS_CASE"):
                continue
            match = case_re.fullmatch(line)
            if match is None:
                raise ValueError("malformed case proof")
            record = {
                "family": match.group(1),
                "quality": int(match.group(2)),
                "image": match.group(3),
                "group": match.group(4),
                "checkpoint_sha": match.group(5),
                "params": int(match.group(6)),
                "streams": int(match.group(7)),
                "pixels": int(match.group(8)),
                "bytes": int(match.group(9)),
                "bpp": float(match.group(10)),
                "psnr": float(match.group(11)),
                "recon_sha": match.group(12),
            }
            key = (record["quality"], record["image"])
            if key in seen_cases:
                raise ValueError("duplicate encode/decode case")
            seen_cases.add(key)
            if (
                record["quality"] not in QUALITIES
                or record["image"] not in ALL_IMAGES
                or record["image"] not in GROUPS[record["group"]]
            ):
                raise ValueError("case inventory mismatch")
            model = model_by_key.get((record["family"], record["quality"]))
            if model is None:
                raise ValueError("case references an unproved model")
            if (
                record["checkpoint_sha"] != model["checkpoint_sha"]
                or record["params"] != model["params"]
                or record["streams"] != EXPECTED_STREAMS[record["family"]]
            ):
                raise ValueError("case model/checkpoint/parameter/stream mismatch")
            if record["pixels"] != 512 * 768 or record["bytes"] <= 0:
                raise ValueError("invalid case size")
            if not all(math.isfinite(record[key]) for key in ("bpp", "psnr")):
                raise ValueError("non-finite case metric")
            if not _close(
                record["bpp"],
                8.0 * record["bytes"] / record["pixels"],
                5e-10,
            ):
                raise ValueError("case bpp is not serialized bytes per original pixel")
            if not (0 < record["bpp"] < 16 and 0 < record["psnr"] < 100):
                raise ValueError("case metric outside physical bounds")
            cases.append(record)

        expected_case_keys = {
            (quality, f"kodim{index:02d}.png")
            for quality in QUALITIES
            for index in range(1, 25)
        }
        if seen_cases != expected_case_keys or len(cases) != 192:
            raise ValueError("incomplete 24-image by 8-quality case matrix")
        expected_case_order = [
            (quality, f"kodim{index:02d}.png")
            for quality in QUALITIES
            for index in range(1, 25)
        ]
        if [(case["quality"], case["image"]) for case in cases] != expected_case_order:
            raise ValueError("encode/decode cases are out of protocol order")

        choice = _derive_choice(mode, cases)
        choice_record = {"mode": mode, "surface": surface, "choice": choice}
        choice_sha = _canonical_sha(choice_record)
        dispatch = [
            {
                "quality": case["quality"],
                "image": case["image"],
                "group": case["group"],
                "family": case["family"],
            }
            for case in cases
        ]
        dispatch_sha = _canonical_sha(dispatch)
        if choice_sha != protocol_choice_sha or dispatch_sha != protocol_dispatch_sha:
            raise ValueError("policy choice/dispatch digest mismatch")
        expected_model_order = [
            (quality, family)
            for quality in QUALITIES
            for family in FAMILIES
            if any(
                case["quality"] == quality and case["family"] == family
                for case in cases
            )
        ]
        if [(model["quality"], model["family"]) for model in models] != expected_model_order:
            raise ValueError("model proofs are incomplete or out of protocol order")

        rates = {}
        rate_records = []
        for line in lines:
            if not line.startswith("COMPRESS_RATE"):
                continue
            match = rate_re.fullmatch(line)
            if match is None:
                raise ValueError("malformed rate aggregate")
            record = {
                "setting": match.group(1),
                "quality": int(match.group(2)),
                "count": int(match.group(3)),
                "pixels": int(match.group(4)),
                "bytes": int(match.group(5)),
                "bpp": float(match.group(6)),
                "psnr": float(match.group(7)),
                "rd6": float(match.group(8)),
                "rd12": float(match.group(9)),
                "rd18": float(match.group(10)),
                "target_utility": float(match.group(11)),
                "mean_params": float(match.group(12)),
                "mean_streams": float(match.group(13)),
            }
            key = (record["setting"], record["quality"])
            if key in rates:
                raise ValueError("duplicate rate aggregate")
            rates[key] = record
            rate_records.append(record)
        expected_rate_keys = {
            (setting, quality)
            for setting in SETTINGS
            for quality in QUALITIES
        }
        if set(rates) != expected_rate_keys:
            raise ValueError("incomplete rate aggregates")
        if list(rates) != [
            (setting, quality)
            for setting in SETTINGS
            for quality in QUALITIES
        ]:
            raise ValueError("rate aggregates are out of protocol order")

        for setting in SETTINGS:
            names = ALL_IMAGES if setting == "full" else GROUPS[setting]
            for quality in QUALITIES:
                selected = [
                    case
                    for case in cases
                    if case["quality"] == quality and case["image"] in names
                ]
                rate = rates[(setting, quality)]
                expected_pixels = sum(case["pixels"] for case in selected)
                expected_bytes = sum(case["bytes"] for case in selected)
                expected_bpp = 8.0 * expected_bytes / expected_pixels
                expected_psnr = _mean(selected, "psnr")
                expected = {
                    "count": len(names),
                    "pixels": expected_pixels,
                    "bytes": expected_bytes,
                    "bpp": expected_bpp,
                    "psnr": expected_psnr,
                    "rd6": expected_psnr - 6.0 * expected_bpp,
                    "rd12": expected_psnr - 12.0 * expected_bpp,
                    "rd18": expected_psnr - 18.0 * expected_bpp,
                    "target_utility": expected_psnr
                    - TARGET_PENALTY * abs(expected_bpp - TARGET_BPP[quality]),
                    "mean_params": _mean(selected, "params"),
                    "mean_streams": _mean(selected, "streams"),
                }
                if any(rate[key] != expected[key] for key in ("count", "pixels", "bytes")):
                    raise ValueError("rate aggregate count mismatch")
                if not all(
                    _close(
                        rate[key],
                        expected[key],
                        5e-6 if key in {"mean_params", "mean_streams"} else 5e-8,
                    )
                    for key in (
                        "bpp", "psnr", "rd6", "rd12", "rd18",
                        "target_utility", "mean_params", "mean_streams",
                    )
                ):
                    raise ValueError("rate aggregate metric mismatch")

        settings = []
        seen_settings = set()
        for line in lines:
            if not line.startswith("COMPRESS_SETTING"):
                continue
            match = setting_re.fullmatch(line)
            if match is None:
                raise ValueError("malformed setting aggregate")
            record = {
                "setting": match.group(1),
                "qualities": int(match.group(2)),
                "images": int(match.group(3)),
                "cases": int(match.group(4)),
                "rd6": float(match.group(5)),
                "rd12": float(match.group(6)),
                "rd18": float(match.group(7)),
                "lowq_rd12": float(match.group(8)),
                "midq_rd12": float(match.group(9)),
                "highq_rd12": float(match.group(10)),
                "target_utility": float(match.group(11)),
                "mean_psnr": float(match.group(12)),
                "mean_bpp": float(match.group(13)),
                "mean_params": float(match.group(14)),
                "mean_streams": float(match.group(15)),
            }
            setting = record["setting"]
            if setting in seen_settings:
                raise ValueError("duplicate setting aggregate")
            seen_settings.add(setting)
            per_quality = [rates[(setting, quality)] for quality in QUALITIES]

            def band_mean(band: str) -> float:
                return _mean(
                    [
                        rate
                        for rate in per_quality
                        if rate["quality"] in QUALITY_BANDS[band]
                    ],
                    "rd12",
                )

            names = ALL_IMAGES if setting == "full" else GROUPS[setting]
            expected = {
                "qualities": 8,
                "images": len(names),
                "cases": len(names) * 8,
                "rd6": _mean(per_quality, "rd6"),
                "rd12": _mean(per_quality, "rd12"),
                "rd18": _mean(per_quality, "rd18"),
                "lowq_rd12": band_mean("low"),
                "midq_rd12": band_mean("mid"),
                "highq_rd12": band_mean("high"),
                "target_utility": _mean(per_quality, "target_utility"),
                "mean_psnr": _mean(per_quality, "psnr"),
                "mean_bpp": _mean(per_quality, "bpp"),
                "mean_params": _mean(per_quality, "mean_params"),
                "mean_streams": _mean(per_quality, "mean_streams"),
            }
            if any(record[key] != expected[key] for key in ("qualities", "images", "cases")):
                raise ValueError("setting count mismatch")
            if not all(
                _close(
                    record[key],
                    expected[key],
                    5e-6 if key in {"mean_params", "mean_streams"} else 5e-8,
                )
                for key in SETTING_METRICS
            ):
                raise ValueError("setting metric mismatch")
            settings.append(record)
        if seen_settings != set(SETTINGS) or [r["setting"] for r in settings] != list(SETTINGS):
            raise ValueError("settings are incomplete or out of order")

        final_matches = [
            final_re.fullmatch(line)
            for line in lines
            if line.startswith("COMPRESS_FINAL")
        ]
        if (
            len(final_matches) != 1
            or final_matches[0] is None
            or not lines[-1].startswith("COMPRESS_FINAL")
        ):
            raise ValueError("missing, duplicate, or non-terminal final proof")
        final = final_matches[0]
        if final.group(1, 2) != (mode, surface):
            raise ValueError("final policy identity mismatch")
        if tuple(map(int, final.group(3, 4, 5))) != (24, 8, 192):
            raise ValueError("final case inventory mismatch")
        if int(final.group(6)) != len(models):
            raise ValueError("final model count mismatch")
        if int(final.group(7)) != sum(case["streams"] for case in cases):
            raise ValueError("final stream count mismatch")
        expected_hashes = (
            choice_sha,
            dispatch_sha,
            _canonical_sha(cases),
            _canonical_sha(models),
            _canonical_sha(rate_records),
            _canonical_sha(settings),
        )
        if final.group(8, 9, 10, 11, 12, 13) != expected_hashes:
            raise ValueError("terminal completion digest mismatch")
        elapsed = float(final.group(14))
        if not math.isfinite(elapsed) or elapsed <= 0:
            raise ValueError("invalid elapsed time")
        expected_line_count = 1 + len(models) + 192 + 32 + 4 + 1
        if len(lines) != expected_line_count:
            raise ValueError("unexpected output outside the complete protocol proof")

        metrics = {
            f"{metric}_{record['setting']}": record[metric]
            for record in settings
            for metric in SETTING_METRICS
        }
        return metrics, (
            f"Completed {mode} policy {surface}: 192/192 pinned bitstream cases, "
            f"{len(models)} selected family-quality models, {sum(case['streams'] for case in cases)} "
            f"entropy streams in {elapsed:.1f}s."
        )


def make_parser(expected_mode: str, expected_surface: str):
    if expected_mode not in POLICY_MODES:
        raise ValueError(f"unknown expected policy mode: {expected_mode}")
    if not expected_surface:
        raise ValueError("expected surface must not be empty")

    class Parser(PolicyParser):
        pass

    Parser.expected_mode = expected_mode
    Parser.expected_surface = expected_surface
    Parser.__name__ = "Parser"
    return Parser
