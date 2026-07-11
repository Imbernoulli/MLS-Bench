"""Strict parser for the full Kodak-24 CompressAI bitstream protocol."""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mlsbench.agent.parsers import OutputParser, ParseResult


PROTOCOL = "compressai_zoo_kodak24_q1q8_bitstream_v1"
QUALITIES = tuple(range(1, 9))
GROUPS = {
    "low": {"kodim02.png", "kodim03.png", "kodim09.png", "kodim10.png", "kodim12.png", "kodim15.png", "kodim20.png", "kodim23.png"},
    "mid": {"kodim04.png", "kodim07.png", "kodim11.png", "kodim16.png", "kodim17.png", "kodim19.png", "kodim21.png", "kodim22.png"},
    "high": {"kodim01.png", "kodim05.png", "kodim06.png", "kodim08.png", "kodim13.png", "kodim14.png", "kodim18.png", "kodim24.png"},
}
ALL_IMAGES = set().union(*GROUPS.values())
SETTINGS = ("full", "low", "mid", "high")
RD_BETA = 12.0
HEX64 = r"[0-9a-f]{64}"
NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"


def _canonical_sha(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _close(left: float, right: float, tolerance: float = 5e-8) -> bool:
    return math.isclose(left, right, rel_tol=tolerance, abs_tol=tolerance)


class Parser(OutputParser):
    def parse(self, cmd_label: str, raw_output: str) -> ParseResult:
        del cmd_label
        try:
            metrics, feedback = self._parse_strict(raw_output)
            return ParseResult(feedback=feedback, metrics=metrics)
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            tail = raw_output[-3000:]
            return ParseResult(feedback=f"full protocol rejected: {exc}\n{tail}", metrics={})

    def _parse_strict(self, raw_output: str) -> tuple[dict, str]:
        lines = [line.strip() for line in raw_output.splitlines() if line.strip()]
        if not lines:
            raise ValueError("empty verifier output")
        if any(marker in raw_output for marker in ("Traceback (most recent call last)", "SURFACE_ERROR", "COMPRESS_FAILED")):
            raise ValueError("failure marker present")

        protocol_re = re.compile(
            rf"^COMPRESS_PROTOCOL protocol={PROTOCOL} family=(factorized|hyperprior_scale|meanscale) "
            rf"images=24 qualities=8 cases=192 protocol_sha=({HEX64}) network=blocked$"
        )
        model_re = re.compile(
            rf"^COMPRESS_MODEL quality=(\d+) checkpoint_sha=({HEX64}) cdf_sha=({HEX64}) "
            rf"params=(\d+) updated=([01])$"
        )
        case_re = re.compile(
            rf"^COMPRESS_CASE quality=(\d+) image=(kodim\d{{2}}\.png) group=(low|mid|high) "
            rf"pixels=(\d+) bytes=(\d+) bpp=({NUMBER}) psnr=({NUMBER}) recon_sha=({HEX64})$"
        )
        rate_re = re.compile(
            rf"^COMPRESS_RATE setting=(full|low|mid|high) quality=(\d+) count=(\d+) "
            rf"pixels=(\d+) bytes=(\d+) bpp=({NUMBER}) psnr=({NUMBER}) utility=({NUMBER})$"
        )
        setting_re = re.compile(
            rf"^COMPRESS_SETTING setting=(full|low|mid|high) qualities=(\d+) images=(\d+) "
            rf"cases=(\d+) mean_rd_utility=({NUMBER}) mean_psnr=({NUMBER}) mean_bpp=({NUMBER})$"
        )
        final_re = re.compile(
            rf"^COMPRESS_FINAL protocol={PROTOCOL} family=(factorized|hyperprior_scale|meanscale) "
            rf"images=(\d+) qualities=(\d+) cases=(\d+) streams=(\d+) "
            rf"cases_sha=({HEX64}) checkpoints_sha=({HEX64}) settings_sha=({HEX64}) elapsed=({NUMBER})$"
        )

        protocol_rows = [protocol_re.fullmatch(line) for line in lines if line.startswith("COMPRESS_PROTOCOL")]
        if len(protocol_rows) != 1 or protocol_rows[0] is None:
            raise ValueError("missing or duplicate protocol proof")
        family = protocol_rows[0].group(1)

        models = []
        for line in lines:
            if not line.startswith("COMPRESS_MODEL"):
                continue
            match = model_re.fullmatch(line)
            if match is None:
                raise ValueError("malformed model proof")
            record = {
                "quality": int(match.group(1)),
                "checkpoint_sha": match.group(2),
                "cdf_sha": match.group(3),
                "params": int(match.group(4)),
                "updated": bool(int(match.group(5))),
            }
            models.append(record)
        if [record["quality"] for record in models] != list(QUALITIES):
            raise ValueError("incomplete or out-of-order checkpoint proof")
        if any(record["params"] <= 0 for record in models):
            raise ValueError("invalid parameter count")
        if any(not record["updated"] for record in models):
            raise ValueError("entropy CDF update proof is missing")

        cases = []
        seen_cases = set()
        for line in lines:
            if not line.startswith("COMPRESS_CASE"):
                continue
            match = case_re.fullmatch(line)
            if match is None:
                raise ValueError("malformed case proof")
            quality = int(match.group(1))
            image = match.group(2)
            group = match.group(3)
            pixels = int(match.group(4))
            nbytes = int(match.group(5))
            bpp = float(match.group(6))
            psnr = float(match.group(7))
            key = (quality, image)
            if key in seen_cases:
                raise ValueError("duplicate encode/decode case")
            seen_cases.add(key)
            if quality not in QUALITIES or image not in ALL_IMAGES or image not in GROUPS[group]:
                raise ValueError("case inventory mismatch")
            if pixels != 512 * 768 or nbytes <= 0:
                raise ValueError("invalid case size")
            if not all(math.isfinite(value) for value in (bpp, psnr)):
                raise ValueError("non-finite case metric")
            if not _close(bpp, 8.0 * nbytes / pixels, 5e-10):
                raise ValueError("case bpp is not exact serialized bytes per original pixel")
            if not (0 < bpp < 16 and 0 < psnr < 100):
                raise ValueError("case metric outside physical bounds")
            cases.append(
                {
                    "quality": quality,
                    "image": image,
                    "group": group,
                    "pixels": pixels,
                    "bytes": nbytes,
                    "bpp": bpp,
                    "psnr": psnr,
                    "recon_sha": match.group(8),
                }
            )
        expected_cases = {(quality, image) for quality in QUALITIES for image in ALL_IMAGES}
        if seen_cases != expected_cases or len(cases) != 192:
            raise ValueError("incomplete 24-image by 8-quality case matrix")
        expected_case_order = [
            (quality, f"kodim{index:02d}.png")
            for quality in QUALITIES
            for index in range(1, 25)
        ]
        if [(case["quality"], case["image"]) for case in cases] != expected_case_order:
            raise ValueError("encode/decode cases are out of protocol order")

        rates = {}
        for line in lines:
            if not line.startswith("COMPRESS_RATE"):
                continue
            match = rate_re.fullmatch(line)
            if match is None:
                raise ValueError("malformed rate aggregate")
            setting = match.group(1)
            quality = int(match.group(2))
            key = (setting, quality)
            if key in rates:
                raise ValueError("duplicate rate aggregate")
            rates[key] = {
                "count": int(match.group(3)),
                "pixels": int(match.group(4)),
                "bytes": int(match.group(5)),
                "bpp": float(match.group(6)),
                "psnr": float(match.group(7)),
                "utility": float(match.group(8)),
            }
        if set(rates) != {(setting, quality) for setting in SETTINGS for quality in QUALITIES}:
            raise ValueError("incomplete rate aggregates")
        if list(rates) != [
            (setting, quality) for setting in SETTINGS for quality in QUALITIES
        ]:
            raise ValueError("rate aggregates are out of protocol order")

        for setting in SETTINGS:
            names = ALL_IMAGES if setting == "full" else GROUPS[setting]
            for quality in QUALITIES:
                selected = [case for case in cases if case["quality"] == quality and case["image"] in names]
                rate = rates[(setting, quality)]
                expected_count = len(names)
                expected_pixels = sum(case["pixels"] for case in selected)
                expected_bytes = sum(case["bytes"] for case in selected)
                expected_bpp = 8.0 * expected_bytes / expected_pixels
                expected_psnr = sum(case["psnr"] for case in selected) / expected_count
                expected_utility = expected_psnr - RD_BETA * expected_bpp
                if (rate["count"], rate["pixels"], rate["bytes"]) != (expected_count, expected_pixels, expected_bytes):
                    raise ValueError("rate aggregate count mismatch")
                if not all(
                    _close(rate[key], expected, 5e-8)
                    for key, expected in (("bpp", expected_bpp), ("psnr", expected_psnr), ("utility", expected_utility))
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
            setting = match.group(1)
            if setting in seen_settings:
                raise ValueError("duplicate setting aggregate")
            seen_settings.add(setting)
            record = {
                "setting": setting,
                "qualities": int(match.group(2)),
                "images": int(match.group(3)),
                "cases": int(match.group(4)),
                "mean_rd_utility": float(match.group(5)),
                "mean_psnr": float(match.group(6)),
                "mean_bpp": float(match.group(7)),
            }
            names = ALL_IMAGES if setting == "full" else GROUPS[setting]
            expected_rates = [rates[(setting, quality)] for quality in QUALITIES]
            expected = {
                "qualities": 8,
                "images": len(names),
                "cases": len(names) * 8,
                "mean_rd_utility": sum(rate["utility"] for rate in expected_rates) / 8,
                "mean_psnr": sum(rate["psnr"] for rate in expected_rates) / 8,
                "mean_bpp": sum(rate["bpp"] for rate in expected_rates) / 8,
            }
            if any(record[key] != expected[key] for key in ("qualities", "images", "cases")):
                raise ValueError("setting count mismatch")
            if not all(_close(record[key], expected[key], 5e-8) for key in ("mean_rd_utility", "mean_psnr", "mean_bpp")):
                raise ValueError("setting metric mismatch")
            settings.append(record)
        if seen_settings != set(SETTINGS) or len(settings) != 4:
            raise ValueError("incomplete required settings")
        if [record["setting"] for record in settings] != list(SETTINGS):
            raise ValueError("setting aggregates are out of protocol order")

        final_rows = [final_re.fullmatch(line) for line in lines if line.startswith("COMPRESS_FINAL")]
        if len(final_rows) != 1 or final_rows[0] is None or not lines[-1].startswith("COMPRESS_FINAL"):
            raise ValueError("missing, duplicate, or non-terminal final proof")
        final = final_rows[0]
        if final.group(1) != family or tuple(map(int, final.group(2, 3, 4))) != (24, 8, 192):
            raise ValueError("final inventory mismatch")
        if int(final.group(5)) < 192:
            raise ValueError("final entropy stream count is incomplete")
        if final.group(6) != _canonical_sha(cases):
            raise ValueError("case completion digest mismatch")
        if final.group(7) != _canonical_sha(models):
            raise ValueError("checkpoint/CDF completion digest mismatch")
        if final.group(8) != _canonical_sha(settings):
            raise ValueError("setting completion digest mismatch")
        elapsed = float(final.group(9))
        if not math.isfinite(elapsed) or elapsed <= 0:
            raise ValueError("invalid elapsed time")

        expected_line_count = 1 + 8 + 192 + 32 + 4 + 1
        if len(lines) != expected_line_count:
            raise ValueError("unexpected output outside the complete protocol proof")

        metrics = {}
        for record in settings:
            setting = record["setting"]
            metrics[f"mean_rd_utility_{setting}"] = record["mean_rd_utility"]
            metrics[f"psnr_{setting}"] = record["mean_psnr"]
            metrics[f"bpp_{setting}"] = record["mean_bpp"]
        feedback = (
            f"Completed official CompressAI {family} on Kodak-24: 192/192 actual "
            f"bitstream cases in {elapsed:.1f}s; full mean R-D utility="
            f"{metrics['mean_rd_utility_full']:.4f}."
        )
        return metrics, feedback
