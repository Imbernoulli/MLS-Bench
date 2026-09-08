"""Output parser for cv-vae-loss task.

Extracts VAE reconstruction metrics from training output.
"""

import re

from mlsbench.agent.parsers import OutputParser, ParseResult


class Parser(OutputParser):

    def parse(self, cmd_label: str, raw_output: str) -> ParseResult:
        feedback_parts: list[str] = []
        metrics: dict = {}

        for line in raw_output.splitlines():
            stripped = line.strip()

            # Training progress
            if stripped.startswith("step ") or "Model parameters" in stripped:
                feedback_parts.append(stripped)

            # Train-time metrics
            if "TRAIN_METRICS:" in stripped:
                feedback_parts.append(stripped)

            # Final test metrics
            if "TEST_METRICS:" in stripped:
                feedback_parts.append(stripped)
                rfid_m = re.search(r"rfid=([\d.]+)", stripped)
                psnr_m = re.search(r"psnr=([\d.]+)", stripped)
                ssim_m = re.search(r"ssim=([\d.]+)", stripped)
                best_rfid_m = re.search(r"best_rfid=([\d.]+)", stripped)
                if rfid_m:
                    metrics["rfid"] = float(rfid_m.group(1))
                if psnr_m:
                    metrics["psnr"] = float(psnr_m.group(1))
                if ssim_m:
                    metrics["ssim"] = float(ssim_m.group(1))
                if best_rfid_m:
                    metrics["best_rfid"] = float(best_rfid_m.group(1))

                size = None
                for s in ("small", "medium", "large"):
                    if s in cmd_label:
                        size = s
                        break
                if size:
                    if rfid_m:
                        metrics[f"rfid_{size}"] = float(rfid_m.group(1))
                    if psnr_m:
                        metrics[f"psnr_{size}"] = float(psnr_m.group(1))
                    if ssim_m:
                        metrics[f"ssim_{size}"] = float(ssim_m.group(1))
                    if best_rfid_m:
                        metrics[f"best_rfid_{size}"] = float(best_rfid_m.group(1))

        # If no TEST_METRICS found, include tail of raw output for debugging
        if not metrics:
            tail_lines = raw_output.strip().splitlines()[-50:]
            feedback_parts.append("\n--- DEBUG: no TEST_METRICS found, showing tail ---")
            feedback_parts.extend(tail_lines)

        feedback = "\n".join(feedback_parts) if feedback_parts else raw_output
        return ParseResult(feedback=feedback, metrics=metrics)
