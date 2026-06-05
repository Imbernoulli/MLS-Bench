import re
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mlsbench.agent.parsers import OutputParser, ParseResult

class Parser(OutputParser):
    def parse(self, cmd_label: str, raw_output: str) -> ParseResult:
        metrics = {}
        feedback = ""
        # Extract the FID value. Require a decimal point and take the LAST match:
        # when the FID reference statistics are generated on the fly (precomputed
        # assets/stats/fid_imagenet_256_val.npz missing), the inception feature
        # loop prints a tqdm bar "FID:   0%|...", "FID:   1%|..." etc. A greedy
        # "[\d.]+" matched the integer "0" from that progress bar instead of the
        # real "FID: 5.55" result line, zeroing best_fid_Imagenet. tqdm
        # percentages are integers; the real FID is always fractional, so
        # \d+\.\d+ skips the bar and matches the result line.
        _fid_vals = re.findall(r"FID(?:\s*score)?:\s*(\d+\.\d+)", raw_output, re.IGNORECASE)
        # Also match dict format: {'fid': np.float64(<number>), ...}
        if not _fid_vals:
            _fid_vals = re.findall(r"'fid':\s*(?:np\.float64\()?(\d+\.\d+)", raw_output)
        fid_str = _fid_vals[-1] if _fid_vals else None
        
        # NFE budget enforcement: reject if the karras_sample wrapper reports
        # ACTUAL_NFE > EXPECTED_NFE (agent double-denoised / Heun-corrected
        # beyond budget). Also detect the hard-error case.
        actual_match = re.search(r"ACTUAL_NFE:\s*(\d+)\s*/\s*EXPECTED_NFE:\s*(\d+)", raw_output)
        nfe_exceeded = "NFE_BUDGET_EXCEEDED" in raw_output
        nfe_info = ""
        if actual_match:
            actual_nfe, expected_nfe = int(actual_match.group(1)), int(actual_match.group(2))
            nfe_info = f" (NFE used: {actual_nfe}/{expected_nfe})"
            if actual_nfe > expected_nfe:
                nfe_exceeded = True

        if nfe_exceeded:
            feedback = (
                f"[{cmd_label}] NFE_BUDGET_EXCEEDED{nfe_info}. Your sampler made more "
                f"denoiser calls than allowed. Do not double-denoise, Heun-correct, or "
                f"any trick that uses extra model passes beyond the NFE budget. This "
                f"result is REJECTED and not recorded."
            )
            # Do not write metrics — leaderboard row stays empty
        elif fid_str:
            fid_val = round(float(fid_str), 3)
            # Key by env label so both envs' FIDs survive in the leaderboard
            # (a bare "fid" key makes the second env overwrite the first).
            # Mirror to best_fid_<label> for score_spec compatibility.
            if cmd_label:
                metrics[f"fid_{cmd_label}"] = fid_val
                metrics[f"best_fid_{cmd_label}"] = fid_val
            else:
                metrics["fid"] = fid_val
                metrics["best_fid"] = fid_val
            feedback = f"Optimization Feedback: {cmd_label} yielded an FID of {fid_val:.3f}{nfe_info}."
        else:
            last_logs = "\n".join(raw_output.splitlines()[-50:])
            feedback = f"[{cmd_label}] Could not find FID score in output. Last logs:\n{last_logs}"
            
        return ParseResult(feedback=feedback, metrics=metrics)
