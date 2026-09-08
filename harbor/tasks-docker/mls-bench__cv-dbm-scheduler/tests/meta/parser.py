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
        # Require a decimal point and take the LAST match: when FID reference
        # statistics are generated on the fly, the inception feature loop prints a
        # tqdm bar "FID:   0%|..." whose integer "0" a greedy "[\d.]+" would match
        # instead of the real "FID: 5.55" result line (zeroing best_fid). tqdm
        # percentages are integers; the real FID is fractional, so \d+\.\d+ skips
        # the bar (an optional exponent tolerates scientific notation). Mirrors
        # the fix in tasks/cv-dbm-sampler/parser.py.
        _fid_vals = re.findall(
            r"FID(?:\s*score)?:\s*(\d+\.\d+(?:[eE][+-]?\d+)?)", raw_output, re.IGNORECASE
        )
        if not _fid_vals:
            _fid_vals = re.findall(
                r"'fid':\s*(?:np\.float64\()?(\d+\.\d+(?:[eE][+-]?\d+)?)", raw_output
            )
        if _fid_vals:
            fid_val = float(_fid_vals[-1])
            label = cmd_label.replace("-", "_")
            metrics["fid"] = fid_val
            metrics["best_fid"] = fid_val
            metrics[f"fid_{label}"] = fid_val
            metrics[f"best_fid_{label}"] = fid_val
            feedback = (
                f"Optimization Feedback ({cmd_label}): "
                f"Your modification yielded an FID of {fid_val:.4f}."
            )
        else:
            last_logs = "\n".join(raw_output.splitlines()[-50:])
            feedback = f"Could not find FID score in output. Last logs:\n{last_logs}"
            
        return ParseResult(feedback=feedback, metrics=metrics)
