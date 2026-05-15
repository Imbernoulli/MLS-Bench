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
        fid_match = re.search(r"FID(?:\s*score)?:\s*([\d.]+)", raw_output, re.IGNORECASE)
        if fid_match:
            fid_val = float(fid_match.group(1))
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
