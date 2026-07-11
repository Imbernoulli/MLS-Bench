from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from mlsbench.agent.mamba_parser import StrictMambaParser


class Parser(StrictMambaParser):
    TASK = "mamba-dt-init"
    EXPECTED_SURFACES = {"scheme.too_large", "scheme.too_small", "scheme.log_uniform_s4d"}
    METRIC_KIND = "MAMBA_INIT_METRICS"
