from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from mlsbench.agent.mamba_parser import StrictMambaParser


class Parser(StrictMambaParser):
    TASK = "mamba-state-init"
    EXPECTED_SURFACES = {"scheme.constant_rate", "scheme.s4d_spectrum"}
    METRIC_KIND = "MAMBA_INIT_METRICS"
