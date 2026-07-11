from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from mlsbench.agent.mamba_parser import StrictMambaParser


class Parser(StrictMambaParser):
    TASK = "mamba-a-stability"
    EXPECTED_SURFACES = {"transform.identity", "transform.neg_abs", "transform.neg_exp"}
