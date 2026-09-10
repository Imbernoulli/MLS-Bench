"""Mid-edit: create the editable template for ai4sci-weather-forecast-aggregation."""

from pathlib import Path

_TEMPLATE = Path(__file__).parent / "custom_template.py"
_CONTENT = _TEMPLATE.read_text()

OPS = [
    {
        "op": "create",
        "file": "ClimaX/custom_forecast.py",
        "content": _CONTENT,
    },
]
