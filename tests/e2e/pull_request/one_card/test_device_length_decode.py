"""Hardware qualification of the isolated device-length attention prototype."""

import json
import math
import subprocess
import sys
from pathlib import Path


def test_device_length_decode_graph(tmp_path):
    repo = Path(__file__).resolve().parents[4]
    destination = tmp_path / "device-length.json"
    subprocess.run(
        [
            sys.executable,
            str(repo / "benchmarks/device_length_decode.py"),
            "--output",
            str(destination),
            "--repeats",
            "50",
        ],
        cwd=repo,
        check=True,
        timeout=240,
    )
    result = json.loads(destination.read_text())
    assert len(result["correctness"]) == 40
    assert all(case["passed"] for case in result["correctness"])
    for measurement in result["measurements"]:
        for name, value in measurement.items():
            if name.endswith("_us"):
                assert math.isfinite(value) and value > 0
