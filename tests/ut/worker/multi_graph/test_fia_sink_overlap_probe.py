# SPDX-License-Identifier: Apache-2.0
"""Keep optimistic resource controls balanced and scope explicit."""

import importlib
from pathlib import Path


def test_mode_order_is_counterbalanced(monkeypatch):
    repo = Path(__file__).resolve().parents[4]
    monkeypatch.syspath_prepend(str(repo / "benchmarks"))
    probe = importlib.import_module("aclgraph_fia_sink_overlap_probe")
    assert probe.measurement_order(1) == probe.measurement_order(0)[::-1]
    assert set(probe.measurement_order(0)) == {"full_serial", "half_serial", "full_parallel", "half_parallel"}
    assert probe.measurement_order(2) == probe.measurement_order(0)
