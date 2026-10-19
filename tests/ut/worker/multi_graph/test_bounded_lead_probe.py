# SPDX-License-Identifier: Apache-2.0

import importlib.util
from pathlib import Path

import pytest


def load():
    path = Path(__file__).resolve().parents[4] / "benchmarks" / "aclgraph_bounded_lead_probe.py"
    spec = importlib.util.spec_from_file_location("bounded_lead", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_slowest_rank_is_selected_per_sample_before_statistics():
    a = [dict(mode="padding", round=0, sample=i, wall_us=v, device_us=v - 1) for i, v in enumerate((10, 30))]
    b = [dict(mode="padding", round=0, sample=i, wall_us=v, device_us=v - 1) for i, v in enumerate((20, 5))]
    paired = load().pair_samples([a, b])
    assert [row["wall_us"] for row in paired] == [20, 30]
    assert [row["device_us"] for row in paired] == [19, 29]


@pytest.mark.parametrize("field,value", [("mode", "split_early"), ("round", 1), ("sample", 1)])
def test_mismatched_rank_identity_is_rejected(field, value):
    row = dict(mode="padding", round=0, sample=0, wall_us=10, device_us=9)
    with pytest.raises(ValueError, match="identities"):
        load().pair_samples([[row], [dict(row, **{field: value})]])


def test_truncated_rank_report_is_rejected():
    with pytest.raises(ValueError, match="equally sized"):
        load().pair_samples([[], [dict(mode="padding")]])
