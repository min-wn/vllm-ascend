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


def test_join_preserves_order_all_fields_and_null_output(monkeypatch):
    import torch

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[4] / "benchmarks"))
    from multi_graph_tp_gdn_phases import GDNPhase

    owner = object()
    values = [torch.arange(24).reshape(4, 6) + offset for offset in range(6)]
    values[4].zero_()
    phases = [
        GDNPhase(owner, name, *(v[start:stop] for v in values))
        for name, start, stop in (("left", 0, 2), ("right", 2, 4))
    ]
    merged = load().join_projected_phases(phases)
    assert merged.name == "full" and merged.owner is owner
    for field, expected in zip(("mixed_qkv", "z", "b", "a", "core_output", "residual"), values, strict=True):
        assert torch.equal(getattr(merged, field), expected)
    assert all(p.completed for p in phases)
    with pytest.raises(ValueError, match="unconsumed"):
        load().join_projected_phases(phases)


@pytest.mark.parametrize("invalid", ["order", "owner", "core", "extent"])
def test_join_rejects_invalid_phases(monkeypatch, invalid):
    import torch

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[4] / "benchmarks"))
    from multi_graph_tp_gdn_phases import GDNPhase

    owner = object()
    phases = [GDNPhase(owner, name, *(torch.zeros(2, 3) for _ in range(6))) for name in ("left", "right")]
    if invalid == "order":
        phases.reverse()
    elif invalid == "owner":
        phases[1].owner = object()
    elif invalid == "core":
        phases[0].normalized_output = torch.zeros(2, 3)
    else:
        phases[0].mixed_qkv = torch.zeros(1, 3)
    with pytest.raises(ValueError):
        load().join_projected_phases(phases)
