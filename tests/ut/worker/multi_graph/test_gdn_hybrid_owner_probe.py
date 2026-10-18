"""Verify uneven stable owners before investing in the NPU dense budget gate."""

import importlib.util
import json
from pathlib import Path

import pytest
import torch


@pytest.fixture
def probe():
    path = Path(__file__).resolve().parents[4] / "benchmarks" / "multi_graph_gdn_hybrid_owner_probe.py"
    spec = importlib.util.spec_from_file_location("gdn_hybrid_owner_probe_cpu", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fixed_trace_owner_capacity_cost_is_counted(probe):
    root = Path(__file__).resolve().parents[4] / "benchmarks" / "results" / "multi_graph_structural"
    trace = json.loads((root / "structural-dynamic-trace-v1.json").read_text())["trace"]
    catalog = json.loads((root / "gdn-q4-full-allocation-v1.rank0.json").read_text())["selection"]["capture_sizes"]
    audit = probe.audit_owner_capacities(trace, catalog)
    assert audit["total_padding_capacity_tokens"] == 50688
    assert audit["total_hybrid_capacity_tokens"] == 52160
    assert audit["total_capacity_increase_percent"] == pytest.approx(2.9040404040404)
    assert audit["same_total_unequal_pair_steps"] == 107
    assert len(audit["required_owner_tokens"]) == 17
    worst = audit["worst_capacity_step"]
    assert (worst["step"], worst["owner_live_requests"], worst["padding_tokens"], worst["hybrid_mlp_tokens"]) == (
        53,
        [37, 27],
        256,
        304,
    )
    assert all(row["hybrid_mlp_tokens"] in catalog for row in audit["steps"])


def test_unequal_owner_gather_reconstruction_preserves_request_and_token_order(probe):
    requests = [8, 3, 2, 6, 5, 12, 14]
    rows, owner_rows, width = 32, 24, 3
    canonical = torch.arange(rows * width).reshape(rows, width)
    local = []
    for rank in (0, 1):
        plan = probe.owner_row_plan(requests, rows, owner_rows, rank)
        owned = torch.zeros(owner_rows, width, dtype=canonical.dtype)
        indices = list(plan["local_token_indices"])
        owned[: len(indices)] = canonical[indices]
        local.append(owned)
    gathered = torch.cat(local)
    active = gathered[list(plan["gather_active_token_indices"])]
    restored = torch.zeros_like(canonical)
    restored[list(plan["canonical_active_token_indices"])] = active
    assert torch.equal(restored[: len(requests) * 4], canonical[: len(requests) * 4])
    assert torch.equal(restored[len(requests) * 4 :], torch.zeros_like(restored[len(requests) * 4 :]))


@pytest.mark.parametrize("bad", [([2, 2], 8, 8, 0), ([2, 4, 6], 12, 8, 0), ([0], 2, 4, 0), ([0], 4, 4, 2)])
def test_owner_row_plan_rejects_aliasing_or_overflow(probe, bad):
    with pytest.raises(ValueError):
        probe.owner_row_plan(*bad)


def test_capacity_audit_refuses_nondefault_reconstructed_mlp(probe):
    trace = dict(steps=[dict(step=0, batch=3, request_ids=[0, 2, 1])])
    with pytest.raises(ValueError, match="reconstructed MLP"):
        probe.audit_owner_capacities(trace, [4, 8, 12])
