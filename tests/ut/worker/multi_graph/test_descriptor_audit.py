# SPDX-License-Identifier: Apache-2.0

import itertools
from pathlib import Path

import pytest


@pytest.mark.parametrize("costs", [(0, 0, 0), (1, 0, 2, 0, 1), (9, 1, 1, 9, 1, 1), (1, 2, 3, 4, 5)])
@pytest.mark.parametrize("cores", [1, 2, 3, 5])
def test_jump_partition_matches_independent_minimax(monkeypatch, costs, cores):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[4] / "benchmarks"))
    from aclgraph_descriptor_audit import jump_cuts

    cuts = jump_cuts(costs, cores)
    used = max(1, min(cores, sum(c > 0 for c in costs)))
    assert len(cuts) == used and cuts[-1] == len(costs)
    assert all(a < b for a, b in zip([0, *cuts[:-1]], cuts))
    observed = max(sum(costs[a:b]) for a, b in zip([0, *cuts[:-1]], cuts))
    candidates = []
    for middle in itertools.combinations(range(1, len(costs)), used - 1):
        ends = [*middle, len(costs)]
        loads = [sum(costs[a:b]) for a, b in zip([0, *middle], ends)]
        if not any(costs) or all(loads):
            candidates.append(max(loads))
    assert observed == min(candidates)


def test_endpoint_audit_rejects_incomplete_duplicate_and_invalid_cost(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[4] / "benchmarks"))
    from aclgraph_descriptor_audit import measured_costs

    samples = [
        dict(trial=t, mode=m, sample=s, device_us=10 + t, wall_us=100 + t)
        for t, m, s in itertools.product(range(4), ("aicpu", "aiv"), range(10))
    ]
    result = measured_costs(dict(samples=samples), ("aicpu", "aiv"))
    assert result["device_us"]["aiv"] == 11.5
    assert result["wall_us"]["aiv"] == 101.5
    for invalid in (samples[:-1], [*samples[:-1], samples[0]]):
        with pytest.raises(ValueError, match="incomplete or duplicate"):
            measured_costs(dict(samples=invalid), ("aicpu", "aiv"))
    with pytest.raises(ValueError, match="nonpositive"):
        measured_costs(dict(samples=[samples[0] | dict(device_us=0), *samples[1:]]), ("aicpu", "aiv"))


def test_profile_inventory_does_not_double_count_wait(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[4] / "benchmarks"))
    from aclgraph_descriptor_audit import profile_inventory

    def event(name, task, duration):
        return dict(ph="X", name=name, dur=duration, args={"Task Type": task})

    events = [
        event("MEMCPY_ASYNC", "PCIE_DMA_SQE", 2),
        event("descriptor_plan", "KERNEL_AIVEC", 190),
        event("NOTIFY_WAIT", "NOTIFY_WAIT", 185),
    ]
    result = profile_inventory(events, "aiv")
    assert len(result["kernels"]) == 1 and result["kernels"][0]["duration_us"] == 190
    assert result["dma_us"] == [2]
    with pytest.raises(ValueError, match="inventory"):
        profile_inventory(events[:1], "aiv")
