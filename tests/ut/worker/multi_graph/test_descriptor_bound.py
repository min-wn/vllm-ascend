# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "costs",
    [
        (0,) * 12,
        (1000, 1, 1, 1, 1, 1),
        (1, 1, 1, 1, 1, 1000),
        (0, 1000, 0, 1, 0, 1, 0, 1000, 0),
        (1,) * 24,
        (0, 1, 0, 2, 0),
    ],
)
@pytest.mark.parametrize("cores", [1, 2, 20])
def test_feasible_bound_retains_exact_canonical_partition(monkeypatch, costs, cores):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[4] / "benchmarks"))
    from aclgraph_descriptor_bound_audit import check

    record = check(costs, cores)
    assert record["lower"] <= record["limit"] <= record["upper"] <= sum(costs)
    assert sum(record["chunk_loads"]) == sum(costs)
    assert record["prefix_ends"] == sorted(record["prefix_ends"])


def test_empty_quantile_chunks_are_not_exported_to_consumer(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[4] / "benchmarks"))
    from aclgraph_descriptor_bound_audit import check

    record = check((1000, 1, 1, 1, 1, 1), 5)
    assert len(set(record["prefix_ends"])) < len(record["prefix_ends"])
    assert len(set(record["cuts"])) == len(record["cuts"]) == 5
    assert record["limit"] == 1000


@pytest.mark.parametrize("costs,cores", [([], 1), ([-1], 1), ([1.5], 1), ([1], 0), ([1], True)])
def test_bound_rejects_out_of_domain_costs(monkeypatch, costs, cores):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[4] / "benchmarks"))
    from aclgraph_descriptor_bound_audit import prefix_bound

    with pytest.raises(ValueError, match="nonnegative integer"):
        prefix_bound(costs, cores)


def test_stability_gate_uses_aicpu_and_requires_wall_gain_in_every_round(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[4] / "benchmarks"))
    from aclgraph_descriptor_bound_audit import paired_gate

    def rows(bounded):
        return dict(
            rounds=[
                dict(
                    trial=t,
                    **{
                        endpoint: dict(aicpu=100, aiv_unbounded=200, aiv_bounded=bounded)
                        for endpoint in ("device_us", "wall_us")
                    },
                )
                for t in range(4)
            ]
        )

    assert paired_gate(rows(90))[1]
    assert not paired_gate(rows(150))[1]  # Beating only old AIV is insufficient.
    partial = rows(90)
    partial["rounds"][3]["wall_us"]["aiv_bounded"] = 101
    assert not paired_gate(partial)[1]
