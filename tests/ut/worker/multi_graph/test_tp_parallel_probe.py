import importlib.util
from pathlib import Path

import pytest
import torch


@pytest.fixture
def probe(monkeypatch):
    directory = Path(__file__).resolve().parents[4] / "benchmarks"
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location("tp_parallel_probe", directory / "multi_graph_tp_parallel_probe.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reverse_order_keeps_original_request_slices(probe):
    assert probe.branch_slices(400, 272) == ((0, 272), (272, 400))
    assert probe.branch_slices(400, 272, (1, 0)) == ((272, 400), (0, 272))


@pytest.mark.parametrize("boundary,order", [(0, (0, 1)), (400, (0, 1)), (200, (0, 0)), (200, (1,))])
def test_slices_reject_missing_or_empty_branches(probe, boundary, order):
    with pytest.raises(ValueError, match="nonempty"):
        probe.branch_slices(400, boundary, order)


def test_plan_has_whole_controls_and_same_arithmetic_pairs(probe):
    cases = probe.probe_cases(400)
    by_name = {case.name: case for case in cases}
    assert len(by_name) == len(cases) == 12
    assert {case.family for case in cases if case.boundary is None} == {"down", "norm"}
    for control, candidate, mechanism in probe.paired_comparisons(cases):
        left, right = by_name[control], by_name[candidate]
        assert left.family == right.family
        if mechanism == "overlap":
            assert (left.boundary, left.order) == (right.boundary, right.order)
            assert not left.overlap and right.overlap
        elif mechanism == "branch_order":
            assert left.boundary == right.boundary == 272
            assert left.overlap and right.overlap
            assert left.order == (0, 1) and right.order == (1, 0)
        else:
            assert mechanism == "versus_full_bucket" and left.boundary is None
    assert by_name["down_272_128_reverse_overlap"].boundary == 272


@pytest.mark.parametrize(
    "rows,pairs", [(0, ((1, 1),)), (400, ()), (400, ((200, 0),)), (400, ((200, 199),)), (400, ((200, 200), (200, 200)))]
)
def test_plan_rejects_inconsistent_capacities(probe, rows, pairs):
    with pytest.raises(ValueError):
        probe.probe_cases(rows, pairs)


@pytest.mark.parametrize("rounds,repeats", [(0, 6), (3, 0)])
def test_measure_rejects_invalid_sample_counts_before_npu_access(probe, rounds, repeats):
    value = torch.zeros(4, 8)
    with pytest.raises(ValueError, match="positive"):
        probe.measure_parallel_probe(
            value, value, value, None, None, None, None, None, None, rounds=rounds, repeats=repeats
        )


def test_measure_rejects_unmatched_rows_before_npu_access(probe):
    value = torch.zeros(4, 8)
    with pytest.raises(ValueError, match="same rank-two shape"):
        probe.measure_parallel_probe(value, value[:2], value, None, None, None, None, None, None)


def test_measure_rejects_gate_up_channel_partitioning(probe):
    value = torch.zeros(4, 8)
    with pytest.raises(ValueError, match="one whole GateUp"):
        probe.measure_parallel_probe(value, value, value, None, None, (value, value), value, None, None)


def test_measure_rejects_nonordinary_mlp_shapes(probe):
    value = torch.zeros(4, 8)
    with pytest.raises(ValueError, match="ordinary GateUp/Down"):
        probe.measure_parallel_probe(value, value, value, None, None, torch.zeros(16, 8), torch.zeros(8, 7), None, None)


def test_priority_reorders_only_branch_collectives(probe):
    from multi_graph_tp_priority_probe import priority_orders

    orders = priority_orders(400, (200, 200))
    assert orders["fifo"] == orders["fifo_barrier"] == ("O_A", "O_B", "Down_A", "Down_B")
    assert orders["depth_priority"] == ("O_A", "Down_A", "O_B", "Down_B")
    assert set(orders["depth_priority"]) == set(orders["fifo"])
    for pair in ((0, 400), (200, 199), (400,), (200, 200, 0)):
        with pytest.raises(ValueError):
            priority_orders(400, pair)
