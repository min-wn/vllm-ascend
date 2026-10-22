# SPDX-License-Identifier: Apache-2.0
"""CPU checks for the bounded complete-layer experiment's fixed contract."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / "benchmarks/multi_graph_gdn_q14_layer_protocol.py"


def module():
    assert SOURCE.exists(), "complete-layer protocol is not implemented"
    sys.path.insert(0, str(SOURCE.parent))
    try:
        spec = importlib.util.spec_from_file_location("q14_layer_protocol_tested", SOURCE)
        result = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(result)
        return result
    finally:
        sys.path.pop(0)


def row(owners=(2, 0), lengths=(1, 4), accepted=(4, 1)):
    return dict(request_ids=list(owners), query_lengths=list(lengths), accepted_from_previous=list(accepted))


def samples(baseline=(100.0, 100.1), candidate=(99.0, 99.1)):
    result = []
    for process in range(2):
        for config, times in (("forced_q4", baseline), ("q14_ready", candidate)):
            for sample in range(8):
                cost = times[process] + (sample % 2) * 0.002
                result.append(
                    dict(
                        process=process,
                        config=config,
                        sample=sample,
                        initialization_ms=1.0,
                        execution_ms=cost - 1.5,
                        terminal_ms=0.5,
                        lifecycle_ms=cost,
                    )
                )
    return result


def test_packing_preserves_previous_acceptance_and_maps_only_executed_rows():
    p = module().payload(row(), 3, "q14_ready", {2: 4, 0: 1})
    assert p["sequence"] == [0, 1, 4, 0]
    assert p["query_start_loc"] == [0, 1, 5, 5]
    assert p["ids"] == [3, 1, 1, 1, 1, *([0] * 7)]
    assert p["accepted"] == [4, 1, 1]
    assert p["convslots"] == [[9, 10, 11, 12], [1, 2, 3, 4], [0, 0, 0, 0]]
    assert p["pack_indices"] == [0, 4, 5, 6, 7, *([0] * 7)]
    assert p["inverse_indices"] == [0, 12, 12, 12, 1, 2, 3, 4, 12, 12, 12, 12]
    assert p["valid_dense_mask"] == [True, False, False, False, True, True, True, True, *([False] * 4)]


def test_forced_q4_has_common_dense_capacity_and_actual_validity():
    p = module().payload(row(), 3, "forced_q4")
    assert p["sequence"] == [0, 4, 4, 0]
    assert p["query_start_loc"] == [0, 4, 8, 8]
    assert p["pack_indices"] == [*range(8), 0, 0, 0, 0]
    assert p["inverse_indices"] == [*range(8), 12, 12, 12, 12]
    assert p["actual_query_tokens"] == 5
    assert p["executed_query_tokens"] == 8
    assert sum(p["valid_dense_mask"]) == 5


def test_pause_reorder_and_query_switch_do_not_reassign_owner_slots():
    m = module()
    previous = {0: 4, 1: 1, 2: 4}
    first = m.payload(row((2, 0), (1, 4), (4, 1)), 3, "q14_ready", previous)
    second = m.payload(row((1, 2, 0), (4, 4, 1), (1, 1, 4)), 3, "q14_ready", {0: 4, 1: 1, 2: 1})
    assert first["convslots"][0] == second["convslots"][1]
    assert second["convslots"][0] == [5, 6, 7, 8]
    assert second["accepted"] == [1, 1, 4]
    with pytest.raises(ValueError, match="previous"):
        m.payload(row(accepted=(4, 2)), 3, "q14_ready", {2: 4, 0: 1})
    with pytest.raises(ValueError, match="owner"):
        m.payload(row(owners=(3, 0)), 3, "q14_ready")
    with pytest.raises(ValueError, match="mode"):
        m.payload(row(), 3, "native")


def test_original_frozen_segment_keeps_224_dense_rows_and_inverse_prefixes():
    m = module()
    manifest = json.loads(
        (ROOT / "benchmarks/results/multi_graph_structural/gdn-compact-q14-v1/manifest.json").read_bytes()
    )
    frozen = manifest["payload"]
    previous = {}
    kinds = {}
    for r in frozen["steps"]:
        p = m.payload(r, frozen["capacity_requests"], "q14_ready", previous)
        kinds[p["segment"]] = kinds.get(p["segment"], 0) + 1
        assert p["buffer_rows"] == 224
        assert len(p["pack_indices"]) == len(p["inverse_indices"]) == 224
        for packed, dense in enumerate(p["pack_indices"][: p["executed_query_tokens"]]):
            assert p["inverse_indices"][dense] == packed
        previous.update(zip(r["request_ids"], r["query_lengths"], strict=True))
    assert kinds == {"mixed": 6, "q1_only": 10}


def test_screen_uses_complete_samples_and_frozen_independent_process_order():
    m = module()
    assert m.PROCESSES == 2 and m.REPEATS == 8 and m.WARMUP == 2
    assert m.process_order(0) == ("forced_q4", "q14_ready")
    assert m.process_order(1) == ("q14_ready", "forced_q4")
    result = m.summarize(samples())
    assert result["stable"]
    assert result["sample_count"] == 32
    assert result["absolute_saving_ms"] == pytest.approx(1.0)
    assert result["noise_threshold_percent"] < result["improvement_percent"] < 1.0
    assert all(p["improvement_percent"] > 0 for p in result["processes"])


def test_screen_rejects_noise_and_a_losing_independent_process():
    m = module()
    assert not m.summarize(samples(baseline=(100, 102), candidate=(99.5, 101.5)))["stable"]
    assert not m.summarize(samples(baseline=(100, 100), candidate=(98, 100.01)))["stable"]


def test_screen_rejects_missing_duplicate_and_incomplete_lifecycle_samples():
    m = module()
    raw = samples()
    with pytest.raises(ValueError, match="32"):
        m.summarize(raw[:-1])
    raw[-1] = raw[-2].copy()
    with pytest.raises(ValueError, match="identity"):
        m.summarize(raw)
    raw = samples()
    raw[0]["lifecycle_ms"] += 1.0
    with pytest.raises(ValueError, match="lifecycle"):
        m.summarize(raw)
    raw = samples()
    raw[0]["execution_ms"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        m.summarize(raw)
