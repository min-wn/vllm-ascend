# SPDX-License-Identifier: Apache-2.0
"""Independent CPU coverage and balancing contracts for no-FD metadata."""

import importlib.util
import itertools
import random
import time
from collections import Counter
from pathlib import Path

import pytest


@pytest.fixture
def planner(monkeypatch):
    root = Path(__file__).resolve().parents[4] / "benchmarks"
    monkeypatch.syspath_prepend(str(root))
    spec = importlib.util.spec_from_file_location(
        "whole_plan_stage4_cpu", root / "aclgraph_fia_sink_stage4_whole_plan.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def record(lengths):
    width = (max(lengths, default=0) + 127) // 128 or 1
    batch = len(lengths)
    return dict(
        physical_tokens=4 * batch,
        physical_batch=batch,
        q_lengths=[4] * batch,
        actual_seq_qlen=list(range(4, 4 * batch + 1, 4)),
        kv_lengths=list(lengths),
        block_table=[[row * width + page for page in range(width)] for row in range(batch)],
        page_count=batch * width,
    )


def template(module, row, *, poison=12345):
    words = [poison] * module.METADATA_WORDS
    maximum = max(row["kv_lengths"]) or 1024
    words[module.BASE_OFFSET : module.BASE_OFFSET + 10] = [
        512,
        min(512, maximum),
        8,
        12,
        20,
        36,
        4,
        maximum,
        row["physical_batch"],
        row["physical_batch"],
    ]
    return words


def independent_coverage(row, words):
    base = words[576:586]
    seen = Counter()
    physical_heads = Counter()
    start = 0
    for core in range(base[4]):
        end = words[10 * core + 1]
        assert words[10 * core : 10 * core + 10] == [1, end, *([0] * 8)]
        assert start < end <= 2 * len(row["kv_lengths"])
        for head in range(start, end):
            request, kv_head = divmod(head, 2)
            physical_heads[request, kv_head] += 1
            for query_token in range(4):
                causal_end = max(0, row["kv_lengths"][request] - 4 + query_token + 1)
                for kv_token in range(causal_end):
                    seen[request, kv_head, query_token, kv_token] += 1
        start = end
    assert start == len(row["kv_lengths"]) * 2
    expected = Counter(
        (request, head, query, token)
        for request, length in enumerate(row["kv_lengths"])
        for head in range(2)
        for query in range(4)
        for token in range(max(0, length - 4 + query + 1))
    )
    assert seen == expected
    assert physical_heads == Counter((request, head) for request in range(len(row["kv_lengths"])) for head in range(2))


@pytest.mark.parametrize(
    "lengths",
    [
        [4],
        [127, 128, 129],
        [511, 512, 513],
        [4097],
        [129, 0, 513, 0],
        [0, 129, 0, 513],
        [0, 0, 0, 0],
        [513, 127, 129, 0],
    ],
)
def test_complete_head_assignment_preserves_exact_attention_domain(planner, lengths):
    row = record(lengths)
    before = template(planner, row)
    saved = before.copy()
    words = planner.whole_plan(row, before)
    assert before == saved
    assert len(words) == 1024
    assert words[579] == words[581] == 0
    assert words[580] == min(20, 2 * sum(length > 0 for length in lengths)) or not any(lengths)
    independent_coverage(row, words)
    summary = planner.validate_whole_plan(row, words)
    assert summary["physical_heads"] == 2 * len(lengths)
    assert summary["cost_model"] == planner.COST_MODEL
    assert words[586:] == [0] * (1024 - 586)


def test_original_trace_keeps_every_default_bucket_and_zero_padding(planner):
    from aclgraph_fia_sink_stage3_trace import load_trace_schedule

    schedule = load_trace_schedule()
    visited = set()
    for row in schedule["records"]:
        words = planner.whole_plan(row, template(planner, row))
        summary = planner.validate_whole_plan(row, words)
        assert summary["used_cores"] == 20
        assert words[581] == words[579] == 0
        assert words[10 * 19 + 1] == row["physical_batch"] * 2
        assert words[582:586] == [4, max(row["kv_lengths"]), row["physical_batch"], row["physical_batch"]]
        assert not any(row["kv_lengths"][row["batch"] :])
        visited.add(row["physical_tokens"])
    assert visited == set(schedule["required_capture_sizes"])
    assert len(visited) == 17


def test_fast_builder_matches_strict_plan_and_has_a_timing_boundary(planner):
    row = record([129, 0, 513, 0])
    qualified = template(planner, row)
    strict = planner.whole_plan(row, qualified)
    preflight = planner.preflight_whole_plan(row, qualified)
    start = time.perf_counter_ns()
    fast = planner.whole_plan_fast(row, qualified, preflight=preflight)
    elapsed_us = (time.perf_counter_ns() - start) / 1000
    assert fast == strict
    assert elapsed_us >= 0
    assert planner.validate_whole_plan(row, fast)["cost_model"] == planner.COST_MODEL


def test_fast_builder_cannot_bypass_preflight_identity(planner):
    row = record([129, 0, 513, 0])
    qualified = template(planner, row)
    preflight = planner.preflight_whole_plan(row, qualified)
    with pytest.raises(ValueError, match="different stable"):
        planner.whole_plan_fast(dict(row), qualified, preflight=preflight)
    with pytest.raises(ValueError, match="different stable"):
        planner.whole_plan_fast(row, qualified.copy(), preflight=preflight)
    with pytest.raises(ValueError, match="requires strict preflight"):
        planner.whole_plan_fast(row, qualified)


def test_contiguous_balance_matches_bruteforce_minimax(planner):
    generator = random.Random(1703)
    for count in range(1, 9):
        for _ in range(15):
            costs = [generator.randrange(7) for _ in range(count)]
            if not any(costs):
                costs[0] = 1
            for cores in range(1, min(4, sum(value > 0 for value in costs)) + 1):
                cuts = planner._cuts(costs, cores)
                previous = 0
                loads = []
                for end in cuts:
                    loads.append(sum(costs[previous:end]))
                    previous = end
                candidates = (
                    max(sum(costs[a:b]) for a, b in zip((0, *interior), (*interior, count), strict=True))
                    for interior in itertools.combinations(range(1, count), cores - 1)
                )
                assert max(loads) == min(candidates)
                assert len(loads) == cores and min(loads) > 0


@pytest.mark.parametrize("word", [0, 2, 3, 4, 7, 360, 361, 579, 580, 581, 583, 584, 900])
def test_coverage_validator_rejects_corrupted_metadata(planner, word):
    row = record([129, 0, 513])
    words = planner.whole_plan(row, template(planner, row))
    words[word] += 1
    with pytest.raises(ValueError):
        planner.validate_whole_plan(row, words)


@pytest.mark.parametrize("cores", [0, 21, True, 1.5])
def test_unsupported_core_counts_are_rejected(planner, cores):
    row = record([129])
    with pytest.raises(ValueError):
        planner.whole_plan(row, template(planner, row), cores=cores)


def test_template_geometry_and_physical_descriptor_changes_are_rejected(planner):
    row = record([129, 513])
    original = template(planner, row)
    for index, value in ((576, 256), (577, 256), (578, 16), (582, 8), (584, 0), (585, 99)):
        changed = original.copy()
        changed[index] = value
        with pytest.raises(ValueError):
            planner.whole_plan(row, changed)
    for changed in (
        {**row, "actual_seq_qlen": [4, 7]},
        {**row, "kv_lengths": [129, False]},
        {**row, "q_lengths": [4, 0]},
        {**row, "physical_tokens": 9},
        {**row, "query_heads": 24},
        {**row, "sparse_mode": 0},
        {**row, "kv_lengths": [4098, 0]},
        {**row, "block_table": [[0], [1]]},
        {**row, "block_table": [[0, 1], [2]], "page_count": 4},
        {**row, "block_table": [[0, True], [2, 3]], "page_count": 4},
        {**row, "block_table": [[0, 1], [2, 4]], "page_count": 4},
        {**row, "block_table": [[0, 1], [2, 3]], "page_count": True},
    ):
        with pytest.raises(ValueError):
            planner.whole_plan(changed, original)


def test_repeated_plan_changes_do_not_retain_old_fd_state(planner):
    for lengths in ([129, 513, 0, 0], [0, 0, 0, 0], [4097, 0, 129, 0], [4, 0, 0, 0]):
        row = record(lengths)
        words = planner.whole_plan(row, template(planner, row))
        independent_coverage(row, words)
        assert all(words[10 * core + field] == 0 for core in range(36) for field in range(2, 10))
