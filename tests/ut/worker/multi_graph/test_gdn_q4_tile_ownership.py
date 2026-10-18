# SPDX-License-Identifier: Apache-2.0
"""Publication ownership oracle, including ranges smaller than the AIC grid.

The native replay qualifier checks these assignments by comparing every head's
output and staged state. This CPU oracle additionally exhausts grid/range tails
and verifies the two physical AIVs have equal work for every publication group.
"""

from collections import Counter
from pathlib import Path

import pytest


def publication_ownership(batch, cores, start=0, stop=None, inactive=()):
    if stop is None:
        stop = batch
    if not 1 <= batch <= 128 or not 1 <= cores <= 20 or not 0 <= start < stop <= batch:
        raise ValueError("invalid Q4 publication extent")
    inactive = set(inactive)
    records = []
    for core in range(cores):
        for group in range(start * 4 + core, stop * 4, cores):
            row, within_request = divmod(group, 4)
            keys = (within_request * 2, within_request * 2 + 1)
            heads = tuple(
                tuple(within_request * 6 + offset for offset in range(sub, 6, 2)) if row not in inactive else ()
                for sub in range(2)
            )
            records.append((core, group, row, keys, heads))
    return records


def test_native_publication_constants_match_ownership_oracle():
    source = Path(__file__).resolve().parents[4] / "benchmarks/multi_graph_gdn_q4_tile_pipeline_kernel"
    native = (source / "tile_pipeline_kernel.cpp").read_text(encoding="utf-8")
    assert "constexpr uint32_t kKeysPerGroup = 2;" in native
    assert "constexpr uint32_t kGroupsPerRequest = kKeyHeads / kKeysPerGroup;" in native
    assert "constexpr uint32_t kHeadsPerGroup = 3 * kKeysPerGroup;" in native
    assert "headOffset < kHeadsPerGroup; headOffset += 2" in native


@pytest.mark.parametrize("batch", [1, 3, 7, 20, 98, 100, 128])
@pytest.mark.parametrize("cores", [1, 2, 3, 7, 20])
def test_every_key_and_value_head_has_one_owner_with_equal_aiv_work(batch, cores):
    for start, stop in {(0, batch), (batch - 1, batch), (batch // 2, batch)}:
        records = publication_ownership(batch, cores, start, stop)
        key_visits, head_visits, group_visits = Counter(), Counter(), Counter()
        vector_work = Counter()
        for core, group, row, keys, heads in records:
            group_visits[group] += 1
            key_visits.update((row, key) for key in keys)
            for sub in range(2):
                assert len(heads[sub]) == 3
                head_visits.update((row, head) for head in heads[sub])
                vector_work[2 * core + sub] += len(heads[sub])
        assert key_visits == Counter((row, key) for row in range(start, stop) for key in range(8))
        assert head_visits == Counter((row, head) for row in range(start, stop) for head in range(24))
        assert group_visits == Counter(range(start * 4, stop * 4))
        for core in range(cores):
            assert vector_work[2 * core] == vector_work[2 * core + 1]


def test_inactive_requests_keep_both_flag_waits_but_no_head_writes():
    active = publication_ownership(7, 20)
    inactive = publication_ownership(7, 20, inactive={0, 3, 6})
    assert [(core, group) for core, group, *_ in active] == [(core, group) for core, group, *_ in inactive]
    for _, _, row, _, heads in inactive:
        assert sum(map(len, heads)) == (0 if row in {0, 3, 6} else 6)


@pytest.mark.parametrize("args", [(0, 20), (129, 20), (100, 0), (100, 21), (100, 20, 50, 50)])
def test_invalid_publication_ranges_are_rejected(args):
    with pytest.raises(ValueError):
        publication_ownership(*args)
