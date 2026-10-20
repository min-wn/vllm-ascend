# SPDX-License-Identifier: Apache-2.0
import importlib.util
import sys
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[4] / "benchmarks" / "multi_graph_gdn_ready_range_protocol.py"
SPEC = importlib.util.spec_from_file_location("gdn_ready_range_protocol", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_load_ranges_are_frozen_and_disjoint():
    profiles = MODULE.frozen_profiles()
    ranges = [(min(row["live"] for row in p["trace"]), max(row["live"] for row in p["trace"])) for p in profiles]
    assert ranges == [(22, 26), (54, 60), (94, 100)]
    assert MODULE.protocol() == MODULE.protocol()


def test_acceptance_and_lengths_follow_owners_across_reordering_and_pause():
    for profile in MODULE.frozen_profiles():
        generation, lengths, previous = [0] * 100, [MODULE.PREFIX_TOKENS] * 100, set()
        resumed, reordered = set(), False
        for row in profile["trace"]:
            owners = row["request_ids"]
            assert len(set(owners)) == row["live"]
            reordered |= owners != sorted(owners)
            for owner, count, length in zip(owners, row["accepted"], row["seq_lengths"], strict=True):
                assert count == 1 + (owner + generation[owner]) % 4
                if generation[owner]:
                    lengths[owner] += count
                    if owner not in previous:
                        resumed.add(owner)
                assert length == lengths[owner] + 4
                generation[owner] += 1
            previous = set(owners)
        assert reordered and resumed
        assert all(generation[owner] > 0 for owner in range(profile["pool"]))
        assert profile["terminal_counts"] == [1 + (owner + n) % 4 for owner, n in enumerate(generation)]
