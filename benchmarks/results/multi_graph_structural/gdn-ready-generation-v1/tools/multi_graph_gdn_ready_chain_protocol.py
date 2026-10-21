# SPDX-License-Identifier: Apache-2.0
"""Frozen full-model replay endpoints; no generation or load routing."""

import hashlib
import json
from types import MappingProxyType

from multi_graph_gdn_q4_full_metadata import FrozenTrace, load_fixed_trace
from multi_graph_gdn_ready_full_lifecycle import terminal_counts
from multi_graph_gdn_ready_range_protocol import PREFIX_TOKENS, frozen_profiles

MODES = ("delayed", "ready")
LOADS = ("original", "low", "medium")
PAIRS = 2


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def workload(path, load):
    original = load_fixed_trace(path)
    if load == "original":
        return (
            original,
            terminal_counts(128),
            dict(
                source="unchanged original Q4 128-step trace",
                resident_owners=128,
                active_pool=128,
                token_source="unchanged deterministic per-step random vocabulary tokens",
            ),
        )
    if load not in LOADS:
        raise ValueError("only original, frozen low and frozen medium are in scope")
    profile = next(p for p in frozen_profiles() if p["name"] == load)
    steps = tuple(
        MappingProxyType(
            dict(
                step=row["step"],
                batch=row["live"],
                request_ids=tuple(row["request_ids"]),
                seq_lengths=tuple(row["seq_lengths"]),
                accepted_from_previous=tuple(row["accepted"]),
                input_seed=original.seed + 10000 * (1 + LOADS.index(load)) + row["step"],
            )
        )
        for row in profile["trace"]
    )
    prefixes = (PREFIX_TOKENS,) * 128
    body = dict(steps=[dict(row) for row in steps], initial_prefix_by_request=prefixes, seed=original.seed)
    trace = FrozenTrace(steps, prefixes, original.seed, digest(body))
    terminal = tuple(profile["terminal_counts"]) + tuple(1 + owner % 4 for owner in range(100, 128))
    return (
        trace,
        terminal,
        dict(
            source="unchanged frozen 40-step single-layer range request/length/acceptance schedule",
            resident_owners=128,
            active_pool=profile["pool"],
            difference_from_old_range=(
                "64 real layers, original random-token fixture, default buckets instead of fixed 400"
            ),
            dormant_owners="original full-model 128-owner cache fixture; unused owners retain initialized state",
            token_source="deterministic per-step random tokens; not the prior single-layer text activation sequence",
        ),
    )


def run_plan():
    return [
        dict(load=load, pair=pair, order=index, mode=mode)
        for load in LOADS
        for pair in range(PAIRS)
        for index, mode in enumerate(MODES if pair % 2 == 0 else reversed(MODES))
    ]


def cost_columns(rank):
    keys = ("execution_us", "initialization_us", "terminal_us", "total_us")
    if any(not isinstance(rank[k], (int, float)) or not 0 < rank[k] < float("inf") for k in keys):
        raise ValueError("positive finite closed costs required")
    if abs(sum(rank[k] for k in keys[:-1]) - rank["total_us"]) > 0.01:
        raise ValueError("closed timing columns must be additive on one rank")
    return {k: rank[k] for k in keys}


def critical_cost(ranks):
    if len(ranks) != 2 or {r["rank"] for r in ranks} != {0, 1}:
        raise ValueError("exactly two TP rank endpoints required")
    for rank in ranks:
        cost_columns(rank)
    selected = max(ranks, key=lambda r: r["total_us"])
    return dict(rank=selected["rank"], **cost_columns(selected))
