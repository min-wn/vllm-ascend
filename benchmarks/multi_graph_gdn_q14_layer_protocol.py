# SPDX-License-Identifier: Apache-2.0
"""Host-only contract for the bounded Q1/Q4 complete GDN-layer comparison."""

import math
import statistics

from multi_graph_gdn_q14_protocol import metadata, validate_contract

CONFIGS = ("forced_q4", "q14_ready")
PROCESSES = 2
REPEATS = 8
WARMUP = 2


def process_order(process):
    if type(process) is not int or not 0 <= process < PROCESSES:
        raise ValueError("exactly two independent process identities are allowed")
    return CONFIGS if process == 0 else CONFIGS[::-1]


def settings():
    return dict(
        schema="gdn-compact-q14-complete-layer-v1",
        processes=PROCESSES,
        samples_per_config_per_process=REPEATS,
        warmup_per_config=WARMUP,
        process_orders=[list(process_order(p)) for p in range(PROCESSES)],
        main_metric="initialization_ms + execution_ms + terminal_ms",
        primary_comparison="q14_ready versus forced_q4; same dense shapes and graph capacity",
        screen=dict(
            both_process_improvements_positive=True,
            pooled_median_improvement_gt_noise=True,
            noise="2 * max(relative span of process medians, pooled relative sample MAD), in percent",
            fixed_percent_floor=None,
        ),
        segment_measurement="separate event-endpoint sums for q1_only and mixed; not the admission metric",
        input_origin="new teacher-forced text layer inputs; not the original generation ledger activations",
        scope="one complete GDN decoder layer; no runner, generation, service, or quality qualification",
        historical_failures_preserved=True,
    )


def payload(row, capacity, mode, previous_widths=None):
    """Keep dense quartets while packing only each current effective prefix.

    `inverse_indices` uses an extra zero row at `buffer_rows` for every
    unexecuted dense row. Consumers must append that row before gathering back;
    selecting arbitrary unwritten packed tails would expose undefined values.
    Previous acceptance indexes the previous candidate history independently of
    the current width, including accepted=4 followed by current Q1.
    """
    if mode not in (*CONFIGS, "q14_delayed"):
        raise ValueError("unknown complete-layer mode")
    owners = row["request_ids"]
    actual = row["query_lengths"]
    accepted = row["accepted_from_previous"]
    validate_contract(owners, actual, accepted, previous_widths)
    if type(capacity) is not int or not len(owners) <= capacity <= 128:
        raise ValueError("capacity must cover live owners within 1..128")
    if any(owner >= capacity for owner in owners):
        raise ValueError("owner must have a stable state slot within capacity")
    lengths = [4] * len(owners) if mode == "forced_q4" else list(actual)
    result = metadata(owners, lengths, accepted, capacity)
    rows = 4 * capacity
    pack = [4 * row_id + token for row_id, q in enumerate(lengths) for token in range(q)]
    inverse = [rows] * rows
    for packed, dense in enumerate(pack):
        inverse[dense] = packed
    starts = [0]
    for q in [*lengths, *([0] * (capacity - len(owners)))]:
        starts.append(starts[-1] + q)
    valid = [False] * rows
    for row_id, q in enumerate(actual):
        valid[4 * row_id : 4 * row_id + q] = [True] * q
    result.update(
        convslots=[
            *[list(range(1 + 4 * owner, 5 + 4 * owner)) for owner in owners],
            *([[0, 0, 0, 0]] * (capacity - len(owners))),
        ],
        query_start_loc=starts,
        pack_indices=[*pack, *([0] * (rows - len(pack)))],
        inverse_indices=inverse,
        valid_dense_mask=valid,
        actual_query_tokens=sum(actual),
        executed_query_tokens=sum(lengths),
        zero_sentinel_row=rows,
        segment="q1_only" if all(q == 1 for q in actual) else "q4_only" if all(q == 4 for q in actual) else "mixed",
    )
    return result


def summarize(samples):
    """Apply the predeclared screen to all 32 complete lifecycle samples."""
    expected = {(p, c, s) for p in range(PROCESSES) for c in CONFIGS for s in range(REPEATS)}
    if len(samples) != len(expected):
        raise ValueError("exactly 32 complete lifecycle samples are required")
    identities = [(s["process"], s["config"], s["sample"]) for s in samples]
    if len(set(identities)) != len(expected) or set(identities) != expected:
        raise ValueError("sample identity missing, duplicated, or outside the frozen experiment")
    values = {(p, c): [] for p in range(PROCESSES) for c in CONFIGS}
    for sample in samples:
        fields = [sample[k] for k in ("initialization_ms", "execution_ms", "terminal_ms", "lifecycle_ms")]
        if any(not isinstance(x, (int, float)) or not math.isfinite(x) or x < 0 for x in fields):
            raise ValueError("finite nonnegative lifecycle components are required")
        if fields[-1] <= 0 or not math.isclose(sum(fields[:-1]), fields[-1], rel_tol=1e-9, abs_tol=1e-6):
            raise ValueError("lifecycle must include initialization, execution, and terminal costs")
        values[sample["process"], sample["config"]].append(fields[-1])
    processes = []
    for process in range(PROCESSES):
        medians = {config: statistics.median(values[process, config]) for config in CONFIGS}
        baseline, candidate = (medians[c] for c in CONFIGS)
        processes.append(
            dict(
                process=process,
                order=list(process_order(process)),
                median_lifecycle_ms=medians,
                absolute_saving_ms=baseline - candidate,
                improvement_percent=100 * (baseline - candidate) / baseline,
            )
        )
    pooled = {config: [x for p in range(PROCESSES) for x in values[p, config]] for config in CONFIGS}
    medians = {config: statistics.median(pooled[config]) for config in CONFIGS}
    spans, mads = {}, {}
    for config in CONFIGS:
        endpoints = [p["median_lifecycle_ms"][config] for p in processes]
        spans[config] = 100 * (max(endpoints) - min(endpoints)) / statistics.median(endpoints)
        mads[config] = 100 * statistics.median(abs(x - medians[config]) for x in pooled[config]) / medians[config]
    baseline, candidate = (medians[c] for c in CONFIGS)
    improvement = 100 * (baseline - candidate) / baseline
    noise = 2 * max(*spans.values(), *mads.values())
    return dict(
        sample_count=len(samples),
        processes=processes,
        pooled_median_lifecycle_ms=medians,
        absolute_saving_ms=baseline - candidate,
        improvement_percent=improvement,
        relative_process_span_percent=spans,
        relative_sample_mad_percent=mads,
        noise_threshold_percent=noise,
        stable=all(p["improvement_percent"] > 0 for p in processes) and improvement > noise,
        scope=settings()["scope"],
    )
