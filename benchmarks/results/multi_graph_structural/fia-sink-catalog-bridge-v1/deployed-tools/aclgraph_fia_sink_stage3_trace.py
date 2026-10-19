# SPDX-License-Identifier: Apache-2.0
"""CPU schedule for the unchanged Q4 trace and its recorded default buckets.

All physical request rows, including padding rows, have four query tokens.
Padding KV lengths are zero, matching the established full-model FIA control.
This file never imports Torch or vLLM and does not create a new capture size.
"""

import argparse
import bisect
import hashlib
import json
from pathlib import Path

from multi_graph_gdn_q4_full_metadata import (
    BLOCK_SIZE,
    FIXED_TRACE_SHA256,
    QUERY,
    decode_values,
    load_fixed_trace,
    reserve_owner_pages,
)

OWNER_COUNT = 128
STEP_COUNT = 128
RESULTS_DIRECTORY = Path(__file__).parent / "results" / "multi_graph_structural"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _capture_evidence(path, digest, steps):
    evidence = json.loads(Path(path).read_text(encoding="utf-8"))
    if evidence.get("trace_sha256") != digest:
        raise ValueError("default capture evidence belongs to another trace")
    selection = evidence["selection"]
    sizes = tuple(selection["capture_sizes"])
    dispatched = tuple(selection["dispatched_tokens"])
    if (
        not sizes
        or any(isinstance(size, bool) or not isinstance(size, int) or size < 1 for size in sizes)
        or tuple(sorted(set(sizes))) != sizes
        or len(dispatched) != steps
        or selection.get("query") != QUERY
        or selection.get("config_source") != "installed EngineArgs defaults; no capture sizes supplied"
    ):
        raise ValueError("expected the recorded unmodified installed default capture catalog")
    return sizes, dispatched, selection["config_source"]


def _check_lifecycle(step, prefix, pending, seen):
    requests = tuple(step["request_ids"])
    first = tuple(request not in seen for request in requests)
    if tuple(step["first_use"]) != first:
        raise ValueError("trace first-use markers disagree with stable owner lifetimes")
    if tuple(step["accepted_from_previous"]) != tuple(pending[request] for request in requests):
        raise ValueError("pending accepted prefix changed while an owner was paused")
    expected = []
    for request in requests:
        if request in seen:
            prefix[request] += pending[request]
        expected.append(prefix[request] + QUERY)
    if tuple(step["seq_lengths"]) != tuple(expected):
        raise ValueError("exact sequence lengths do not follow the declared owner history")
    accepted_after = tuple(step["accepted_after_step"])
    if len(accepted_after) != len(requests) or any(
        isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= QUERY for count in accepted_after
    ):
        raise ValueError("each live owner requires an accepted prefix in 1..4")
    for request, accepted in zip(requests, accepted_after, strict=True):
        pending[request] = accepted


def _frame(step, capacity, pages, previous, seen):
    values = decode_values(step, capacity, pages)
    requests = tuple(step["request_ids"])
    current = set(requests)
    physical_batch = capacity // QUERY
    padding = physical_batch - len(requests)
    return dict(
        step=step["step"],
        batch=len(requests),
        live_tokens=len(requests) * QUERY,
        physical_tokens=capacity,
        default_capacity=capacity,
        physical_batch=physical_batch,
        padding_requests=padding,
        request_ids=requests,
        owner_keys=tuple((request, 0) for request in requests) + (None,) * padding,
        owner_active=tuple(request in current for request in range(OWNER_COUNT)),
        q_lengths=(QUERY,) * physical_batch,
        actual_seq_qlen=tuple(values["fia_boundaries"][1:]),
        kv_lengths=tuple(values["lengths"]),
        block_table=tuple(tuple(row) for row in values["pages"]),
        positions=tuple(values["positions"]),
        slot_mapping=tuple(values["slots"]),
        active_tokens=tuple(values["active"]),
        first_use=tuple(step["first_use"]),
        accepted_from_previous=tuple(step["accepted_from_previous"]),
        accepted_after_step=tuple(step["accepted_after_step"]),
        admitted=tuple(request for request in requests if request not in seen),
        resumed=tuple(request for request in requests if request in seen and request not in previous),
        departed=tuple(sorted(previous - current)),
        retired=(),
        input_seed=step["input_seed"],
    )


def load_trace_schedule(trace_path=None, selection_path=None):
    """Return exact CPU frames plus trace, capture and source provenance.

    ``records`` contains plain dictionaries with immutable tuple vectors.
    ``q_lengths`` is per-request, while ``actual_seq_qlen`` is cumulative
    native TND Q length. Both include physical dummy requests with Q4/KV0.
    Owners retain generation zero throughout this trace: absence is a pause,
    and there is no retirement or request-ID recycling in this workload.

    The archived selection is checked, not recomputed from Python constants.
    A device runner must additionally verify it against the currently installed
    public dispatcher before calling the result a current-default experiment.
    """
    trace_path = Path(trace_path or RESULTS_DIRECTORY / "structural-dynamic-trace-v1.json")
    selection_path = Path(selection_path or RESULTS_DIRECTORY / "q4-tile-full-replay-v8-queue0.rank0.json")
    trace = load_fixed_trace(trace_path)
    if len(trace.steps) != STEP_COUNT or len(trace.initial_prefix_by_request) != OWNER_COUNT:
        raise ValueError("expected the original 128-step trajectory and 128 stable owners")
    sizes, dispatched, source = _capture_evidence(selection_path, trace.digest, len(trace.steps))
    pages = reserve_owner_pages(trace)
    prefix, pending = list(trace.initial_prefix_by_request), [1] * OWNER_COUNT
    previous, seen, records = set(), set(), []
    for index, step in enumerate(trace.steps):
        if step["step"] != index:
            raise ValueError("trace step indices must remain unchanged and ordered")
        _check_lifecycle(step, prefix, pending, seen)
        offset = bisect.bisect_left(sizes, step["batch"] * QUERY)
        if offset == len(sizes) or sizes[offset] % QUERY or dispatched[index] != sizes[offset]:
            raise ValueError("recorded default dispatch differs from the smallest legal default bucket")
        records.append(_frame(step, sizes[offset], pages, previous, seen))
        previous = set(step["request_ids"])
        seen.update(previous)
    return dict(
        records=tuple(records),
        canonical_digest=trace.digest,
        trace_sha256=trace.digest,
        trace_file_sha256=sha256(trace_path),
        selection_sha256=sha256(selection_path),
        metadata_source_sha256=sha256(Path(__file__).with_name("multi_graph_gdn_q4_full_metadata.py")),
        catalog_resolver_source_sha256=sha256(Path(__file__).with_name("multi_graph_gdn_q4_full_loader.py")),
        default_capture_sizes=sizes,
        required_capture_sizes=tuple(sorted({record["physical_tokens"] for record in records})),
        catalog_source=source,
        installed_catalog_revalidated=False,
        query=QUERY,
        page_size=BLOCK_SIZE,
        owner_count=OWNER_COUNT,
        owner_pages=dict(
            pages=pages.pages,
            maximum_lengths=pages.maximum_lengths,
            page_count=pages.page_count,
            width=pages.width,
        ),
    )


def cpu_check(schedule):
    """Audit physical padding and exact native length vectors independently."""
    if schedule["canonical_digest"] != FIXED_TRACE_SHA256:
        raise AssertionError("canonical trace digest changed")
    for frame in schedule["records"]:
        physical, live = frame["physical_batch"], frame["batch"]
        if frame["q_lengths"] != (QUERY,) * physical or frame["actual_seq_qlen"] != tuple(
            QUERY * (index + 1) for index in range(physical)
        ):
            raise AssertionError("physical Q4 TND boundaries changed")
        if frame["actual_seq_qlen"][-1] != frame["physical_tokens"]:
            raise AssertionError("native TND Q lengths do not cover the physical query tensor")
        if any(frame["kv_lengths"][live:]) or any(frame["owner_keys"][live:]):
            raise AssertionError("padding requests must be ownerless with zero KV")
        if any(frame["active_tokens"][live * QUERY :]) or any(
            slot != -1 for slot in frame["slot_mapping"][live * QUERY :]
        ):
            raise AssertionError("physical padding must not mutate owner state")
        for row, request in enumerate(frame["request_ids"]):
            if frame["owner_keys"][row] != (request, 0) or frame["kv_lengths"][row] < QUERY:
                raise AssertionError("logical owner identity or exact KV length changed")
            if (
                tuple(frame["block_table"][row][: len(schedule["owner_pages"]["pages"][request])])
                != schedule["owner_pages"]["pages"][request]
            ):
                raise AssertionError("batch permutation changed the owner's fixed physical pages")
    first = schedule["records"][0]
    if (first["batch"], first["live_tokens"], first["physical_tokens"]) != (98, 392, 400):
        raise AssertionError("the predeclared B98 example must retain 392 live rows and default capacity 400")
    return dict(
        steps=len(schedule["records"]),
        owner_count=schedule["owner_count"],
        canonical_digest=schedule["canonical_digest"],
        selection_sha256=schedule["selection_sha256"],
        default_capture_sizes=schedule["default_capture_sizes"],
        required_capture_sizes=schedule["required_capture_sizes"],
        first_step=dict(batch=98, live_tokens=392, physical_tokens=400, dummy_q_length=4, dummy_kv_length=0),
        active_observations=sum(frame["batch"] for frame in schedule["records"]),
        native_work_executed=False,
        installed_catalog_revalidated=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    schedule = load_trace_schedule(args.trace, args.selection)
    report = cpu_check(schedule)
    if args.output:
        if args.output.exists():
            parser.error("refuse to overwrite existing schedule evidence")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(dict(schedule=schedule, check=report), indent=2) + "\n", encoding="ascii")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
