# SPDX-License-Identifier: Apache-2.0
"""Reconcile FIA update waits in saved segment and complete-model traces.

Only same-trace intervals are compared. This does not measure unprofiled gain,
identify event memory addresses or reconstruct every historical runtime binary.
"""

import hashlib
import json
from decimal import Decimal
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
RESULTS = HERE.parent
OUTSIDE_MODEL = 4294967295
HOST_FIA = "npu::npu_fused_infer_attention_score"
DEVICE_FIA = "FusedInferAttentionScore"


def read(path):
    return json.loads(path.read_bytes())


def raw_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def start(event):
    return Decimal(str(event["ts"]))


def stop(event):
    return start(event) + Decimal(str(event.get("dur", 0)))


def union_length(intervals):
    """Endpoint scan; verify.py uses a separate sort-and-merge implementation."""
    endpoints = {}
    for begin, end in intervals:
        if end > begin:
            endpoints[begin] = endpoints.get(begin, 0) + 1
            endpoints[end] = endpoints.get(end, 0) - 1
    active, total, previous = 0, Decimal(0), None
    for point, change in sorted(endpoints.items()):
        if previous is not None and active:
            total += point - previous
        active += change
        previous = point
    assert active == 0
    return total


def clipped_intervals(events, begin, end):
    return [
        (max(begin, start(event)), min(end, stop(event)))
        for event in events
        if start(event) < end and stop(event) > begin
    ]


def profile_inventory(directory, rank):
    filename = "probe" if directory.name == "two-layer-fused-v1" else "diagnostic"
    path = directory / "results" / f"{filename}.rank{rank}.json"
    return path, read(path)


def inspect_trace(directory, rank, descriptor, scope):
    filename = descriptor.get("file") or Path(descriptor["path"]).name
    path = directory / "results" / filename
    assert raw_hash(path) == descriptor["sha256"], path
    document = json.loads(path.read_bytes(), parse_float=Decimal)
    events = document["traceEvents"] if isinstance(document, dict) else document
    complete = [event for event in events if event.get("ph") == "X"]
    ai = [
        event
        for event in complete
        if str(event.get("args", {}).get("Task Type", "")).startswith("KERNEL_")
        and event["args"]["Task Type"] != "KERNEL_AICPU"
    ]
    device = sorted((event for event in ai if event["name"] == DEVICE_FIA), key=start)
    host = sorted(
        (event for event in complete if event["name"] == HOST_FIA and event.get("cat") == "cpu_op"), key=start
    )
    external_writes = sorted(
        (
            event
            for event in complete
            if event["name"] == "MEM_WRITE_VALUE" and event.get("args", {}).get("Model Id") == OUTSIDE_MODEL
        ),
        key=start,
    )
    expected = 1 if scope == "two_layers" else 16
    assert len(device) == len(host) == len(external_writes) == expected, path
    waits = [event for event in complete if event["name"] == "MEM_WAIT_VALUE"]
    rows = []
    # The archived caller resolves tasks in captured layer order. Index pairing
    # is an order-based association, not matching hidden event memory addresses.
    for index, (fia, cpu) in enumerate(zip(device, host, strict=True)):
        predecessor = max(
            (event for event in ai if event.get("tid") == fia.get("tid") and stop(event) <= start(fia)),
            key=stop,
        )
        assert "ScatterPaKvCache" in predecessor["name"], (path, index, predecessor["name"])
        begin, end = stop(predecessor), start(fia)
        stream_waits = [event for event in waits if event.get("tid") == fia.get("tid")]
        clipped = clipped_intervals(stream_waits, begin, end)
        wait_union = union_length(clipped)
        assert wait_union <= end - begin
        rows.append(
            dict(
                fia_index=index,
                stream_gap_us=float(end - begin),
                polling_interval_count=len(clipped),
                polling_union_us=float(wait_union),
                host_fia_call_us=float(stop(cpu) - start(cpu)),
                host_call_overlap_with_gap_us=float(max(Decimal(0), min(end, stop(cpu)) - max(begin, start(cpu)))),
                host_return_before_device_fia_us=float(end - stop(cpu)),
            )
        )
    first = start(device[0])
    return dict(
        file=str(path.relative_to(ROOT)).replace("\\", "/"),
        sha256=raw_hash(path),
        scope=scope,
        mode=descriptor["mode"],
        rank=rank,
        step=descriptor.get("step"),
        batch=descriptor.get("batch", 98),
        capacity=descriptor.get("capacity", 400),
        fia_count=len(device),
        stream_gap_sum_us=sum(row["stream_gap_us"] for row in rows),
        stream_gap_max_us=max(row["stream_gap_us"] for row in rows),
        polling_union_sum_us=sum(row["polling_union_us"] for row in rows),
        last_host_fia_return_before_first_device_fia_us=float(first - max(map(stop, host))),
        last_outside_model_write_before_first_device_fia_us=float(first - max(map(stop, external_writes))),
        all_outside_model_writes_before_first_device_fia=all(stop(event) <= first for event in external_writes),
        per_fia=rows,
    )


def source_inventory(directory):
    launch_path = directory / "launch.json"
    launch = read(launch_path)
    checked = []
    for remote_path, declared in launch["sources"].items():
        source = directory / "sources" / Path(remote_path).name
        if source.exists():
            assert raw_hash(source) == declared, source
            checked.append(dict(file=str(source.relative_to(ROOT)).replace("\\", "/"), sha256=declared))
    caller = (
        "aclgraph_bounded_lead_probe.py"
        if directory.name == "two-layer-fused-v1"
        else "multi_graph_gdn_q4_full_probe.py"
    )
    assert any(Path(entry["file"]).name == caller for entry in checked)
    return dict(
        launch_file=str(launch_path.relative_to(ROOT)).replace("\\", "/"),
        launch_sha256=raw_hash(launch_path),
        task_queue_enable=launch["environment"].get("TASK_QUEUE_ENABLE"),
        checked_archived_sources=checked,
        historical_runtime_binaries_reconstructed=False,
    )


def memory_record(directory, rank, report):
    phases = {entry["phase"]: entry for entry in report["memory"]}
    before, after = phases["compact_decode_transition"], phases["all_required_default_graphs_captured"]
    caller = directory / "sources/multi_graph_gdn_q4_full_probe.py"
    source = caller.read_text(encoding="utf-8")
    # Qualified archived source passes the same pool/root stream to each
    # mutually exclusive graph capture. This is not proof of parallel pool use.
    for anchor in ("pool = torch.npu.graph_pool_handle()", "pool=pool, stream=root", "All graphs execute exclusively"):
        assert anchor in source
    return dict(
        run=directory.name,
        rank=rank,
        graph_count=report["capture"]["graphs"],
        buckets=report["capture"]["required_default_buckets"],
        archived_caller_sha256=raw_hash(caller),
        same_pool_requested_for_exclusive_captures=True,
        before_allocated_bytes=before["allocated_bytes"],
        after_allocated_bytes=after["allocated_bytes"],
        allocated_phase_delta_bytes=after["allocated_bytes"] - before["allocated_bytes"],
        reserved_phase_delta_bytes=after["reserved_bytes"] - before["reserved_bytes"],
        phase_delta_is_pure_graph_memory=False,
        concurrent_pool_aliasing_qualified=False,
    )


def build():
    traces, provenance, memory, reports = [], [], [], []
    for name in ("two-layer-fused-v1", "full-core-join-v3", "full-core-join-v4-profile"):
        directory = RESULTS / name
        provenance.append(source_inventory(directory))
        scope = "two_layers" if name == "two-layer-fused-v1" else "all_64_decoders_and_final_norm"
        modes = (
            {"fused_padding", "fused_split_barrier", "fused_split_early"}
            if scope == "two_layers"
            else {"padding", "split_barrier", "split_early"}
        )
        for rank in (0, 1):
            report_path, report = profile_inventory(directory, rank)
            reports.append(
                dict(file=str(report_path.relative_to(ROOT)).replace("\\", "/"), sha256=raw_hash(report_path))
            )
            for descriptor in report["profiles"]:
                if descriptor["mode"] in modes:
                    traces.append(inspect_trace(directory, rank, descriptor, scope))
            if scope != "two_layers":
                memory.append(memory_record(directory, rank, report))
    assert len(traces) == 18
    local = [entry for entry in traces if entry["scope"] == "two_layers"]
    full = [entry for entry in traces if entry["scope"] != "two_layers"]
    assert len(local) == 6 and len(full) == 12
    return dict(
        scope="offline FIA wait and graph-resource attribution; no NPU execution or new speedup result",
        graph_execution=(
            "one composite graph per mode/step; modes replayed separately; not independent multigraph parallelism"
        ),
        units="microseconds; only within-trace times compared; no cross-rank timestamp subtraction",
        provenance=provenance,
        original_reports=reports,
        traces=traces,
        memory_phase_records=memory,
        findings=dict(
            local_gap_range_us=[
                min(entry["stream_gap_sum_us"] for entry in local),
                max(entry["stream_gap_sum_us"] for entry in local),
            ],
            local_polling_coverage_range=[
                min(entry["polling_union_sum_us"] / entry["stream_gap_sum_us"] for entry in local),
                max(entry["polling_union_sum_us"] / entry["stream_gap_sum_us"] for entry in local),
            ],
            full_sum_of_16_gaps_range_us=[
                min(entry["stream_gap_sum_us"] for entry in full),
                max(entry["stream_gap_sum_us"] for entry in full),
            ],
            full_maximum_single_gap_us=max(entry["stream_gap_max_us"] for entry in full),
            full_16_outside_writes_all_complete_before_first_fia=all(
                entry["all_outside_model_writes_before_first_device_fia"] for entry in full
            ),
            full_last_write_lead_range_us=[
                min(entry["last_outside_model_write_before_first_device_fia_us"] for entry in full),
                max(entry["last_outside_model_write_before_first_device_fia_us"] for entry in full),
            ],
        ),
        limitations=[
            "Observed polling/CPU overlap supports update-supply attribution, not a counterfactual speedup.",
            "No external-event target address is exported; write/task correspondence is count/order/source based.",
            "Profiler changes host overhead and rank timing; these are not unprofiled endpoints.",
            "Full-model evidence covers B98 and B73 here, not every step or future fused full-model version.",
            "A phase allocation delta is not pure graph memory, runtime bytes or proof of concurrent pool safety.",
        ],
    )


if __name__ == "__main__":
    result = build()
    (HERE / "audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["findings"]))
