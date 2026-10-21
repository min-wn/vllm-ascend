# SPDX-License-Identifier: Apache-2.0
"""Analyze one Q14 diagnostic trace without adding overlapping durations."""

import argparse
import json
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path


def merged(intervals):
    result = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(end, result[-1][1]))
        else:
            result.append((start, end))
    return result


def length(intervals):
    return sum(end - start for start, end in merged(intervals))


def clipped(intervals, window):
    return merged((max(start, window[0]), min(end, window[1])) for start, end in intervals)


def complement(intervals, window):
    result = []
    cursor = window[0]
    for start, end in clipped(intervals, window):
        if cursor < start:
            result.append((cursor, start))
        cursor = end
    if cursor < window[1]:
        result.append((cursor, window[1]))
    return result


def intersection(first, second):
    return length((max(a, c), min(b, d)) for a, b in first for c, d in second)


def span(event):
    return event["start"], event["end"]


def contains(event, timestamp):
    return event["start"] <= timestamp <= event["end"]


def analyze(trace_path, diagnostic_path):
    raw = json.loads(trace_path.read_text())
    events = raw if isinstance(raw, list) else raw["traceEvents"]
    origin = min(Decimal(str(event["ts"])) for event in events if "ts" in event)
    for event in events:
        if "ts" in event:
            event["start"] = float(Decimal(str(event["ts"])) - origin)
            event["end"] = event["start"] + float(event.get("dur", 0))
    report = json.loads(diagnostic_path.read_text())
    hardware_pid = next(
        event["pid"]
        for event in events
        if event.get("name") == "process_name" and event.get("args", {}).get("name") == "Ascend Hardware"
    )
    hardware = sorted(
        (event for event in events if event.get("pid") == hardware_pid and event.get("ph") == "X"),
        key=lambda event: event["start"],
    )
    work = [
        event
        for event in hardware
        if event.get("args", {}).get("Task Type", "").startswith(("KERNEL", "AI_CORE", "AI_VECTOR", "MIX"))
        or event.get("args", {}).get("Task Type") in ("SDMA", "SDMA_SQE", "PCIE_DMA_SQE")
    ]
    record_events = [event for event in hardware if event["name"] == "EVENT_RECORD"]
    assert len(record_events) == 192, "need exactly three original event records per step"
    cpu = [event for event in events if event.get("ph") == "X" and event.get("cat") == "cpu_op"]
    stage_events = [event for event in cpu if event["name"].startswith("q14diag/stage/")]
    flow_starts = {
        (event.get("cat"), event["id"]): event for event in events if event.get("ph") == "s" and "id" in event
    }
    flow_to_device = defaultdict(dict)
    for event in events:
        if event.get("ph") == "f" and event.get("pid") == hardware_pid:
            start = flow_starts.get((event.get("cat"), event.get("id")))
            if start:
                flow_to_device[event["start"], event["tid"]][event.get("cat")] = start

    output = dict(
        diagnostic_only=True,
        unit="microseconds",
        hardware_pid=hardware_pid,
        event_origin_us=str(origin),
        hardware_task_types=dict(Counter(event.get("args", {}).get("Task Type") for event in hardware)),
        chains=[],
        limitations=[
            "Profile durations cannot replace or be subtracted from unprofiled formal performance.",
            "Host stage ranges include instrumentation; stage clock records exclude some instrumentation.",
            "Synchronous metadata H2D is visible as a host copy call, without a separately attributable DMA task.",
            "Runtime-launch flow timestamp is a launch boundary, not proven API completion or device-ready time.",
        ],
    )
    for chain in report["chains"]:
        index, mode = chain["chain"], chain["mode"]
        stages = [event for event in stage_events if event["name"].startswith(f"q14diag/stage/{index}/{mode}/")]
        host_totals = defaultdict(float)
        host_operations = defaultdict(float)
        hardware_totals = defaultdict(list)
        hardware_counts = Counter()
        gaps_by_host_stage = defaultdict(float)
        gap_boundaries = Counter()
        steps = []
        error_max = 0
        for position, step in enumerate(chain["steps"]):
            begin, graph, end = record_events[48 * index + 3 * position : 48 * index + 3 * position + 3]
            window = (begin["start"], graph["start"])
            error_max = max(error_max, abs((window[1] - window[0]) - step["device_prepare_ms"] * 1000))
            assert abs((end["start"] - graph["start"]) - step["device_graph_ms"] * 1000) < 1
            prefix = f"q14diag/stage/{index}/{mode}/{step['step']}/"
            step_stages = [event for event in stages if event["name"].startswith(prefix)]
            actual_work = [event for event in work if event["start"] < window[1] and event["end"] > window[0]]
            assert len(actual_work) == 11, "expected one cast, five gathers and five copies"
            for event in actual_work:
                association = flow_to_device.get((event["start"], event["tid"]), {})
                source = association.get("async_npu")
                assert source is not None, "all preparation work needs a host association"
                enclosing = [stage for stage in step_stages if contains(stage, source["start"])]
                assert len(enclosing) == 1
                stage_name = enclosing[0]["name"][len(prefix) :]
                if stage_name.startswith("input/"):
                    op = "gather" if "IndexSelect" in event["name"] else "copy"
                    category = f"{stage_name}/{op}"
                else:
                    category = stage_name
                hardware_totals[category].append(span(event))
                hardware_counts[category] += 1
            idle = complement([span(event) for event in actual_work], window)
            stage_overlap = 0
            for stage in step_stages:
                name = stage["name"][len(prefix) :]
                host_totals[name] += stage["dur"]
                operators = [
                    event
                    for event in cpu
                    if event["name"] in ("aten::copy_", "aten::index_select")
                    and event["tid"] == stage["tid"]
                    and contains(stage, event["start"])
                ]
                for operator in operators:
                    host_operations[f"{name}/{operator['name']}"] += operator["dur"]
                if name == "metadata_tensor_and_h2d":
                    copy = next(event for event in operators if event["name"] == "aten::copy_")
                    host_operations["metadata_tensor_prefix_including_instrumentation"] += (
                        copy["start"] - stage["start"]
                    )
                    host_operations["metadata_copy_suffix_including_instrumentation"] += stage["end"] - copy["end"]
                contribution = intersection(idle, [span(stage)])
                gaps_by_host_stage[name] += contribution
                stage_overlap += contribution
            gaps_by_host_stage["outside_stage_markers_including_instrumentation"] += length(idle) - stage_overlap

            # These three disjoint bins split each visible hardware-idle gap.
            # They are an alternative view, never added to host-stage overlaps.
            for low, high in idle:
                successor = min(
                    (event for event in [*actual_work, graph] if event["start"] >= high - 1e-6),
                    key=lambda event: event["start"],
                )
                associations = flow_to_device.get((successor["start"], successor["tid"]), {})
                caller = associations.get("async_npu")
                launch = associations.get("HostToDevice")
                if caller is None or launch is None:
                    gap_boundaries["unassociated"] += high - low
                    continue
                caller_time = min(high, max(low, caller["start"]))
                launch_time = min(high, max(caller_time, launch["start"]))
                gap_boundaries["before_host_operation"] += caller_time - low
                gap_boundaries["host_operation_to_runtime_launch"] += launch_time - caller_time
                gap_boundaries["runtime_launch_to_device_start"] += high - launch_time
            busy = length(clipped([span(event) for event in actual_work], window))
            assert abs(busy + length(idle) - (window[1] - window[0])) < 1e-5
            steps.append(
                dict(
                    step=step["step"],
                    prepared_rows=step["prepared_rows"],
                    preparation_us=window[1] - window[0],
                    graph_us=end["start"] - graph["start"],
                    visible_preparation_busy_union_us=busy,
                    visible_preparation_idle_us=length(idle),
                    preparation_tasks=len(actual_work),
                )
            )
        assert error_max < 1, "trace event boundaries do not match NPU event timing"
        idle_total = sum(step["visible_preparation_idle_us"] for step in steps)
        assert abs(sum(gaps_by_host_stage.values()) - idle_total) < 1e-5
        assert abs(sum(gap_boundaries.values()) - idle_total) < 1e-5
        prepare_markers = [event for event in cpu if event["name"].startswith(f"q14diag/prepare/{index}/{mode}/")]
        host_covered = length([span(event) for event in stages])
        output["chains"].append(
            dict(
                chain=index,
                mode=mode,
                trace_event_vs_timer_max_difference_us=error_max,
                steps=steps,
                preparation_us=sum(step["preparation_us"] for step in steps),
                graph_us=sum(step["graph_us"] for step in steps),
                visible_preparation_busy_union_us=sum(step["visible_preparation_busy_union_us"] for step in steps),
                visible_preparation_idle_us=idle_total,
                host_stage_wall_us=dict(host_totals),
                host_operation_wall_us=dict(host_operations),
                host_prepare_wall_us=length([span(event) for event in prepare_markers]),
                host_prepare_outside_stage_markers_us=length([span(event) for event in prepare_markers]) - host_covered,
                device_categories={
                    name: dict(count=hardware_counts[name], union_us=length(intervals))
                    for name, intervals in hardware_totals.items()
                },
                idle_overlap_host_stage_us=dict(gaps_by_host_stage),
                idle_disjoint_launch_boundaries_us=dict(gap_boundaries),
            )
        )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--diagnostic", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("preserve the previous analysis")
    args.output.write_text(json.dumps(analyze(args.trace, args.diagnostic), indent=2) + "\n")


if __name__ == "__main__":
    main()
