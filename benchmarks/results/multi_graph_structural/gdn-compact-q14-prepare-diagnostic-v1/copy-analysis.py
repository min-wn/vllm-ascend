"""Read-only attribution of the saved Q14 preparation diagnostic trace.

This does not import torch or run a device. All times are microseconds, and
Decimal timestamps are rebased before comparison. HostToDevice is a profiler
flow category here; it does not mean that an input tensor was uploaded.
"""

import argparse
import collections
import json
from decimal import Decimal
from pathlib import Path


D = Decimal


def interval_union_us(intervals):
    result = D(0)
    end = None
    for begin, finish in sorted(intervals):
        if end is None or begin > end:
            result += finish - begin
            end = finish
        elif finish > end:
            result += finish - end
            end = finish
    return result


def decimal_json(value):
    if isinstance(value, D):
        return str(value)
    if isinstance(value, dict):
        return {str(key): decimal_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [decimal_json(item) for item in value]
    return value


def analyze(trace_path, diagnostic_path):
    events = json.loads(trace_path.read_text(encoding="utf-8"))
    diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))
    origin = min(D(event["ts"]) for event in events if "ts" in event)
    for event in events:
        if "ts" in event:
            event["begin"] = D(event["ts"]) - origin
            event["duration"] = D(str(event.get("dur", 0)))
            event["end"] = event["begin"] + event["duration"]

    process_names = {
        event["pid"]: event["args"]["name"]
        for event in events
        if event.get("ph") == "M" and event.get("name") == "process_name"
    }
    hardware_pids = {
        pid for pid, name in process_names.items() if name == "Ascend Hardware"
    }
    stages = [
        event for event in events
        if event.get("ph") == "X"
        and event.get("name", "").startswith("q14diag/stage/")
    ]
    input_stages = [stage for stage in stages if "/input/" in stage["name"]]
    enqueue = {
        event["args"]["correlation_id"]: event
        for event in events if event.get("cat") == "enqueue"
    }
    dequeue = [event for event in events if event.get("cat") == "dequeue"]
    queue_to_stage = {}
    for queue_id, event in enqueue.items():
        matches = [
            stage for stage in stages
            if stage["tid"] == event["tid"]
            and stage["begin"] <= event["begin"] < stage["end"]
        ]
        assert len(matches) <= 1, (queue_id, matches)
        if matches:
            queue_to_stage[queue_id] = matches[0]

    starts = collections.defaultdict(list)
    for event in events:
        if event.get("cat") == "HostToDevice" and event.get("ph") == "s":
            starts[str(event["id"])].append(event)
    hardware = [
        event for event in events
        if event.get("ph") == "X" and event.get("pid") in hardware_pids
    ]
    hardware_at = collections.defaultdict(list)
    for event in hardware:
        hardware_at[event["pid"], event["tid"], event["begin"]].append(event)

    step_info = {
        (chain["chain"], step["step"]): step
        for chain in diagnostic["chains"] for step in chain["steps"]
    }
    cpu_ops = [
        event for event in events
        if event.get("ph") == "X" and event.get("cat") == "cpu_op"
    ]
    rows = []
    unmatched = collections.Counter()
    for finish in events:
        if finish.get("cat") != "HostToDevice" or finish.get("ph") != "f":
            continue
        flow_starts = starts.get(str(finish["id"]), [])
        if len(flow_starts) != 1:
            unmatched["missing_or_reused_flow_start"] += 1
            continue
        start = flow_starts[0]
        queue_matches = [
            event for event in dequeue
            if event["tid"] == start["tid"]
            and event["begin"] <= start["begin"] < event["end"]
        ]
        assert len(queue_matches) == 1, (start, queue_matches)
        queue = queue_matches[0]
        queue_id = queue["args"]["correlation_id"]
        stage = queue_to_stage.get(queue_id)
        if stage is None:
            unmatched["outside_prepare_stage"] += 1
            continue
        matches = hardware_at[finish["pid"], finish["tid"], finish["begin"]]
        assert len(matches) == 1, (finish, matches)
        device = matches[0]
        parts = stage["name"].split("/")
        chain, mode, step = int(parts[2]), parts[3], int(parts[4])
        info = step_info[chain, step]
        queue_enqueue = enqueue[queue_id]
        operator_name = (
            "aten::copy_" if queue["name"].endswith("InplaceCopy")
            else "aten::index_select"
        )
        operators = [
            event for event in cpu_ops
            if event["name"] == operator_name
            and event["tid"] == stage["tid"]
            and event["begin"] <= queue_enqueue["begin"] < event["end"]
        ]
        assert len(operators) == 1, (queue_id, operators)
        operator = operators[0]
        nested = collections.Counter(
            event["name"] for event in cpu_ops
            if event["tid"] == operator["tid"]
            and event is not operator
            and operator["begin"] <= event["begin"]
            and event["end"] <= operator["end"]
        )
        rows.append({
            "chain": chain, "mode": mode, "step": step,
            "load": "q1_only" if set(info["actual_widths"]) == {1} else "mixed",
            "prepared_rows": info["prepared_rows"],
            "stage": "/".join(parts[5:]), "queue_id": queue_id,
            "host_operator": operator_name, "device_name": device["name"],
            "device_task_type": device["args"].get("Task Type"),
            "host_begin_us": operator["begin"], "host_end_us": operator["end"],
            "host_wall_us": operator["duration"],
            "device_begin_us": device["begin"], "device_end_us": device["end"],
            "device_duration_us": device["duration"],
            "host_return_minus_device_end_us": operator["end"] - device["end"],
            "host_return_minus_device_begin_us": operator["end"] - device["begin"],
            "host_pre_enqueue_us": queue_enqueue["begin"] - operator["begin"],
            "host_enqueue_wall_us": queue_enqueue["duration"],
            "enqueue_to_device_begin_us": device["begin"] - queue_enqueue["begin"],
            "host_enqueue_begin_us": queue_enqueue["begin"],
            "host_stage_begin_us": stage["begin"], "host_stage_end_us": stage["end"],
            "host_stage_wall_us": stage["duration"],
            "device_finish_before_host_stage_end_us": stage["end"] - device["end"],
            "nested_cpu_operator_counts": nested,
        })

    inputs = [row for row in rows if row["stage"].startswith("input/")]
    assert len(input_stages) == 320
    assert len(inputs) == 640
    for stage in input_stages:
        parts = stage["name"].split("/")
        matching = [row for row in inputs if (
            row["chain"] == int(parts[2]) and row["step"] == int(parts[4])
            and row["stage"] == "/".join(parts[5:])
        )]
        assert len(matching) == 2
        assert {row["host_operator"] for row in matching} == {
            "aten::copy_", "aten::index_select"
        }

    gathers = {
        (row["chain"], row["step"], row["stage"]): row
        for row in inputs if row["host_operator"] == "aten::index_select"
    }
    for row in inputs:
        if row["host_operator"] != "aten::copy_":
            continue
        gather = gathers[row["chain"], row["step"], row["stage"]]
        row["pre_enqueue_overlap_previous_gather_device_us"] = max(
            D(0),
            min(row["host_enqueue_begin_us"], gather["device_end_us"])
            - max(row["host_begin_us"], gather["device_begin_us"]),
        )
        row["enqueue_minus_previous_gather_finish_us"] = (
            row["host_enqueue_begin_us"] - gather["device_end_us"]
        )

    aggregates = []
    for dimensions in (
        ("mode", "stage", "host_operator", "device_name", "device_task_type"),
        ("chain", "mode", "load", "stage", "host_operator"),
        ("mode", "load", "stage", "host_operator"),
        ("chain", "mode", "host_operator"),
        ("mode", "host_operator", "device_task_type"),
    ):
        buckets = collections.defaultdict(list)
        for row in inputs:
            buckets[tuple(row[key] for key in dimensions)].append(row)
        for key, group in sorted(buckets.items()):
            relative_return = [row["host_return_minus_device_end_us"] for row in group]
            aggregates.append({
                "group_by": dimensions, "group": dict(zip(dimensions, key)),
                "count": len(group),
                "host_wall_sum_us": sum(row["host_wall_us"] for row in group),
                "device_duration_sum_us": sum(row["device_duration_us"] for row in group),
                "host_wall_mean_us": sum(row["host_wall_us"] for row in group) / len(group),
                "device_duration_mean_us": sum(row["device_duration_us"] for row in group) / len(group),
                "host_pre_enqueue_sum_us": sum(row["host_pre_enqueue_us"] for row in group),
                "host_enqueue_wall_sum_us": sum(row["host_enqueue_wall_us"] for row in group),
                "host_pre_enqueue_mean_us": sum(row["host_pre_enqueue_us"] for row in group) / len(group),
                "device_finished_before_host_stage_end_count": sum(
                    row["device_finish_before_host_stage_end_us"] >= 0 for row in group
                ),
                "return_after_device_finish_count": sum(value >= 0 for value in relative_return),
                "return_before_device_begin_count": sum(
                    row["host_return_minus_device_begin_us"] < 0 for row in group
                ),
                "return_minus_device_finish_min_us": min(relative_return),
                "return_minus_device_finish_mean_us": sum(relative_return) / len(group),
                "return_minus_device_finish_max_us": max(relative_return),
                **({
                    "pre_enqueue_overlap_previous_gather_sum_us": sum(
                        row["pre_enqueue_overlap_previous_gather_device_us"] for row in group
                    ),
                    "enqueue_after_previous_gather_finish_count": sum(
                        row["enqueue_minus_previous_gather_finish_us"] >= 0 for row in group
                    ),
                    "enqueue_minus_previous_gather_finish_mean_us": sum(
                        row["enqueue_minus_previous_gather_finish_us"] for row in group
                    ) / len(group),
                    "nested_cpu_operator_counts": dict(sum(
                        (collections.Counter(row["nested_cpu_operator_counts"]) for row in group),
                        collections.Counter(),
                    )),
                } if group[0]["host_operator"] == "aten::copy_" else {}),
            })

    stage_summaries = []
    for dimensions in (("chain", "mode", "load", "stage"), ("mode", "load", "stage")):
        buckets = collections.defaultdict(list)
        for row in inputs:
            if row["host_operator"] == "aten::copy_":
                buckets[tuple(row[key] for key in dimensions)].append(row)
        for key, group in sorted(buckets.items()):
            stage_summaries.append({
                "group_by": dimensions, "group": dict(zip(dimensions, key)),
                "count": len(group),
                "host_input_stage_wall_sum_us": sum(row["host_stage_wall_us"] for row in group),
                "host_input_stage_wall_mean_us": sum(row["host_stage_wall_us"] for row in group) / len(group),
            })

    removal_bounds = []
    for chain in range(4):
        copies = [row for row in inputs if row["chain"] == chain and row["host_operator"] == "aten::copy_"]
        host_intervals = [(row["host_begin_us"], row["host_end_us"]) for row in copies]
        device_intervals = [(row["device_begin_us"], row["device_end_us"]) for row in copies]
        host_sum = sum(row["host_wall_us"] for row in copies)
        device_sum = sum(row["device_duration_us"] for row in copies)
        removal_bounds.append({
            "chain": chain, "mode": copies[0]["mode"],
            "host_copy_wall_sum_us": host_sum,
            "device_copy_duration_sum_us": device_sum,
            "coarse_sum_bound_us": host_sum + device_sum,
            "non_double_counted_interval_union_bound_us": interval_union_us(host_intervals + device_intervals),
            "all_device_copies_finish_before_corresponding_input_host_stage_end": all(
                row["device_finish_before_host_stage_end_us"] >= 0 for row in copies
            ),
            "host_path_only_budget_us": host_sum,
        })

    modes = ("forced_q4", "q14_ready")
    mode_copy_sums = {}
    for mode in modes:
        copies = [row for row in inputs if row["mode"] == mode and row["host_operator"] == "aten::copy_"]
        mode_copy_sums[mode] = {
            "host_copy_wall_per_16_step_chain_us": sum(row["host_wall_us"] for row in copies) / 2,
            "device_copy_duration_per_16_step_chain_us": sum(row["device_duration_us"] for row in copies) / 2,
            "host_copy_pre_enqueue_per_16_step_chain_us": sum(row["host_pre_enqueue_us"] for row in copies) / 2,
        }
    host_delta = (
        mode_copy_sums["q14_ready"]["host_copy_wall_per_16_step_chain_us"]
        - mode_copy_sums["forced_q4"]["host_copy_wall_per_16_step_chain_us"]
    )
    pre_enqueue_delta = (
        mode_copy_sums["q14_ready"]["host_copy_pre_enqueue_per_16_step_chain_us"]
        - mode_copy_sums["forced_q4"]["host_copy_pre_enqueue_per_16_step_chain_us"]
    )

    return decimal_json({
        "trace": str(trace_path.resolve()), "diagnostic": str(diagnostic_path.resolve()),
        "timestamp_origin_us": origin, "events": len(events),
        "mapping": {
            "mapped_prepare_hardware_tasks": len(rows),
            "input_stage_count": len(input_stages),
            "mapped_input_hardware_tasks": len(inputs),
            "expected_input_gathers": 320, "expected_input_copies": 320,
            "method": "HostToDevice source inside dequeue; correlation_id to enqueue inside host stage; flow destination matches exact hardware pid/tid/timestamp",
            "flow_end_exclusions": unmatched,
        },
        "interpretation": [
            "HostToDevice is a profiler flow category, not tensor transfer direction.",
            "Every SDMA copy returns from aten::copy_ before its own device copy starts. Waiting for completion of this copy does not explain the added host duration.",
            "SDMA pre-enqueue host intervals overlap the previous gather device execution, but the trace does not expose an inner synchronization call. Waiting for a previous operation cannot be excluded or established from this ordering alone.",
            "Inside all input copies, the only nested cpu_op is aclnnInplaceCopy; there are no nested empty, reshape or contiguous operations in this trace.",
            "The removal bounds refer only to this profiled trace and ideal zero-cost copies; they are not predictions for an out= implementation or the frozen formal run.",
            "The interval-union bound avoids duplicate host/device time; it is still generous because other host/device work and replacement gather costs can remain on the critical path.",
            "Subtracting the two modes' per-chain bounds is not a rigorous bound on a fair transformed benchmark's relative speedup.",
            "All input device copies finish before their input host stage ends. Their device durations therefore cannot simply be added to host-path savings; the host_path_only_budget is the tighter budget under a fixed-dependency trace replay model.",
        ],
        "mode_copy_sums": mode_copy_sums,
        "diagnostic_host_copy_increment_per_16_step_chain_us": host_delta,
        "diagnostic_host_copy_pre_enqueue_increment_per_16_step_chain_us": pre_enqueue_delta,
        "pre_enqueue_fraction_of_host_copy_increment_percent": 100 * pre_enqueue_delta / host_delta,
        "aggregates": aggregates, "input_stage_wall_summaries": stage_summaries,
        "zero_copy_removal_accounting_bounds": removal_bounds,
        "input_task_rows": inputs,
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, default=Path("work/gdn-compact-q14-prepare-diagnostic-download-v1/trace.json"))
    parser.add_argument("--diagnostic", type=Path, default=Path("work/gdn-compact-q14-prepare-diagnostic-download-v1/diagnostic.json"))
    parser.add_argument("--output", type=Path, default=Path("work/q14_prepare_copy_analysis.json"))
    args = parser.parse_args()
    result = analyze(args.trace, args.diagnostic)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"mapping": result["mapping"], "bounds": result["zero_copy_removal_accounting_bounds"]}, indent=2))


if __name__ == "__main__":
    main()
