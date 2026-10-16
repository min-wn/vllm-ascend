"""Offline trace interval analysis; durations are microseconds, not benchmark latency."""

from __future__ import annotations

from collections import defaultdict
from statistics import median


def union_duration(intervals):
    total = 0.0
    end = float("-inf")
    for start, stop in sorted(intervals):
        total += max(0.0, stop - max(start, end))
        end = max(end, stop)
    return total


def analyze_trace(document, lane_streams=()):
    if lane_streams and (len(lane_streams) != 2 or lane_streams[0] == lane_streams[1]):
        raise ValueError("lane streams must be two distinct physical streams")
    events = document if isinstance(document, list) else document["traceEvents"]
    hardware = {
        event["pid"]
        for event in events
        if event.get("ph") == "M"
        and event.get("name") == "process_name"
        and event.get("args", {}).get("name") == "Ascend Hardware"
    }
    complete = [
        {**event, "ts": float(event["ts"]), "dur": float(event["dur"])}
        for event in events
        if event.get("ph") == "X" and float(event.get("dur", 0)) > 0
    ]
    kernel_streams = {
        event["tid"]
        for event in complete
        if event["pid"] in hardware and event.get("args", {}).get("Task Type", "").startswith("KERNEL")
    }
    if any(stream not in kernel_streams for stream in lane_streams):
        raise ValueError("requested lane stream has no device kernel events")
    markers = sorted(
        (event for event in complete if event["name"].startswith("mg.decode.")), key=lambda event: event["ts"]
    )
    phases = defaultdict(list)
    for event in complete:
        if event["name"].startswith("mg."):
            phases["mg.decode" if event["name"].startswith("mg.decode.") else event["name"]].append(event["dur"])
    steps = []
    for marker, following in zip(markers, markers[1:]):
        start, stop = marker["ts"], following["ts"]
        streams = defaultdict(list)
        compute = defaultdict(list)
        host_phases = defaultdict(float)
        names = defaultdict(lambda: {"count": 0, "sum_us": 0.0})
        for event in complete:
            left, right = max(start, event["ts"]), min(stop, event["ts"] + event["dur"])
            if right > left and event["name"].startswith("mg.") and not event["name"].startswith("mg.decode."):
                host_phases[event["name"]] += right - left
            if right <= left or event["pid"] not in hardware:
                continue
            name = event["name"]
            task_type = event.get("args", {}).get("Task Type", "")
            is_kernel = task_type.startswith("KERNEL")
            if not is_kernel and name != "MEMCPY_ASYNC":
                continue
            if is_kernel:
                compute[str(event["tid"])].append((left, right))
            names[name]["count"] += 1
            names[name]["sum_us"] += right - left
            streams[str(event["tid"])].append((left, right))
        all_intervals = [interval for intervals in streams.values() for interval in intervals]
        busy = union_duration(all_intervals)
        steps.append(
            {
                "marker": marker["name"],
                "window_us": stop - start,
                "host_forward_us": marker["dur"],
                "host_phases_us": dict(host_phases),
                "device_busy_union_us": busy,
                "device_gap_us": stop - start - busy,
                "stream_busy_us": {stream: union_duration(intervals) for stream, intervals in streams.items()},
                "cross_stream_excess_us": sum(union_duration(values) for values in streams.values()) - busy,
                "lane_kernel_overlap_us": (
                    sum(union_duration(compute[str(stream)]) for stream in lane_streams)
                    - union_duration([interval for stream in lane_streams for interval in compute[str(stream)]])
                    if lane_streams
                    else None
                ),
                "device_operations": dict(names),
            }
        )
    return {
        "hardware_processes": sorted(hardware),
        "decode_markers": len(markers),
        "lane_streams": list(lane_streams),
        "phases": {
            name: {"count": len(values), "sum_us": sum(values), "p50_us": median(values)}
            for name, values in sorted(phases.items())
        },
        "steps": steps,
        "limitations": [
            "Nested phase sums overlap and must not be added together.",
            "Device windows span consecutive host decode starts and may include sampling or transfers.",
            "Final decode has no bounded window and is omitted from device analysis.",
            "Cross-stream excess includes copies and does not prove lane compute overlap.",
            "Profiler durations must not be substituted for unprofiled benchmark latency.",
        ],
    }
