# SPDX-License-Identifier: Apache-2.0
"""Cross-check saved FIA wait intervals using an independent stream walk."""

import json
import math
from collections import defaultdict
from decimal import Decimal

import audit


def merge_length(intervals):
    ordered = sorted((left, right) for left, right in intervals if left < right)
    if not ordered:
        return Decimal(0)
    left, right = ordered[0]
    total = Decimal(0)
    for begin, end in ordered[1:]:
        if begin > right:
            total += right - left
            left, right = begin, end
        else:
            right = max(right, end)
    return total + right - left


def main():
    report = audit.read(audit.HERE / "audit.json")
    assert json.loads(json.dumps(audit.build())) == report
    checked = 0
    for trace in report["traces"]:
        document = json.loads((audit.ROOT / trace["file"]).read_bytes(), parse_float=Decimal)
        events = document if isinstance(document, list) else document["traceEvents"]
        by_stream = defaultdict(list)
        for event in events:
            if event.get("ph") == "X" and str(event.get("args", {}).get("Task Type", "")).startswith("KERNEL_"):
                if event["args"]["Task Type"] != "KERNEL_AICPU":
                    by_stream[event["pid"], event["tid"]].append(event)
        observed = []
        for (pid, tid), kernels in by_stream.items():
            kernels.sort(key=lambda event: Decimal(str(event["ts"])))
            previous = None
            for event in kernels:
                begin = Decimal(str(event["ts"]))
                end = begin + Decimal(str(event["dur"]))
                if previous is not None:
                    assert previous[1] <= begin
                if event["name"] == audit.DEVICE_FIA:
                    assert previous is not None and "ScatterPaKvCache" in previous[2]
                    gap_begin = previous[1]
                    clipped = []
                    for wait in events:
                        if wait.get("ph") != "X" or wait["name"] != "MEM_WAIT_VALUE":
                            continue
                        if (wait.get("pid"), wait.get("tid")) != (pid, tid):
                            continue
                        left = max(gap_begin, Decimal(str(wait["ts"])))
                        right = min(begin, Decimal(str(wait["ts"])) + Decimal(str(wait["dur"])))
                        if left < right:
                            clipped.append((left, right))
                    observed.append((begin, float(begin - gap_begin), float(merge_length(clipped))))
                previous = (begin, end, event["name"])
        observed.sort()
        assert len(observed) == trace["fia_count"]
        for (_, gap, polling), original in zip(observed, trace["per_fia"], strict=True):
            assert math.isclose(gap, original["stream_gap_us"], abs_tol=1e-9)
            assert math.isclose(polling, original["polling_union_us"], abs_tol=1e-9)
            assert 0 <= polling <= gap
            checked += 1
    for intervals in ([(0, 3), (1, 2), (2, 7)], [], [(0, 0), (2, 3), (4, 5)]):
        decimals = [(Decimal(left), Decimal(right)) for left, right in intervals]
        assert audit.union_length(decimals) == merge_length(decimals)
    full = [trace for trace in report["traces"] if trace["fia_count"] == 16]
    assert len(full) == 12 and all(trace["all_outside_model_writes_before_first_device_fia"] for trace in full)
    assert all(trace["stream_gap_max_us"] < 5 for trace in full)
    assert checked == 198
    result = dict(
        scope="independent CPU reconstruction; not NPU execution or unprofiled latency validation",
        saved_trace_hashes_checked=len(report["traces"]),
        archived_source_hashes_checked=sum(len(entry["checked_archived_sources"]) for entry in report["provenance"]),
        fia_predecessor_gaps_and_polling_unions_cross_checked=checked,
        deterministic_regeneration=True,
        memory_record_caller_sources_match_launch=True,
    )
    (audit.HERE / "checks.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
