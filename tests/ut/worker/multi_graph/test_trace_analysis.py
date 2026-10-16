import pytest

from vllm_ascend.worker.multi_graph.trace_analysis import analyze_trace, union_duration


def test_union_nested_and_disjoint():
    assert union_duration([(0, 5), (1, 3), (4, 8), (10, 12)]) == 10


def test_device_windows_clip_and_exclude_control():
    def event(name, ts, dur, pid=1, tid=1):
        return dict(
            ph="X",
            name=name,
            ts=str(ts),
            dur=dur,
            pid=pid,
            tid=tid,
            args={"Task Type": "KERNEL_AICORE" if name == "kernel" else "CONTROL"},
        )

    events = [
        dict(ph="M", name="process_name", pid=2, args={"name": "Ascend Hardware"}),
        event("mg.decode.0", 10, 3),
        event("mg.decode.1", 20, 4),
        event("mg.graph.update", 11, 2),
        event("kernel", 8, 7, 2, 9),
        event("kernel", 13, 4, 2, 10),
        event("MEM_WAIT_VALUE", 10, 10, 2, 11),
        event("cpu", 10, 10),
    ]
    result = analyze_trace({"traceEvents": events}, (9, 10))
    step = result["steps"][0]
    assert step["device_busy_union_us"] == 7
    assert step["device_gap_us"] == 3
    assert step["cross_stream_excess_us"] == 2
    assert step["lane_kernel_overlap_us"] == 2
    assert step["stream_busy_us"] == {"9": 5, "10": 4}
    assert result["phases"]["mg.graph.update"]["sum_us"] == 2
    assert result["phases"]["mg.decode"]["count"] == 2


def test_unmarked_trace_reports_no_attribution():
    assert analyze_trace([])["steps"] == []


@pytest.mark.parametrize("streams", [(9, 9), (9,), (9, 10)])
def test_invalid_or_absent_lane_streams_are_rejected(streams):
    with pytest.raises(ValueError, match="stream"):
        analyze_trace([], streams)
