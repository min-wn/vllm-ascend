import importlib.util
from pathlib import Path


def test_trace_clips_kernels_and_separates_host_phases():
    path = Path(__file__).resolve().parents[4] / "benchmarks/multi_graph_bucket_trace.py"
    spec = importlib.util.spec_from_file_location("bucket_trace", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def event(name, ts, dur, pid=1, tid=1, task="CONTROL"):
        return dict(name=name, ts=str(ts), dur=dur, ph="X", pid=pid, tid=tid, args={"Task Type": task})

    report = module.analyze(
        dict(
            traceEvents=[
                dict(name="process_name", ph="M", pid=2, args={"name": "Ascend Hardware"}),
                event("buckets.split_parallel", 10, 10),
                event("bucket_phase.prepare", 10, 2),
                event("bucket_phase.update_lane0", 12, 2),
                event("bucket_phase.update_lane0", 15, 1),
                event("FusedInferAttentionScore", 8, 7, 2, 8, "KERNEL_AICORE"),
                event("FusedInferAttentionScore", 13, 4, 2, 9, "KERNEL_AICORE"),
                event("MatMul", 18, 5, 2, 8, "KERNEL_AICORE"),
                event("MEM_WAIT_VALUE", 10, 10, 2, 8),
                event("FusedInferAttentionScore", 10, 10),
            ]
        )
    )
    assert len(report) == 1
    row = report[0]
    assert row["kernel_union_us"] == 9
    assert row["kernel_overlap_us"] == 2
    assert row["host_phases_us"] == dict(prepare=2, update_lane0=3)
    assert row["operators"]["FIA"] == dict(count=2, sum_us=9, union_us=7, overlap_us=2)
    assert row["operators"]["MatMul"]["sum_us"] == 2
    assert row["lanes"]["8"]["internal_gap_us"] == 3
    assert row["lanes"]["9"]["first_fia_us"] == 3
    assert module.analyze([]) == []
