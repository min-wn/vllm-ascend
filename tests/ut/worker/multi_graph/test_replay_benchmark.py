import pytest

from vllm_ascend.worker.multi_graph.replay_benchmark import (
    ReplayWorkload,
    inspect_replay_trace,
    inspect_shared_dense_trace,
)


@pytest.mark.parametrize("query", [1, 2, 4, 8])
def test_verification_requests_remain_whole_and_dummy_kv_is_disjoint(query):
    case = ReplayWorkload(33, 64, query, 129, (17, 16))
    full = case.metadata_values(0, 64)
    first, second = case.metadata_values(0, 17), case.metadata_values(17, 33)
    assert first["slots"] + second["slots"] == full["slots"][: 33 * query]
    assert first["positions"] + second["positions"] == full["positions"][: 33 * query]
    assert second["boundaries"] == list(range(0, 16 * query + 1, query))
    assert len(set(full["slots"])) == 64 * query
    assert not set(full["slots"][: 33 * query]) & set(full["slots"][33 * query :])
    for index, slot in enumerate(full["slots"]):
        assert slot // 128 in full["pages"][index // query]
    assert case.kv_blocks == 128


@pytest.mark.parametrize(
    "args",
    [
        (33, 32, 4, 512, (17, 16)),
        (33, 64, 3, 512, (17, 16)),
        (33, 64, 8, 4, (17, 16)),
        (33, 64, 4, 512, (17, 17)),
        (33, 64, 4, 512, (33, 0)),
    ],
)
def test_invalid_fixed_replay_workload(args):
    with pytest.raises(ValueError):
        ReplayWorkload(*args)


def test_reject_invalid_request_slice():
    case = ReplayWorkload(7, 8, 2, 64, (4, 3))
    for start, stop in ((-1, 4), (4, 4), (0, 9)):
        with pytest.raises(ValueError):
            case.metadata_values(start, stop)


@pytest.mark.parametrize("query", [1, 2, 4, 8])
def test_capacity_graph_does_not_compute_attention_for_padding(query):
    case = ReplayWorkload(33, 64, query, 512, (17, 16))
    values = case.graph_values(0, 64)
    assert values["boundaries"] == list(range(0, 64 * query + 1, query))
    assert values["lengths"] == [512] * 33 + [0] * 31
    assert values["slots"][33 * query :] == [-1] * (31 * query)
    assert len(values["positions"]) == 64 * query
    assert case.graph_values(17, 33) == case.metadata_values(17, 33)
    with pytest.raises(ValueError):
        case.graph_values(33, 64)


def trace_event(start, duration, model, task="KERNEL_AICORE"):
    return dict(pid=9, ph="X", ts=start, dur=duration, args={"Model Id": model, "Task Type": task})


@pytest.mark.parametrize("wrapped", [False, True])
def test_trace_counts_only_distinct_model_kernel_overlap(wrapped):
    events = [
        dict(name="process_name", pid=9, args={"name": "Ascend Hardware"}),
        dict(name="replay.update.parallel", pid=1, ph="X", ts=10, dur=20),
        trace_event(8, 10, 1),
        trace_event(15, 5, 1),  # Same-model overlap is counted once.
        trace_event(15, 8, 2),
        trace_event(0, 100, 3, "MEM_WAIT_VALUE"),
        trace_event(0, 100, 4294967295),
        trace_event(35, 5, 3),
    ]
    result = inspect_replay_trace({"traceEvents": events} if wrapped else events)[0]
    assert result["model_kernel_busy_us"] == {"1": 10, "2": 8}
    assert result["kernel_union_us"] == 13
    assert result["two_model_kernel_overlap_us"] == 5


def test_trace_serial_models_do_not_overlap():
    events = [
        dict(name="process_name", pid=9, args={"name": "Ascend Hardware"}),
        dict(name="replay.update.serial", pid=1, ph="X", ts=0, dur=20),
        trace_event(1, 5, 1),
        trace_event(10, 5, 2),
    ]
    assert inspect_replay_trace(events)[0]["two_model_kernel_overlap_us"] == 0


@pytest.mark.parametrize("events", [[], [dict(name="replay.update.parallel", ph="X", ts=0, dur=20)]])
def test_trace_rejects_missing_device_evidence(events):
    with pytest.raises(ValueError):
        inspect_replay_trace(events)


@pytest.mark.parametrize("wrapped", [False, True])
def test_shared_trace_measures_attention_overlap_within_one_model(wrapped):
    events = [
        dict(name="process_name", pid=9, args={"name": "Ascend Hardware"}),
        dict(name="layer.joined_parallel", pid=1, ph="X", ts=0, dur=30),
        dict(trace_event(2, 10, 1), name="FusedInferAttentionScore", tid=11),
        dict(trace_event(8, 8, 1), name="FusedInferAttentionScore", tid=12),
        dict(trace_event(17, 5, 1), name="aclnnMatmul_MatMulV2", tid=11),
        dict(trace_event(0, 30, 1, "EVENT_WAIT"), name="wait", tid=11),
        dict(trace_event(40, 10, 1), name="FusedInferAttentionScore", tid=11),
    ]
    row = inspect_shared_dense_trace({"traceEvents": events} if wrapped else events)[0]
    assert row["attention_overlap_us"] == 4
    assert row["attention_stream_count"] == 2
    assert row["attention_count"] == 2
    assert row["matmul_count"] == 1
    assert row["matmul_duration_sum_us"] == 5
    assert row["kernel_union_us"] == 19
    assert row["kernel_duration_sum_us"] == 23


@pytest.mark.parametrize("events", [[], [dict(name="layer.joined_parallel", ph="X", ts=0, dur=20)]])
def test_shared_trace_rejects_missing_attention_evidence(events):
    with pytest.raises(ValueError):
        inspect_shared_dense_trace(events)
