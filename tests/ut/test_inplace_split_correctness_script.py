import json

from examples.test_split_batch_correctness_npu import (
    _apply_capture_sizes,
    _build_split_additional_config,
    _ensure_fixed_batch_graph_capacity,
    _expected_inplace_split,
    _summarize_split_debug_trace,
)


def test_expected_inplace_split_for_phase10_fixed_batch():
    assert _expected_inplace_split(416, [256, 384, 512]) == {
        "total_tokens": 416,
        "first_tokens": 384,
        "second_tokens": 32,
        "first_start_num_tokens": 0,
        "second_start_num_tokens": 384,
    }


def test_apply_capture_sizes_overrides_compilation_config():
    args = {"compilation_config": {"level": 3, "cudagraph_mode": "FULL"}}

    _apply_capture_sizes(args, [512, 256, 384, 384])

    assert args["compilation_config"]["cudagraph_capture_sizes"] == [
        256, 384, 512
    ]
    assert args["compilation_config"]["level"] == 3


def test_fixed_batch_graph_capacity_uses_padded_capture_size():
    args = {}

    _ensure_fixed_batch_graph_capacity(
        args,
        batch_size=416,
        capture_sizes=[256, 384, 512],
    )

    assert args["max_num_seqs"] == 512


def test_fixed_batch_graph_capacity_rejects_too_small_override():
    args = {"max_num_seqs": 416}

    try:
        _ensure_fixed_batch_graph_capacity(
            args,
            batch_size=416,
            capture_sizes=[256, 384, 512],
        )
    except ValueError:
        pass
    else:
        raise AssertionError("expected max_num_seqs capacity validation")


def test_build_split_config_for_inplace_serial_validation():
    config = _build_split_additional_config(
        enabled=True,
        split_mode="inplace_serial",
        num_splits=2,
        enable_parallel_streams=False,
        min_batch_size_for_split=4,
        parallel_capture_sizes=None,
        force_split=False,
        validate_ptrs=True,
    )

    split_cfg = config["split_batch_config"]
    assert split_cfg["enabled"] is True
    assert split_cfg["mode"] == "inplace_serial"
    assert split_cfg["enable_inplace_lazy_capture"] is True
    assert split_cfg["inplace_validate_metadata_ptrs"] is True


def test_split_debug_trace_summary_validates_expected_split(tmp_path):
    trace_path = tmp_path / "split.jsonl"
    events = []
    for step_id in range(4, 7):
        events.extend([
            {
                "event": "split_slices",
                "step_id": step_id,
                "splits": [
                    {
                        "num_tokens": 384,
                        "start_num_tokens": 0
                    },
                    {
                        "num_tokens": 32,
                        "start_num_tokens": 384
                    },
                ],
            },
            {
                "event": "split_descriptor",
                "step_id": step_id,
                "idx": 1,
                "actual_num_tokens": 32,
                "batch_descriptor": {
                    "num_tokens": 32,
                    "start_num_tokens": 384,
                    "graph_variant": "inplace_serial",
                },
            },
            {
                "event": "inplace_serial_execution",
                "step_id": step_id,
                "start_num_tokens": 384,
                "num_tokens": 32,
                "input_ids": {
                    "data_ptr": 11
                },
                "positions": {
                    "data_ptr": 22
                },
                "metadata": {
                    "query_start_loc": {
                        "ptr": 33
                    },
                    "seq_lens": {
                        "ptr": 44
                    },
                    "block_tables": {
                        "ptr": 66
                    },
                    "slot_mapping": {
                        "ptr": 55
                    },
                },
            },
        ])
    events.extend([
        {
            "event": "inplace_lazy_capture_complete",
            "step_id": 4,
            "batch_descriptor": {
                "num_tokens": 32,
                "start_num_tokens": 384,
                "graph_variant": "inplace_serial",
            },
        },
        {
            "event": "split_slices",
            "step_id": 7,
            "splits": [
                {
                    "num_tokens": 384,
                    "start_num_tokens": 0
                },
                {
                    "num_tokens": 31,
                    "start_num_tokens": 384
                },
            ],
        },
    ])
    with open(trace_path, "w", encoding="utf-8") as f:
        for event in events:
            f.write(json.dumps(event) + "\n")

    summary = _summarize_split_debug_trace(
        str(trace_path),
        expected_split=_expected_inplace_split(416, [256, 384, 512]),
    )

    assert summary["failures"] == []
    assert summary["observed_split"]["first_tokens"] == 384
    assert summary["observed_split"]["second_tokens"] == 32
    assert summary["expected_split_observations"]["count"] == 3
    assert summary["observed_split_histogram"] == {
        "384+32@384": 3,
        "384+31@384": 1,
    }
    assert summary["offset_graph"]["capture_count"] == 1
    assert summary["offset_graph"]["inferred_replay_count"] == 2
    assert summary["ptr_stability"]["query_start_loc"]["stable"] is True
    assert summary["ptr_stability"]["block_tables"]["stable"] is True


def test_split_debug_trace_summary_fails_with_too_few_expected_steps(tmp_path):
    trace_path = tmp_path / "split.jsonl"
    events = [
        {
            "event": "split_slices",
            "step_id": 4,
            "splits": [
                {
                    "num_tokens": 384,
                    "start_num_tokens": 0
                },
                {
                    "num_tokens": 32,
                    "start_num_tokens": 384
                },
            ],
        },
    ]
    with open(trace_path, "w", encoding="utf-8") as f:
        for event in events:
            f.write(json.dumps(event) + "\n")

    summary = _summarize_split_debug_trace(
        str(trace_path),
        expected_split=_expected_inplace_split(416, [256, 384, 512]),
    )

    assert any("fewer than 3" in failure for failure in summary["failures"])
