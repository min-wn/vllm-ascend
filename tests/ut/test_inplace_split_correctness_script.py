import json

from examples.test_split_batch_correctness_npu import (
    _apply_capture_sizes,
    _build_split_additional_config,
    _ensure_fixed_batch_graph_capacity,
    _expected_inplace_split,
    _summarize_split_debug_trace,
    create_parser,
)


def test_expected_inplace_split_for_phase10_fixed_batch():
    assert _expected_inplace_split(416, [256, 384, 512]) == {
        "total_tokens": 416,
        "first_tokens": 384,
        "second_tokens": 32,
        "first_start_num_tokens": 0,
        "second_start_num_tokens": 384,
    }


def test_expected_inplace_split_for_spec_decode_fixed_batch_tokens():
    assert _expected_inplace_split(104 * 4, [256, 384, 512]) == {
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


def test_parser_defaults_and_overrides_fixed_batch_query_len():
    parser = create_parser()

    assert parser.parse_args([]).fixed_batch_query_len == 1
    assert parser.parse_args([]).pa_shape_list is None
    assert parser.parse_args([
        "--fixed-batch-query-len",
        "4",
    ]).fixed_batch_query_len == 4
    assert parser.parse_args(["--pa-shape-list", "384,512"]).pa_shape_list == (
        "384,512")


def test_build_split_config_keeps_inplace_feature_flags_disabled_by_default():
    config = _build_split_additional_config(
        enabled=True,
        split_mode="inplace_serial",
        num_splits=2,
        enable_parallel_streams=False,
        min_batch_size_for_split=4,
    )

    split_cfg = config["split_batch_config"]
    assert split_cfg["enable_inplace_spec_decode"] is False
    assert split_cfg["enable_inplace_mrope"] is False


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
        enable_inplace_spec_decode=True,
        enable_inplace_mrope=True,
        pa_shape_list=[512],
    )

    split_cfg = config["split_batch_config"]
    assert split_cfg["enabled"] is True
    assert split_cfg["mode"] == "inplace_serial"
    assert split_cfg["enable_inplace_lazy_capture"] is True
    assert split_cfg["inplace_validate_metadata_ptrs"] is True
    assert split_cfg["enable_inplace_spec_decode"] is True
    assert split_cfg["enable_inplace_mrope"] is True
    assert config["pa_shape_list"] == [512]


def test_build_split_config_for_macro_graph_disables_lazy_capture():
    macro_graph_config = {
        "enabled": True,
        "capture_plans": [{
            "total_tokens": 427,
            "split_actual_tokens": [224, 203],
            "split_graph_tokens": [224, 224],
        }],
    }

    config = _build_split_additional_config(
        enabled=True,
        split_mode="inplace_parallel",
        num_splits=2,
        enable_parallel_streams=True,
        min_batch_size_for_split=1,
        inplace_split_planner_policy="macro_cube_balanced",
        macro_graph_config=macro_graph_config,
    )

    split_cfg = config["split_batch_config"]
    assert split_cfg["enable_inplace_lazy_capture"] is False
    assert split_cfg["inplace_split_planner_policy"] == "macro_cube_balanced"
    assert split_cfg["macro_graph_config"] == macro_graph_config


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
                    "data_ptr": 22,
                    "shape": [3, 32],
                    "ndim": 2,
                    "storage_offset": 384,
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
    assert summary["ptr_stability"]["positions"]["shapes"] == [[3, 32]]
    assert summary["ptr_stability"]["positions"]["ndims"] == [2]
    assert summary["ptr_stability"]["positions"]["storage_offsets"] == [384]


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


def test_split_debug_trace_summary_validates_expected_no_split_reason(
        tmp_path):
    trace_path = tmp_path / "split_fallback.jsonl"
    events = [
        {
            "event": "split_planner_decision",
            "step_id": 1,
            "decision": "no_split",
            "mode": "inplace_serial",
            "reason": "no_split_mrope",
            "dry_run": True,
            "fallback_to": "no_split",
        },
        {
            "event": "split_planner_decision",
            "step_id": 2,
            "decision": "no_split",
            "mode": "inplace_serial",
            "reason": "no_split_mrope",
            "dry_run": True,
            "fallback_to": "no_split",
        },
    ]
    with open(trace_path, "w", encoding="utf-8") as f:
        for event in events:
            f.write(json.dumps(event) + "\n")

    summary = _summarize_split_debug_trace(
        str(trace_path),
        expected_split=None,
        split_mode="inplace_serial",
        expected_no_split_reason="no_split_mrope",
    )

    assert summary["failures"] == []
    assert summary["fallback_reason_histogram"] == {"no_split_mrope": 2}
    assert summary["expected_no_split_reason"] == "no_split_mrope"


def test_split_debug_trace_summary_validates_spec_decode_attn_state_fallback(
        tmp_path):
    trace_path = tmp_path / "split_spec_attn_state_fallback.jsonl"
    event = {
        "event": "split_planner_decision",
        "step_id": 1,
        "decision": "no_split",
        "mode": "inplace_serial",
        "reason": "no_split_spec_decode_attn_state",
        "dry_run": True,
        "fallback_to": "no_split",
    }
    with open(trace_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")

    summary = _summarize_split_debug_trace(
        str(trace_path),
        expected_split=None,
        split_mode="inplace_serial",
        expected_no_split_reason="no_split_spec_decode_attn_state",
    )

    assert summary["failures"] == []
    assert summary["fallback_reason_histogram"] == {
        "no_split_spec_decode_attn_state": 1
    }


def test_split_debug_trace_summary_fails_no_split_when_inplace_exec_seen(
        tmp_path):
    trace_path = tmp_path / "split_fallback_bad.jsonl"
    events = [
        {
            "event": "split_planner_decision",
            "step_id": 1,
            "decision": "no_split",
            "mode": "inplace_serial",
            "reason": "no_split_mla",
            "dry_run": True,
            "fallback_to": "no_split",
        },
        {
            "event": "inplace_serial_execution",
            "step_id": 1,
            "start_num_tokens": 384,
            "num_tokens": 32,
        },
    ]
    with open(trace_path, "w", encoding="utf-8") as f:
        for event in events:
            f.write(json.dumps(event) + "\n")

    summary = _summarize_split_debug_trace(
        str(trace_path),
        expected_split=None,
        split_mode="inplace_serial",
        expected_no_split_reason="no_split_mla",
    )

    assert any("inplace execution observed" in failure
               for failure in summary["failures"])


def test_split_debug_trace_summary_fails_missing_no_split_reason(tmp_path):
    trace_path = tmp_path / "split_fallback_missing.jsonl"
    event = {
        "event": "split_planner_decision",
        "step_id": 1,
        "decision": "no_split",
        "mode": "inplace_serial",
        "reason": "no_split_mla",
        "dry_run": True,
        "fallback_to": "no_split",
    }
    with open(trace_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")

    summary = _summarize_split_debug_trace(
        str(trace_path),
        expected_split=None,
        split_mode="inplace_serial",
        expected_no_split_reason="no_split_mrope",
    )

    assert any("expected no-split fallback reason not observed" in failure
               for failure in summary["failures"])


def test_split_debug_trace_summary_validates_inplace_parallel_gate(tmp_path):
    trace_path = tmp_path / "split_parallel.jsonl"
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
                "in_parallel_streams": True,
                "batch_descriptor": {
                    "num_tokens": 32,
                    "start_num_tokens": 384,
                    "graph_variant": "inplace_parallel",
                },
            },
            {
                "event": "inplace_parallel_execution",
                "step_id": step_id,
                "stream": "parallel",
                "buffer_source": "original_offset_view",
                "graph_entry_pool": "parallel",
                "graph_params_pool": "parallel",
                "start_num_tokens": 384,
                "num_tokens": 32,
                "input_ids": {
                    "data_ptr": 111
                },
                "positions": {
                    "data_ptr": 222
                },
                "metadata": {
                    "query_start_loc": {
                        "ptr": 333
                    },
                    "seq_lens": {
                        "ptr": 444
                    },
                    "block_tables": {
                        "ptr": 666
                    },
                    "slot_mapping": {
                        "ptr": 555
                    },
                },
            },
        ])
    events.append({
        "event": "inplace_lazy_capture_complete",
        "step_id": 4,
        "batch_descriptor": {
            "num_tokens": 32,
            "start_num_tokens": 384,
            "graph_variant": "inplace_parallel",
        },
    })
    with open(trace_path, "w", encoding="utf-8") as f:
        for event in events:
            f.write(json.dumps(event) + "\n")

    summary = _summarize_split_debug_trace(
        str(trace_path),
        expected_split=_expected_inplace_split(416, [256, 384, 512]),
        split_mode="inplace_parallel",
    )

    assert summary["failures"] == []
    assert summary["parallel_gate"] == {
        "descriptor_parallel_stream_count": 3,
        "execution_parallel_stream_count": 3,
        "original_offset_view_count": 3,
        "parallel_graph_entry_pool_count": 3,
        "parallel_graph_params_pool_count": 3,
    }


def test_split_debug_trace_summary_fails_inplace_parallel_without_pools(
        tmp_path):
    trace_path = tmp_path / "split_parallel_bad_pool.jsonl"
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
                "in_parallel_streams": True,
                "batch_descriptor": {
                    "num_tokens": 32,
                    "start_num_tokens": 384,
                    "graph_variant": "inplace_parallel",
                },
            },
            {
                "event": "inplace_parallel_execution",
                "step_id": step_id,
                "stream": "parallel",
                "buffer_source": "original_offset_view",
                "graph_entry_pool": "main",
                "graph_params_pool": "main",
                "start_num_tokens": 384,
                "num_tokens": 32,
                "input_ids": {
                    "data_ptr": 111
                },
                "positions": {
                    "data_ptr": 222
                },
                "metadata": {
                    "query_start_loc": {
                        "ptr": 333
                    },
                    "seq_lens": {
                        "ptr": 444
                    },
                    "block_tables": {
                        "ptr": 666
                    },
                    "slot_mapping": {
                        "ptr": 555
                    },
                },
            },
        ])
    events.append({
        "event": "inplace_lazy_capture_complete",
        "step_id": 4,
        "batch_descriptor": {
            "num_tokens": 32,
            "start_num_tokens": 384,
            "graph_variant": "inplace_parallel",
        },
    })
    with open(trace_path, "w", encoding="utf-8") as f:
        for event in events:
            f.write(json.dumps(event) + "\n")

    summary = _summarize_split_debug_trace(
        str(trace_path),
        expected_split=_expected_inplace_split(416, [256, 384, 512]),
        split_mode="inplace_parallel",
    )

    assert any("parallel graph entry pool" in failure
               for failure in summary["failures"])
    assert any("parallel GraphParams pool" in failure
               for failure in summary["failures"])
