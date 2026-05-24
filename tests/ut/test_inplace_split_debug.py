import json

import torch
from vllm.forward_context import BatchDescriptor

from vllm_ascend import inplace_split_debug as split_debug
from vllm_ascend.worker.ubatch_utils import SplitBatchSlice


def test_log_event_disabled_does_not_create_file(tmp_path, monkeypatch):
    path = tmp_path / "disabled.jsonl"
    monkeypatch.delenv("VLLM_ASCEND_SPLIT_INPLACE_DEBUG", raising=False)
    monkeypatch.setenv("VLLM_ASCEND_SPLIT_INPLACE_DEBUG_FILE", str(path))

    split_debug.log_event("disabled", {"value": 1}, step_id=7)

    assert not path.exists()


def test_log_event_writes_jsonl_with_tensor_info(tmp_path, monkeypatch):
    path = tmp_path / "enabled.jsonl"
    monkeypatch.setenv("VLLM_ASCEND_SPLIT_INPLACE_DEBUG", "1")
    monkeypatch.setenv("VLLM_ASCEND_SPLIT_INPLACE_DEBUG_FILE", str(path))

    tensor = torch.arange(8, dtype=torch.int32).reshape(2, 4)[:, 1:]
    split_debug.log_event(
        "tensor_event",
        {"tensor": split_debug.tensor_info(tensor)},
        step_id=42,
    )

    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(records) == 1
    assert records[0]["event"] == "tensor_event"
    assert records[0]["step_id"] == 42
    assert records[0]["tensor"]["shape"] == [2, 3]
    assert records[0]["tensor"]["dtype"] == "torch.int32"
    assert records[0]["tensor"]["storage_offset"] == 1


def test_tensor_view_info_uses_data_ptr_key():
    tensor = torch.arange(8, dtype=torch.int32)[3:6]

    info = split_debug.tensor_view_info(tensor)

    assert info is not None
    assert "ptr" not in info
    assert info["data_ptr"] == tensor.data_ptr()
    assert info["shape"] == [3]
    assert info["storage_offset"] == 3
    assert info["stride"] == [1]


def test_tensor_view_info_none():
    assert split_debug.tensor_view_info(None) is None


def test_metadata_tensor_info_finds_common_metadata_tensors():
    class Metadata:
        pass

    metadata = Metadata()
    metadata.query_start_loc = torch.arange(3, dtype=torch.int32)
    metadata.seq_lens = torch.arange(2, dtype=torch.int32)
    metadata.block_table_tensor = torch.zeros((2, 4), dtype=torch.int32)
    metadata.slot_mapping = torch.arange(2, dtype=torch.int64)
    metadata.positions = torch.arange(2, dtype=torch.int64)

    info = split_debug.metadata_tensor_info({"layer": metadata})

    assert info["query_start_loc"]["shape"] == [3]
    assert info["seq_lens"]["shape"] == [2]
    assert info["block_table_tensor"]["shape"] == [2, 4]
    assert info["slot_mapping"]["dtype"] == "torch.int64"
    assert info["positions"]["shape"] == [2]


def test_common_metadata_tensor_info_finds_cpu_and_gpu_fields():
    class CommonMetadata:
        pass

    common = CommonMetadata()
    common.query_start_loc = torch.arange(3, dtype=torch.int32)
    common.query_start_loc_cpu = torch.arange(3, dtype=torch.int32)
    common.seq_lens = torch.arange(2, dtype=torch.int32)
    common.seq_lens_cpu = torch.arange(2, dtype=torch.int32)
    common.block_table_tensor = torch.zeros((2, 4), dtype=torch.int32)
    common.slot_mapping = torch.arange(2, dtype=torch.int64)
    common.num_computed_tokens_cpu = torch.arange(2, dtype=torch.int32)
    common.positions = torch.arange(2, dtype=torch.int64)

    info = split_debug.common_metadata_tensor_info(common)

    assert info["query_start_loc"]["shape"] == [3]
    assert info["query_start_loc_cpu"]["shape"] == [3]
    assert info["seq_lens"]["shape"] == [2]
    assert info["seq_lens_cpu"]["shape"] == [2]
    assert info["block_table_tensor"]["shape"] == [2, 4]
    assert info["slot_mapping"]["dtype"] == "torch.int64"
    assert info["num_computed_tokens_cpu"]["shape"] == [2]
    assert info["positions"]["shape"] == [2]


def test_batch_descriptor_info_includes_start_num_tokens():
    desc = BatchDescriptor(num_tokens=32,
                           num_reqs=32,
                           uniform=True,
                           start_num_tokens=384)

    info = split_debug.batch_descriptor_info(desc)

    assert info["num_tokens"] == 32
    assert info["num_reqs"] == 32
    assert info["uniform"] is True
    assert info["start_num_tokens"] == 384


def test_split_slices_info_includes_inplace_offsets():
    split_slices = [
        SplitBatchSlice(
            request_slice=slice(0, 384),
            token_slice=slice(0, 384),
            padded_num_tokens=384,
            start_num_tokens=0,
        ),
        SplitBatchSlice(
            request_slice=slice(384, 416),
            token_slice=slice(384, 416),
            padded_num_tokens=32,
            start_num_tokens=384,
        ),
    ]

    info = split_debug.split_slices_info(split_slices)

    assert info[0]["graph_num_tokens"] == 384
    assert info[0]["start_num_tokens"] == 0
    assert info[1]["graph_num_tokens"] == 32
    assert info[1]["start_num_tokens"] == 384


def test_log_event_uses_current_step_id(tmp_path, monkeypatch):
    path = tmp_path / "current_step.jsonl"
    monkeypatch.setenv("VLLM_ASCEND_SPLIT_INPLACE_DEBUG", "1")
    monkeypatch.setenv("VLLM_ASCEND_SPLIT_INPLACE_DEBUG_FILE", str(path))

    split_debug.set_current_step_id(99)
    try:
        split_debug.log_event("current_step")
    finally:
        split_debug.set_current_step_id(None)

    record = json.loads(path.read_text().splitlines()[0])
    assert record["step_id"] == 99
