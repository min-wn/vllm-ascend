from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm.v1.worker.ubatch_utils import UBatchSlice
from vllm.config import CUDAGraphMode
from vllm.forward_context import BatchDescriptor

from vllm_ascend.attention.utils import (
    AscendCommonAttentionMetadata,
    split_attn_metadata,
    stabilize_inplace_common_attn_metadata,
    using_paged_attention,
)
from vllm_ascend.worker.ubatch_utils import create_inplace_split_batch_slices


def _buffer(*shape: int, dtype: torch.dtype) -> SimpleNamespace:
    return SimpleNamespace(gpu=torch.empty(*shape, dtype=dtype),
                           cpu=torch.empty(*shape, dtype=dtype))


def _common_metadata(num_reqs: int = 416) -> AscendCommonAttentionMetadata:
    query_start_loc = torch.arange(num_reqs + 1, dtype=torch.int32)
    seq_lens = torch.arange(1000, 1000 + num_reqs, dtype=torch.int32)
    return AscendCommonAttentionMetadata(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc.clone(),
        seq_lens=seq_lens,
        seq_lens_cpu=seq_lens.clone(),
        num_computed_tokens_cpu=torch.arange(num_reqs, dtype=torch.int32),
        num_reqs=num_reqs,
        num_actual_tokens=num_reqs,
        max_query_len=1,
        decode_token_per_req=1,
        block_table_tensor=torch.arange(num_reqs * 2,
                                        dtype=torch.int32).reshape(
                                            num_reqs, 2),
        slot_mapping=torch.arange(num_reqs, dtype=torch.int64),
        actual_seq_lengths_q=[],
        positions=torch.arange(num_reqs, dtype=torch.int64),
    )


def _spec_common_metadata(num_reqs: int = 104,
                          query_len: int = 4) -> AscendCommonAttentionMetadata:
    total_tokens = num_reqs * query_len
    query_start_loc = torch.arange(0,
                                   total_tokens + query_len,
                                   query_len,
                                   dtype=torch.int32)
    seq_lens = torch.arange(1000, 1000 + num_reqs, dtype=torch.int32)
    return AscendCommonAttentionMetadata(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc.clone(),
        seq_lens=seq_lens,
        seq_lens_cpu=seq_lens.clone(),
        num_computed_tokens_cpu=torch.arange(num_reqs, dtype=torch.int32),
        num_reqs=num_reqs,
        num_actual_tokens=total_tokens,
        max_query_len=query_len,
        decode_token_per_req=query_len,
        block_table_tensor=torch.arange(num_reqs * 2,
                                        dtype=torch.int32).reshape(
                                            num_reqs, 2),
        slot_mapping=torch.arange(total_tokens, dtype=torch.int64),
        actual_seq_lengths_q=list(range(query_len, total_tokens + 1,
                                        query_len)),
        positions=torch.arange(total_tokens, dtype=torch.int64),
    )


def _split_metadata() -> list[AscendCommonAttentionMetadata]:
    common = _common_metadata()
    plan, reason = create_inplace_split_batch_slices(
        np.ones(416, dtype=np.int32),
        total_num_tokens=416,
        uniform_decode_query_len=1,
        cudagraph_capture_sizes={256, 384, 512},
    )
    assert plan is not None, reason
    return split_attn_metadata(
        [UBatchSlice(s.request_slice, s.token_slice) for s in plan.split_slices],
        common,
        max_num_tokens=416,
    )


def test_split_attn_metadata_preserves_spec_decode_request_alignment():
    common = _spec_common_metadata()
    plan, reason = create_inplace_split_batch_slices(
        np.full(104, 4, dtype=np.int32),
        total_num_tokens=416,
        uniform_decode_query_len=4,
        cudagraph_capture_sizes={256, 384, 512},
    )
    assert plan is not None, reason

    first, second = split_attn_metadata(
        [UBatchSlice(s.request_slice, s.token_slice) for s in plan.split_slices],
        common,
        max_num_tokens=416,
    )

    assert first.num_reqs == 96
    assert first.num_actual_tokens == 384
    assert first.max_query_len == 4
    assert first.decode_token_per_req == 4
    assert first.actual_seq_lengths_q[-1] == 416
    assert second.num_reqs == 8
    assert second.num_actual_tokens == 32
    assert second.max_query_len == 4
    assert second.decode_token_per_req == 4
    assert second.query_start_loc_cpu.tolist() == [
        0, 4, 8, 12, 16, 20, 24, 28, 32
    ]
    assert second.actual_seq_lengths_q[-1] == 416


def test_stabilize_inplace_common_metadata_copies_second_split_to_buffers():
    _, second = _split_metadata()
    query_start_loc_secondary = _buffer(417, dtype=torch.int32)
    seq_lens_secondary = _buffer(416, dtype=torch.int32)
    slot_mapping_secondary = _buffer(416, dtype=torch.int32)
    block_table_secondary = _buffer(416, 2, dtype=torch.int32)

    stable = stabilize_inplace_common_attn_metadata(
        second,
        split_idx=1,
        max_num_reqs=416,
        max_num_tokens=416,
        query_start_loc_secondary=query_start_loc_secondary,
        seq_lens_secondary=seq_lens_secondary,
        slot_mapping_secondary=slot_mapping_secondary,
        block_table_secondary=block_table_secondary,
    )

    assert torch.equal(stable.query_start_loc, second.query_start_loc)
    assert torch.equal(stable.query_start_loc_cpu, second.query_start_loc_cpu)
    assert torch.equal(stable.seq_lens, second.seq_lens)
    assert torch.equal(stable.seq_lens_cpu, second.seq_lens_cpu)
    assert torch.equal(stable.slot_mapping, second.slot_mapping)
    assert torch.equal(stable.block_table_tensor, second.block_table_tensor)
    assert stable.query_start_loc.data_ptr(
    ) == query_start_loc_secondary.gpu[:33].data_ptr()
    assert stable.query_start_loc_cpu.data_ptr(
    ) == query_start_loc_secondary.cpu[:33].data_ptr()
    assert stable.seq_lens.data_ptr() == seq_lens_secondary.gpu[:32].data_ptr()
    assert stable.seq_lens_cpu.data_ptr(
    ) == seq_lens_secondary.cpu[:32].data_ptr()
    assert stable.slot_mapping.data_ptr(
    ) == slot_mapping_secondary.gpu[:32].data_ptr()
    assert stable.block_table_tensor.data_ptr(
    ) == block_table_secondary.gpu[:32, :2].data_ptr()
    assert stable.positions.data_ptr() == second.positions.data_ptr()


def test_stabilize_inplace_common_metadata_keeps_second_split_ptrs_stable():
    _, second = _split_metadata()
    query_start_loc_secondary = _buffer(417, dtype=torch.int32)
    seq_lens_secondary = _buffer(416, dtype=torch.int32)
    slot_mapping_secondary = _buffer(416, dtype=torch.int32)
    block_table_secondary = _buffer(416, 2, dtype=torch.int32)

    first_stable = stabilize_inplace_common_attn_metadata(
        second,
        split_idx=1,
        max_num_reqs=416,
        max_num_tokens=416,
        query_start_loc_secondary=query_start_loc_secondary,
        seq_lens_secondary=seq_lens_secondary,
        slot_mapping_secondary=slot_mapping_secondary,
        block_table_secondary=block_table_secondary,
    )
    second_stable = stabilize_inplace_common_attn_metadata(
        second,
        split_idx=1,
        max_num_reqs=416,
        max_num_tokens=416,
        query_start_loc_secondary=query_start_loc_secondary,
        seq_lens_secondary=seq_lens_secondary,
        slot_mapping_secondary=slot_mapping_secondary,
        block_table_secondary=block_table_secondary,
    )

    assert first_stable.query_start_loc.data_ptr(
    ) == second_stable.query_start_loc.data_ptr()
    assert first_stable.seq_lens.data_ptr() == second_stable.seq_lens.data_ptr()
    assert first_stable.slot_mapping.data_ptr(
    ) == second_stable.slot_mapping.data_ptr()
    assert first_stable.block_table_tensor.data_ptr(
    ) == second_stable.block_table_tensor.data_ptr()


def test_stabilize_inplace_common_metadata_leaves_first_split_unchanged():
    first, _ = _split_metadata()

    stable = stabilize_inplace_common_attn_metadata(
        first,
        split_idx=0,
        max_num_reqs=416,
        max_num_tokens=416,
        query_start_loc_secondary=_buffer(417, dtype=torch.int32),
        seq_lens_secondary=_buffer(416, dtype=torch.int32),
        slot_mapping_secondary=_buffer(416, dtype=torch.int32),
    )

    assert stable is first


def test_stabilize_inplace_common_metadata_checks_capacity():
    _, second = _split_metadata()

    with pytest.raises(ValueError, match="nreq overflow"):
        stabilize_inplace_common_attn_metadata(
            second,
            split_idx=1,
            max_num_reqs=16,
            max_num_tokens=416,
            query_start_loc_secondary=_buffer(17, dtype=torch.int32),
            seq_lens_secondary=_buffer(16, dtype=torch.int32),
            slot_mapping_secondary=_buffer(416, dtype=torch.int32),
        )

    with pytest.raises(ValueError, match="ntok overflow"):
        stabilize_inplace_common_attn_metadata(
            second,
            split_idx=1,
            max_num_reqs=416,
            max_num_tokens=16,
            query_start_loc_secondary=_buffer(417, dtype=torch.int32),
            seq_lens_secondary=_buffer(416, dtype=torch.int32),
            slot_mapping_secondary=_buffer(16, dtype=torch.int32),
        )


def test_using_paged_attention_offset_descriptor_follows_shape_list(monkeypatch):
    context = SimpleNamespace(
        batch_descriptor=BatchDescriptor(num_tokens=32,
                                         num_reqs=32,
                                         uniform=True,
                                         has_lora=False,
                                         start_num_tokens=384))
    config = SimpleNamespace(
        speculative_config=None,
        compilation_config=SimpleNamespace(
            cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY))

    monkeypatch.setattr("vllm_ascend.attention.utils.get_ascend_device_type",
                        lambda: object())
    monkeypatch.setattr("vllm_ascend.attention.utils.get_ascend_config",
                        lambda: SimpleNamespace(pa_shape_list=[]))

    assert using_paged_attention(32, config, context) is False


def test_using_paged_attention_full_decode_descriptor_follows_shape_list(
        monkeypatch):
    context = SimpleNamespace(
        with_prefill=False,
        batch_descriptor=BatchDescriptor(num_tokens=512,
                                         num_reqs=512,
                                         uniform=True,
                                         has_lora=False))
    config = SimpleNamespace(
        speculative_config=None,
        compilation_config=SimpleNamespace(
            cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY))

    monkeypatch.setattr("vllm_ascend.attention.utils.get_ascend_device_type",
                        lambda: object())
    monkeypatch.setattr("vllm_ascend.attention.utils.get_ascend_config",
                        lambda: SimpleNamespace(pa_shape_list=[512]))

    assert using_paged_attention(512, config, context) is True


def test_using_paged_attention_honors_forced_backend(monkeypatch):
    config = SimpleNamespace(
        speculative_config=None,
        compilation_config=SimpleNamespace(
            cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY))

    monkeypatch.setattr("vllm_ascend.attention.utils.get_ascend_device_type",
                        lambda: object())
    monkeypatch.setattr("vllm_ascend.attention.utils.get_ascend_config",
                        lambda: SimpleNamespace(pa_shape_list=[]))

    assert using_paged_attention(
        32, config,
        SimpleNamespace(forced_attention_backend="pa")) is True
    assert using_paged_attention(
        512, config,
        SimpleNamespace(forced_attention_backend="fia")) is False
