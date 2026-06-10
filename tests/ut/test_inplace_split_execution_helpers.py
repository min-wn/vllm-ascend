import numpy as np
import torch
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch
from vllm.config import CUDAGraphMode
from vllm.forward_context import BatchDescriptor
from vllm.v1.worker.ubatch_utils import UBatchSlice

from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.utils import (AscendCommonAttentionMetadata,
                                         split_attn_metadata)
from vllm_ascend.worker.model_runner_v3 import (
    NPUModelRunner,
    _inplace_split_precheck_reason,
    _inplace_plan_to_execution_slices,
    _template_fia_seq_lens_list,
)
from vllm_ascend.worker.ubatch_utils import create_inplace_split_batch_slices
from vllm_ascend.worker.ubatch_utils import SplitBatchSlice


def _plan_416():
    plan, reason = create_inplace_split_batch_slices(
        np.ones(416, dtype=np.int32),
        total_num_tokens=416,
        uniform_decode_query_len=1,
        cudagraph_capture_sizes={256, 384, 512},
    )
    assert plan is not None, reason
    return plan


def _plan_249_bucket():
    plan, reason = create_inplace_split_batch_slices(
        np.ones(249, dtype=np.int32),
        total_num_tokens=249,
        uniform_decode_query_len=1,
        cudagraph_capture_sizes={128, 256},
        offset_match_policy="bucket",
        offset_capture_sizes={64, 128},
    )
    assert plan is not None, reason
    return plan


def _precheck_reason(**overrides):
    kwargs = {
        "split_enabled": True,
        "split_mode": "inplace_serial",
        "enable_parallel_streams": False,
        "use_aclgraph": True,
        "num_splits": 2,
        "uniform_decode": True,
        "enable_dbo": False,
        "with_prefill": False,
        "attn_state": AscendAttentionState.DecodeOnly,
        "has_spec_decode_tokens": False,
        "enable_spec_decode": False,
        "has_lora": False,
        "uses_mrope": False,
        "enable_mrope": False,
        "use_mla": False,
        "pcp_size": 1,
        "dcp_size": 1,
    }
    kwargs.update(overrides)
    return _inplace_split_precheck_reason(**kwargs)


def _common_decode_metadata(num_reqs: int,
                            *,
                            max_tokens: int | None = None
                            ) -> AscendCommonAttentionMetadata:
    if max_tokens is None:
        max_tokens = num_reqs
    max_tokens = max(max_tokens, num_reqs)
    query_start_loc = torch.arange(num_reqs + 1, dtype=torch.int32)
    seq_lens_full = torch.arange(100, 100 + max_tokens, dtype=torch.int32)
    seq_lens_cpu = seq_lens_full[:num_reqs]
    seq_lens = seq_lens_full.clone()[:num_reqs]
    block_table_full = torch.arange(
        max_tokens * 4, dtype=torch.int32).view(max_tokens, 4)
    block_table = block_table_full[:num_reqs]
    num_computed_tokens_cpu = torch.arange(max_tokens,
                                           dtype=torch.int32)[:num_reqs]
    slot_mapping = torch.arange(max_tokens, dtype=torch.int32)[:num_reqs]
    return AscendCommonAttentionMetadata(
        query_start_loc=query_start_loc.clone(),
        query_start_loc_cpu=query_start_loc.clone(),
        seq_lens=seq_lens,
        seq_lens_cpu=seq_lens_cpu,
        num_computed_tokens_cpu=num_computed_tokens_cpu,
        num_reqs=num_reqs,
        num_actual_tokens=num_reqs,
        max_query_len=1,
        decode_token_per_req=1,
        block_table_tensor=block_table,
        slot_mapping=slot_mapping,
        actual_seq_lengths_q=list(range(1, max_tokens + 1)),
        positions=torch.arange(max_tokens, dtype=torch.int64),
        attn_state=AscendAttentionState.DecodeOnly,
        num_input_tokens=num_reqs,
    )


def test_inplace_split_precheck_accepts_supported_decode_only():
    assert _precheck_reason() is None


def test_inplace_split_precheck_rejects_p12_unsupported_cases():
    cases = [
        ({"uniform_decode": False}, "no_split_non_uniform_decode"),
        ({"with_prefill": True}, "no_split_prefill_or_mixed"),
        ({"attn_state": AscendAttentionState.SpecDecoding},
         "no_split_prefill_or_mixed"),
        ({"enable_dbo": True}, "no_split_dbo_active"),
        ({"has_spec_decode_tokens": True}, "no_split_spec_decode"),
        ({"has_lora": True}, "no_split_lora"),
        ({"uses_mrope": True}, "no_split_mrope"),
        ({"use_mla": True}, "no_split_mla"),
        ({"pcp_size": 2}, "no_split_pcp_or_context_parallel"),
        ({"dcp_size": 2}, "no_split_pcp_or_context_parallel"),
    ]

    for overrides, expected in cases:
        assert _precheck_reason(**overrides) == expected


def test_inplace_split_precheck_accepts_uniform_spec_decode_when_enabled():
    assert _precheck_reason(
        has_spec_decode_tokens=True,
        enable_spec_decode=True,
        attn_state=AscendAttentionState.SpecDecoding) is None


def test_inplace_split_precheck_keeps_spec_decode_behind_flag():
    assert (_precheck_reason(
        has_spec_decode_tokens=True,
        enable_spec_decode=False,
        attn_state=AscendAttentionState.SpecDecoding) ==
            "no_split_spec_decode")


def test_inplace_split_precheck_rejects_enabled_spec_decode_attn_state():
    assert (_precheck_reason(
        has_spec_decode_tokens=True,
        enable_spec_decode=True,
        attn_state=AscendAttentionState.ChunkedPrefill) ==
            "no_split_spec_decode_attn_state")


def test_inplace_split_precheck_accepts_mrope_serial_when_enabled():
    assert _precheck_reason(uses_mrope=True, enable_mrope=True) is None


def test_inplace_split_precheck_keeps_mrope_behind_flag():
    assert (_precheck_reason(uses_mrope=True, enable_mrope=False) ==
            "no_split_mrope")


def test_inplace_split_precheck_rejects_mrope_parallel_until_verified():
    assert (_precheck_reason(
        split_mode="inplace_parallel",
        enable_parallel_streams=True,
        uses_mrope=True,
        enable_mrope=True,
    ) == "no_split_mrope_parallel")


def test_inplace_split_precheck_rejects_config_gates():
    assert (_precheck_reason(split_enabled=False) ==
            "no_split_inplace_disabled")
    assert (_precheck_reason(split_mode="inplace_parallel",
                             enable_parallel_streams=False) ==
            "no_split_parallel_streams_disabled")
    assert _precheck_reason(use_aclgraph=False) == "no_split_no_aclgraph"
    assert (_precheck_reason(num_splits=3) ==
            "no_split_num_splits_not_two")
    assert (_precheck_reason(split_mode="parallel_buffer") ==
            "no_split_not_inplace_mode")


def test_inplace_serial_plan_enables_execution_slices():
    plan = _plan_416()

    split_batch_slices, split_ubatch_slices = (
        _inplace_plan_to_execution_slices("inplace_serial", plan))

    assert split_batch_slices == plan.split_slices
    assert split_ubatch_slices == [
        UBatchSlice(slice(0, 384), slice(0, 384)),
        UBatchSlice(slice(384, 416), slice(384, 416)),
    ]


def test_inplace_parallel_plan_stays_dry_run_without_parallel_streams():
    plan = _plan_416()

    split_batch_slices, split_ubatch_slices = (
        _inplace_plan_to_execution_slices("inplace_parallel", plan))

    assert split_batch_slices is None
    assert split_ubatch_slices is None


def test_inplace_parallel_plan_enables_execution_with_parallel_streams():
    plan = _plan_416()

    split_batch_slices, split_ubatch_slices = (
        _inplace_plan_to_execution_slices(
            "inplace_parallel",
            plan,
            enable_parallel_streams=True))

    assert split_batch_slices == plan.split_slices
    assert split_ubatch_slices == [
        UBatchSlice(slice(0, 384), slice(0, 384)),
        UBatchSlice(slice(384, 416), slice(384, 416)),
    ]


def test_trim_and_merge_split_tensor_outputs():
    runner = object.__new__(NPUModelRunner)

    first = torch.arange(5)
    second = torch.arange(10, 15)

    trimmed_first = NPUModelRunner._trim_split_output(runner, first, 3)
    trimmed_second = NPUModelRunner._trim_split_output(runner, second, 2)
    merged = NPUModelRunner._merge_split_outputs(
        runner, [trimmed_first, trimmed_second])

    assert torch.equal(merged, torch.tensor([0, 1, 2, 10, 11]))


def test_inplace_serial_metadata_sets_lazy_capture_and_validation_flags():
    runner = object.__new__(NPUModelRunner)
    runner.ascend_config = SimpleNamespace(
        split_batch_config=SimpleNamespace(enable_inplace_lazy_capture=True,
                                           inplace_validate_metadata_ptrs=True))
    runner.vllm_config = SimpleNamespace()
    runner.cudagraph_dispatcher = SimpleNamespace()
    dispatch_calls = []

    def fake_dispatch(**kwargs):
        dispatch_calls.append(kwargs)
        return CUDAGraphMode.FULL, BatchDescriptor(
            num_tokens=kwargs["num_tokens"],
            num_reqs=kwargs["num_tokens"],
            uniform=kwargs["uniform_decode"],
            has_lora=kwargs["has_lora"],
            start_num_tokens=kwargs["start_num_tokens"],
            graph_variant=kwargs.get("graph_variant", ""),
            attention_backend=kwargs.get("attention_backend", ""),
            capture_metadata_mode=kwargs.get("capture_metadata_mode", ""),
        )

    runner.cudagraph_dispatcher.dispatch = fake_dispatch
    plan = _plan_416()
    split_batch_slices, split_ubatch_slices = (
        _inplace_plan_to_execution_slices("inplace_serial", plan))
    current_context = SimpleNamespace(dp_metadata=None)
    contexts = []

    def fake_create_context(cur_forward_context, **kwargs):
        context = SimpleNamespace(**kwargs)
        context.dp_metadata = kwargs.get("dp_metadata")
        contexts.append(context)
        return context

    input_ids = torch.arange(416)
    positions = torch.arange(416)

    with patch("vllm_ascend.worker.model_runner_v3.get_forward_context",
               return_value=current_context), patch(
                   "vllm_ascend.worker.model_runner_v3.create_ascend_forward_context",
                   side_effect=fake_create_context):
        metadata = NPUModelRunner._make_split_batch_metadata_inplace_serial(
            runner,
            split_ubatch_slices,
            split_batch_slices,
            attn_metadata=None,
            input_ids=input_ids,
            positions=positions,
            inputs_embeds=None,
            intermediate_tensors=None,
            batch_descriptor=BatchDescriptor(num_tokens=416,
                                             num_reqs=416,
                                             uniform=False,
                                             has_lora=False),
            aclgraph_runtime_mode=CUDAGraphMode.FULL,
            inplace_attention_backend="fia",
        )

    assert len(metadata) == 2
    assert [call["uniform_decode"] for call in dispatch_calls] == [True, True]
    assert metadata[0].input_ids.storage_offset() == 0
    assert metadata[0].positions.storage_offset() == 0
    assert metadata[1].input_ids.storage_offset() == 384
    assert metadata[1].positions.storage_offset() == 384
    assert metadata[1].input_ids.data_ptr() == input_ids[384:].data_ptr()
    assert metadata[1].positions.data_ptr() == positions[384:].data_ptr()
    assert contexts[0].allow_inplace_lazy_capture is False
    assert contexts[0].forced_attention_backend == "fia"
    assert contexts[0].batch_descriptor.graph_variant == ""
    assert contexts[0].batch_descriptor.attention_backend == ""
    assert contexts[0].validate_inplace_metadata_ptrs is False
    assert contexts[1].allow_inplace_lazy_capture is True
    assert contexts[1].forced_attention_backend == "fia"
    assert contexts[1].batch_descriptor.graph_variant == "inplace_serial"
    assert contexts[1].batch_descriptor.attention_backend == "fia"
    assert contexts[1].batch_descriptor.capture_metadata_mode == "template"
    assert contexts[1].validate_inplace_input_ptrs is True
    assert contexts[1].validate_inplace_metadata_ptrs is True
    assert contexts[1].split_inplace_mode == "inplace_serial"


def test_inplace_parallel_metadata_sets_parallel_context_and_offset_views():
    runner = object.__new__(NPUModelRunner)
    runner.ascend_config = SimpleNamespace(
        split_batch_config=SimpleNamespace(enable_inplace_lazy_capture=True,
                                           inplace_validate_metadata_ptrs=True,
                                           inplace_force_pa_for_offset=False))
    runner.vllm_config = SimpleNamespace()
    runner.stream_main = object()
    runner.stream_parallel = object()
    runner.cudagraph_dispatcher = SimpleNamespace()
    dispatch_calls = []

    def fake_dispatch(**kwargs):
        dispatch_calls.append(kwargs)
        return CUDAGraphMode.FULL, BatchDescriptor(
            num_tokens=kwargs["num_tokens"],
            num_reqs=kwargs["num_tokens"],
            uniform=kwargs["uniform_decode"],
            has_lora=kwargs["has_lora"],
            start_num_tokens=kwargs["start_num_tokens"],
            graph_variant=kwargs.get("graph_variant", ""),
            attention_backend=kwargs.get("attention_backend", ""),
            capture_metadata_mode=kwargs.get("capture_metadata_mode", ""),
        )

    class FakeNPUStream:

        def __call__(self, stream):
            return nullcontext()

    runner.cudagraph_dispatcher.dispatch = fake_dispatch
    plan = _plan_416()
    split_batch_slices, split_ubatch_slices = (
        _inplace_plan_to_execution_slices(
            "inplace_parallel",
            plan,
            enable_parallel_streams=True))
    current_context = SimpleNamespace(dp_metadata=None)
    contexts = []

    def fake_create_context(cur_forward_context, **kwargs):
        context = SimpleNamespace(**kwargs)
        context.dp_metadata = kwargs.get("dp_metadata")
        contexts.append(context)
        return context

    input_ids = torch.arange(416)
    positions = torch.arange(416)

    with patch("vllm_ascend.worker.model_runner_v3.get_forward_context",
               return_value=current_context), patch(
                   "vllm_ascend.worker.model_runner_v3.create_ascend_forward_context",
                   side_effect=fake_create_context), patch(
                       "vllm_ascend.worker.model_runner_v3.torch.npu.stream",
                       new=FakeNPUStream()):
        metadata = NPUModelRunner._make_split_batch_metadata_inplace_parallel(
            runner,
            split_ubatch_slices,
            split_batch_slices,
            attn_metadata=None,
            input_ids=input_ids,
            positions=positions,
            inputs_embeds=None,
            intermediate_tensors=None,
            batch_descriptor=BatchDescriptor(num_tokens=416,
                                             num_reqs=416,
                                             uniform=False,
                                             has_lora=False),
            aclgraph_runtime_mode=CUDAGraphMode.FULL,
            inplace_attention_backend="fia",
        )

    assert len(metadata) == 2
    assert [call["uniform_decode"] for call in dispatch_calls] == [True, True]
    assert metadata[0].input_ids.storage_offset() == 0
    assert metadata[1].input_ids.storage_offset() == 384
    assert metadata[1].positions.storage_offset() == 384
    assert metadata[1].input_ids.data_ptr() == input_ids[384:].data_ptr()
    assert metadata[1].positions.data_ptr() == positions[384:].data_ptr()
    assert contexts[0].in_parallel_streams is False
    assert contexts[1].in_parallel_streams is True
    assert contexts[1].allow_inplace_lazy_capture is True
    assert contexts[1].forced_attention_backend == "fia"
    assert contexts[1].batch_descriptor.graph_variant == "inplace_parallel"
    assert contexts[1].batch_descriptor.attention_backend == "fia"
    assert contexts[1].batch_descriptor.capture_metadata_mode == "template"
    assert contexts[1].validate_inplace_input_ptrs is True
    assert contexts[1].validate_inplace_metadata_ptrs is True
    assert contexts[1].split_inplace_mode == "inplace_parallel"


def test_mixed_request_compact_inputs_copy_second_split_to_zero_offset():
    runner = object.__new__(NPUModelRunner)
    input_ids = torch.arange(130, dtype=torch.int32)
    positions = torch.arange(130, dtype=torch.int64)
    runner.input_ids = SimpleNamespace(gpu=input_ids.clone())
    runner.positions = SimpleNamespace(gpu=positions.clone())
    runner.inputs_embeds = SimpleNamespace(gpu=None)
    runner.input_ids_parallel_streams = SimpleNamespace(
        gpu=torch.full((130, ), -1, dtype=torch.int32))
    runner.positions_parallel_streams = SimpleNamespace(
        gpu=torch.full((130, ), -1, dtype=torch.int64))
    runner.inputs_embeds_parallel_streams = SimpleNamespace(gpu=None)
    split_slices = [
        SplitBatchSlice(slice(0, 3), slice(0, 66), padded_num_tokens=66),
        SplitBatchSlice(slice(3, 4), slice(66, 130), padded_num_tokens=64),
    ]

    prepared = (
        NPUModelRunner._prepare_mixed_request_split_compact_inputs(
            runner,
            split_slices,
            input_ids,
            positions,
            inputs_embeds=None,
            intermediate_tensors=None))

    assert len(prepared) == 2
    assert torch.equal(prepared[0]["input_ids"], input_ids[:66])
    assert torch.equal(prepared[0]["positions"], positions[:66])
    assert prepared[1]["input_ids"].storage_offset() == 0
    assert prepared[1]["positions"].storage_offset() == 0
    assert torch.equal(prepared[1]["input_ids"], input_ids[66:130])
    assert torch.equal(prepared[1]["positions"], positions[66:130])
    assert prepared[1]["local_ubatch_slice"] == UBatchSlice(
        slice(0, 1), slice(0, 64))


def test_mixed_request_compact_inputs_zero_pad_graph_tail():
    runner = object.__new__(NPUModelRunner)
    input_ids = torch.arange(130, dtype=torch.int32)
    positions = torch.arange(130, dtype=torch.int64)
    runner.input_ids = SimpleNamespace(
        gpu=torch.full((192, ), -1, dtype=torch.int32))
    runner.positions = SimpleNamespace(
        gpu=torch.full((192, ), -1, dtype=torch.int64))
    runner.inputs_embeds = SimpleNamespace(gpu=None)
    runner.input_ids_parallel_streams = SimpleNamespace(
        gpu=torch.full((192, ), -1, dtype=torch.int32))
    runner.positions_parallel_streams = SimpleNamespace(
        gpu=torch.full((192, ), -1, dtype=torch.int64))
    runner.inputs_embeds_parallel_streams = SimpleNamespace(gpu=None)
    split_slices = [
        SplitBatchSlice(slice(0, 3), slice(0, 66), padded_num_tokens=128),
        SplitBatchSlice(slice(3, 4), slice(66, 130), padded_num_tokens=64),
    ]

    prepared = (
        NPUModelRunner._prepare_mixed_request_split_compact_inputs(
            runner,
            split_slices,
            input_ids,
            positions,
            inputs_embeds=None,
            intermediate_tensors=None))

    assert prepared[0]["input_ids"].shape[0] == 128
    assert prepared[0]["positions"].shape[0] == 128
    assert torch.equal(prepared[0]["input_ids"][:66], input_ids[:66])
    assert torch.equal(prepared[0]["positions"][:66], positions[:66])
    assert torch.equal(prepared[0]["input_ids"][66:128],
                       torch.zeros(62, dtype=torch.int32))
    assert torch.equal(prepared[0]["positions"][66:128],
                       torch.zeros(62, dtype=torch.int64))
    assert prepared[0]["local_ubatch_slice"] == UBatchSlice(
        slice(0, 3), slice(0, 128))


def test_mixed_request_compact_inputs_snapshot_aliasing_source():
    runner = object.__new__(NPUModelRunner)
    input_ids = torch.arange(130, dtype=torch.int32)
    positions = torch.arange(130, dtype=torch.int64)
    expected_second_ids = input_ids[66:130].clone()
    expected_second_positions = positions[66:130].clone()
    runner.input_ids = SimpleNamespace(gpu=input_ids)
    runner.positions = SimpleNamespace(gpu=positions)
    runner.inputs_embeds = SimpleNamespace(gpu=None)
    runner.input_ids_parallel_streams = SimpleNamespace(
        gpu=torch.full((130, ), -1, dtype=torch.int32))
    runner.positions_parallel_streams = SimpleNamespace(
        gpu=torch.full((130, ), -1, dtype=torch.int64))
    runner.inputs_embeds_parallel_streams = SimpleNamespace(gpu=None)
    split_slices = [
        SplitBatchSlice(slice(0, 3), slice(0, 66), padded_num_tokens=128),
        SplitBatchSlice(slice(3, 4), slice(66, 130), padded_num_tokens=64),
    ]

    prepared = (
        NPUModelRunner._prepare_mixed_request_split_compact_inputs(
            runner,
            split_slices,
            input_ids,
            positions,
            inputs_embeds=None,
            intermediate_tensors=None))

    assert torch.equal(prepared[0]["input_ids"][66:128],
                       torch.zeros(62, dtype=torch.int32))
    assert torch.equal(prepared[0]["positions"][66:128],
                       torch.zeros(62, dtype=torch.int64))
    assert torch.equal(prepared[1]["input_ids"], expected_second_ids)
    assert torch.equal(prepared[1]["positions"], expected_second_positions)


def test_mixed_request_compact_metadata_padding_uses_fake_tail_requests():
    runner = object.__new__(NPUModelRunner)
    common = AscendCommonAttentionMetadata(
        query_start_loc=torch.tensor([0, 2, 66], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 2, 66], dtype=torch.int32),
        seq_lens_cpu=torch.tensor([2, 64], dtype=torch.int32),
        seq_lens=torch.tensor([2, 64], dtype=torch.int32),
        num_computed_tokens_cpu=torch.tensor([0, 0], dtype=torch.int32),
        num_reqs=2,
        num_actual_tokens=66,
        max_query_len=64,
        decode_token_per_req=1,
        block_table_tensor=torch.ones((2, 4), dtype=torch.int32),
        slot_mapping=torch.arange(66, dtype=torch.int64),
        actual_seq_lengths_q=[2, 66],
        positions=torch.arange(66, dtype=torch.int64),
        attn_mask=None,
        spec_attn_mask=None,
        attn_state=AscendAttentionState.ChunkedPrefill,
    )
    split_slice = SplitBatchSlice(
        slice(0, 2), slice(0, 66), padded_num_tokens=128)

    padded = NPUModelRunner._pad_compact_common_attn_metadata_for_graph(
        runner,
        common,
        split_slice,
        split_idx=0)

    assert padded.num_reqs == 64
    assert padded.num_actual_tokens == 128
    assert padded.num_input_tokens == 128
    assert padded.graph_pad_size == 64
    assert padded.query_start_loc_cpu.shape[0] == 65
    assert padded.query_start_loc_cpu[-1].item() == 128
    assert padded.actual_seq_lengths_q[:2] == [2, 66]
    assert padded.actual_seq_lengths_q[-1] == 128
    assert torch.equal(padded.seq_lens_cpu[:2], common.seq_lens_cpu)
    assert torch.equal(padded.seq_lens_cpu[2:],
                       torch.zeros(62, dtype=torch.int32))
    assert torch.equal(padded.slot_mapping[:66], common.slot_mapping)
    assert torch.equal(padded.slot_mapping[66:],
                       torch.full((62, ), -1, dtype=torch.int64))
    assert torch.equal(padded.block_table_tensor[:2],
                       common.block_table_tensor)
    assert torch.equal(padded.block_table_tensor[2:],
                       torch.full((62, 4), -1, dtype=torch.int32))
    assert torch.equal(padded.positions[:66], common.positions)
    assert torch.equal(padded.positions[66:],
                       torch.zeros(62, dtype=torch.int64))


def test_mixed_request_serial_metadata_reuses_piecewise_graph_keys():
    runner = object.__new__(NPUModelRunner)
    runner.vllm_config = SimpleNamespace()
    runner.input_ids = SimpleNamespace(gpu=torch.zeros(130, dtype=torch.int32))
    runner.positions = SimpleNamespace(gpu=torch.zeros(130, dtype=torch.int64))
    runner.inputs_embeds = SimpleNamespace(gpu=None)
    runner.input_ids_parallel_streams = SimpleNamespace(
        gpu=torch.zeros(130, dtype=torch.int32))
    runner.positions_parallel_streams = SimpleNamespace(
        gpu=torch.zeros(130, dtype=torch.int64))
    runner.inputs_embeds_parallel_streams = SimpleNamespace(gpu=None)
    runner.cudagraph_dispatcher = SimpleNamespace()
    dispatch_calls = []

    def fake_dispatch(**kwargs):
        dispatch_calls.append(kwargs)
        return CUDAGraphMode.PIECEWISE, BatchDescriptor(
            num_tokens=kwargs["num_tokens"],
            num_reqs=None,
            uniform=False,
            has_lora=kwargs["has_lora"],
            start_num_tokens=kwargs["start_num_tokens"],
            graph_variant=kwargs.get("graph_variant", ""),
            attention_backend=kwargs.get("attention_backend", ""),
            capture_metadata_mode=kwargs.get("capture_metadata_mode", ""),
        )

    runner.cudagraph_dispatcher.dispatch = fake_dispatch
    current_context = SimpleNamespace(dp_metadata=None)
    contexts = []

    def fake_create_context(cur_forward_context, **kwargs):
        context = SimpleNamespace(**kwargs)
        contexts.append(context)
        return context

    input_ids = torch.arange(130, dtype=torch.int32)
    positions = torch.arange(130, dtype=torch.int64)
    split_slices = [
        SplitBatchSlice(slice(0, 3), slice(0, 66), padded_num_tokens=66),
        SplitBatchSlice(slice(3, 4), slice(66, 130), padded_num_tokens=64),
    ]
    attn_metadata = [
        {"layer": SimpleNamespace(num_actual_tokens=66)},
        {"layer": SimpleNamespace(num_actual_tokens=64)},
    ]

    with patch("vllm_ascend.worker.model_runner_v3.get_forward_context",
               return_value=current_context), patch(
                   "vllm_ascend.worker.model_runner_v3.create_ascend_forward_context",
                   side_effect=fake_create_context):
        metadata = (
            NPUModelRunner._make_mixed_request_split_metadata_serial(
                runner,
                split_slices,
                attn_metadata,
                input_ids,
                positions,
                inputs_embeds=None,
                intermediate_tensors=None,
                batch_descriptor=BatchDescriptor(num_tokens=130,
                                                 num_reqs=None,
                                                 uniform=False,
                                                 has_lora=False),
            ))

    assert len(metadata) == 2
    assert [call["uniform_decode"] for call in dispatch_calls] == [
        False, False
    ]
    assert [call["disable_full"] for call in dispatch_calls] == [True, True]
    assert [call["start_num_tokens"] for call in dispatch_calls] == [0, 0]
    assert [call["allow_inplace_lazy_key"]
            for call in dispatch_calls] == [False, False]
    assert all("graph_variant" not in call for call in dispatch_calls)
    assert all("attention_backend" not in call for call in dispatch_calls)
    assert all("capture_metadata_mode" not in call for call in dispatch_calls)
    assert torch.equal(metadata[0].input_ids, input_ids[:66])
    assert metadata[1].input_ids.storage_offset() == 0
    assert torch.equal(metadata[1].input_ids, input_ids[66:130])
    assert contexts[0].ubatch_slices[0] == UBatchSlice(
        slice(0, 3), slice(0, 66))
    assert contexts[1].ubatch_slices[1] == UBatchSlice(
        slice(0, 1), slice(0, 64))
    assert contexts[0].batch_descriptor.uniform is False
    assert contexts[0].batch_descriptor.graph_variant == ""
    assert contexts[0].batch_descriptor.capture_metadata_mode == ""
    assert contexts[1].batch_descriptor.start_num_tokens == 0
    assert contexts[1].split_inplace_mode == "mixed_request_serial"


def test_mixed_request_parallel_metadata_reuses_piecewise_graph_keys():
    runner = object.__new__(NPUModelRunner)
    runner.vllm_config = SimpleNamespace()
    runner.stream_main = object()
    runner.stream_parallel = object()
    runner.input_ids = SimpleNamespace(gpu=torch.zeros(130, dtype=torch.int32))
    runner.positions = SimpleNamespace(gpu=torch.zeros(130, dtype=torch.int64))
    runner.inputs_embeds = SimpleNamespace(gpu=None)
    runner.input_ids_parallel_streams = SimpleNamespace(
        gpu=torch.zeros(130, dtype=torch.int32))
    runner.positions_parallel_streams = SimpleNamespace(
        gpu=torch.zeros(130, dtype=torch.int64))
    runner.inputs_embeds_parallel_streams = SimpleNamespace(gpu=None)
    runner.cudagraph_dispatcher = SimpleNamespace()
    dispatch_calls = []

    def fake_dispatch(**kwargs):
        dispatch_calls.append(kwargs)
        return CUDAGraphMode.PIECEWISE, BatchDescriptor(
            num_tokens=kwargs["num_tokens"],
            num_reqs=None,
            uniform=False,
            has_lora=kwargs["has_lora"],
            start_num_tokens=kwargs["start_num_tokens"],
            graph_variant=kwargs.get("graph_variant", ""),
            attention_backend=kwargs.get("attention_backend", ""),
            capture_metadata_mode=kwargs.get("capture_metadata_mode", ""),
        )

    class FakeNPUStream:

        def __call__(self, stream):
            return nullcontext()

    runner.cudagraph_dispatcher.dispatch = fake_dispatch
    current_context = SimpleNamespace(dp_metadata=None)
    contexts = []

    def fake_create_context(cur_forward_context, **kwargs):
        context = SimpleNamespace(**kwargs)
        contexts.append(context)
        return context

    input_ids = torch.arange(130, dtype=torch.int32)
    positions = torch.arange(130, dtype=torch.int64)
    split_slices = [
        SplitBatchSlice(slice(0, 3), slice(0, 66), padded_num_tokens=66),
        SplitBatchSlice(slice(3, 4), slice(66, 130), padded_num_tokens=64),
    ]
    attn_metadata = [
        {"layer": SimpleNamespace(num_actual_tokens=66)},
        {"layer": SimpleNamespace(num_actual_tokens=64)},
    ]

    with patch("vllm_ascend.worker.model_runner_v3.get_forward_context",
               return_value=current_context), patch(
                   "vllm_ascend.worker.model_runner_v3.create_ascend_forward_context",
                   side_effect=fake_create_context), patch(
                       "vllm_ascend.worker.model_runner_v3.torch.npu.stream",
                       new=FakeNPUStream()):
        metadata = (
            NPUModelRunner._make_mixed_request_split_metadata_parallel(
                runner,
                split_slices,
                attn_metadata,
                input_ids,
                positions,
                inputs_embeds=None,
                intermediate_tensors=None,
                batch_descriptor=BatchDescriptor(num_tokens=130,
                                                 num_reqs=None,
                                                 uniform=False,
                                                 has_lora=False),
            ))

    assert len(metadata) == 2
    assert [call["uniform_decode"] for call in dispatch_calls] == [
        False, False
    ]
    assert [call["disable_full"] for call in dispatch_calls] == [True, True]
    assert [call["start_num_tokens"] for call in dispatch_calls] == [0, 0]
    assert [call["allow_inplace_lazy_key"]
            for call in dispatch_calls] == [False, False]
    assert all("graph_variant" not in call for call in dispatch_calls)
    assert all("attention_backend" not in call for call in dispatch_calls)
    assert all("capture_metadata_mode" not in call for call in dispatch_calls)
    assert torch.equal(metadata[0].input_ids, input_ids[:66])
    assert metadata[1].input_ids.storage_offset() == 0
    assert torch.equal(metadata[1].input_ids, input_ids[66:130])
    assert contexts[0].in_parallel_streams is False
    assert contexts[1].in_parallel_streams is True
    assert contexts[0].ubatch_slices[0] == UBatchSlice(
        slice(0, 3), slice(0, 66))
    assert contexts[1].ubatch_slices[1] == UBatchSlice(
        slice(0, 1), slice(0, 64))
    assert contexts[1].cos_sin_slot_id == 1
    assert contexts[1].batch_descriptor.start_num_tokens == 0
    assert contexts[1].batch_descriptor.graph_variant == ""
    assert contexts[1].batch_descriptor.capture_metadata_mode == ""
    assert (contexts[1].split_inplace_mode ==
            "mixed_request_piecewise_attention_parallel")


def test_inplace_serial_metadata_uses_padded_offset_graph_view():
    runner = object.__new__(NPUModelRunner)
    runner.ascend_config = SimpleNamespace(
        split_batch_config=SimpleNamespace(enable_inplace_lazy_capture=True,
                                           inplace_validate_metadata_ptrs=True,
                                           inplace_force_pa_for_offset=False))
    runner.vllm_config = SimpleNamespace()
    runner.cudagraph_dispatcher = SimpleNamespace()
    dispatch_calls = []

    def fake_dispatch(**kwargs):
        dispatch_calls.append(kwargs)
        return CUDAGraphMode.FULL, BatchDescriptor(
            num_tokens=kwargs["num_tokens"],
            num_reqs=kwargs["num_tokens"],
            uniform=kwargs["uniform_decode"],
            has_lora=kwargs["has_lora"],
            start_num_tokens=kwargs["start_num_tokens"],
            graph_variant=kwargs.get("graph_variant", ""),
            attention_backend=kwargs.get("attention_backend", ""),
            capture_metadata_mode=kwargs.get("capture_metadata_mode", ""),
        )

    runner.cudagraph_dispatcher.dispatch = fake_dispatch
    plan = _plan_249_bucket()
    split_batch_slices, split_ubatch_slices = (
        _inplace_plan_to_execution_slices("inplace_serial", plan))
    current_context = SimpleNamespace(dp_metadata=None)
    contexts = []

    def fake_create_context(cur_forward_context, **kwargs):
        context = SimpleNamespace(**kwargs)
        context.dp_metadata = kwargs.get("dp_metadata")
        contexts.append(context)
        return context

    input_ids_full = torch.arange(256)
    positions_full = torch.arange(3 * 256).view(3, 256)
    runner.input_ids = SimpleNamespace(gpu=input_ids_full)
    runner.mrope_positions = SimpleNamespace(gpu=positions_full)
    input_ids = input_ids_full[:249]
    positions = positions_full[:, :249]

    with patch("vllm_ascend.worker.model_runner_v3.get_forward_context",
               return_value=current_context), patch(
                   "vllm_ascend.worker.model_runner_v3.create_ascend_forward_context",
                   side_effect=fake_create_context):
        metadata = NPUModelRunner._make_split_batch_metadata_inplace_serial(
            runner,
            split_ubatch_slices,
            split_batch_slices,
            attn_metadata=None,
            input_ids=input_ids,
            positions=positions,
            inputs_embeds=None,
            intermediate_tensors=None,
            batch_descriptor=BatchDescriptor(num_tokens=249,
                                             num_reqs=249,
                                             uniform=False,
                                             has_lora=False),
            aclgraph_runtime_mode=CUDAGraphMode.FULL,
            inplace_attention_backend="fia",
        )

    assert [call["num_tokens"] for call in dispatch_calls] == [128, 128]
    assert metadata[1].input_ids.shape == (128, )
    assert metadata[1].positions.shape == (3, 128)
    assert metadata[1].input_ids.storage_offset() == 128
    assert metadata[1].positions.storage_offset() == 128
    assert metadata[1].input_ids.data_ptr() == input_ids_full[
        128:256].data_ptr()
    assert metadata[1].positions.data_ptr() == positions_full[:, 128:
                                                              256].data_ptr()
    assert contexts[1].ubatch_slices[1].token_slice == slice(128, 256)
    assert split_ubatch_slices[1].token_slice == slice(128, 249)
    assert torch.equal(metadata[1].input_ids[:121], torch.arange(128, 249))
    assert torch.equal(metadata[1].input_ids[121:],
                       torch.zeros(7, dtype=input_ids.dtype))
    assert torch.equal(metadata[1].positions[:, 121:],
                       torch.zeros(3, 7, dtype=positions.dtype))
    assert metadata[1].num_tokens == 128


def test_inplace_parallel_metadata_uses_padded_offset_graph_view():
    runner = object.__new__(NPUModelRunner)
    runner.ascend_config = SimpleNamespace(
        split_batch_config=SimpleNamespace(enable_inplace_lazy_capture=True,
                                           inplace_validate_metadata_ptrs=True,
                                           inplace_force_pa_for_offset=False))
    runner.vllm_config = SimpleNamespace()
    runner.stream_main = object()
    runner.stream_parallel = object()
    runner.cudagraph_dispatcher = SimpleNamespace()
    dispatch_calls = []

    def fake_dispatch(**kwargs):
        dispatch_calls.append(kwargs)
        return CUDAGraphMode.FULL, BatchDescriptor(
            num_tokens=kwargs["num_tokens"],
            num_reqs=kwargs["num_tokens"],
            uniform=kwargs["uniform_decode"],
            has_lora=kwargs["has_lora"],
            start_num_tokens=kwargs["start_num_tokens"],
            graph_variant=kwargs.get("graph_variant", ""),
            attention_backend=kwargs.get("attention_backend", ""),
            capture_metadata_mode=kwargs.get("capture_metadata_mode", ""),
        )

    class FakeNPUStream:

        def __call__(self, stream):
            return nullcontext()

    runner.cudagraph_dispatcher.dispatch = fake_dispatch
    plan = _plan_249_bucket()
    split_batch_slices, split_ubatch_slices = (
        _inplace_plan_to_execution_slices(
            "inplace_parallel",
            plan,
            enable_parallel_streams=True))
    current_context = SimpleNamespace(dp_metadata=None)
    contexts = []

    def fake_create_context(cur_forward_context, **kwargs):
        context = SimpleNamespace(**kwargs)
        context.dp_metadata = kwargs.get("dp_metadata")
        contexts.append(context)
        return context

    input_ids_full = torch.arange(256)
    positions_full = torch.arange(256)
    runner.input_ids = SimpleNamespace(gpu=input_ids_full)
    runner.positions = SimpleNamespace(gpu=positions_full)
    input_ids = input_ids_full[:249]
    positions = positions_full[:249]

    with patch("vllm_ascend.worker.model_runner_v3.get_forward_context",
               return_value=current_context), patch(
                   "vllm_ascend.worker.model_runner_v3.create_ascend_forward_context",
                   side_effect=fake_create_context), patch(
                       "vllm_ascend.worker.model_runner_v3.torch.npu.stream",
                       new=FakeNPUStream()):
        metadata = NPUModelRunner._make_split_batch_metadata_inplace_parallel(
            runner,
            split_ubatch_slices,
            split_batch_slices,
            attn_metadata=None,
            input_ids=input_ids,
            positions=positions,
            inputs_embeds=None,
            intermediate_tensors=None,
            batch_descriptor=BatchDescriptor(num_tokens=249,
                                             num_reqs=249,
                                             uniform=False,
                                             has_lora=False),
            aclgraph_runtime_mode=CUDAGraphMode.FULL,
            inplace_attention_backend="fia",
        )

    assert [call["num_tokens"] for call in dispatch_calls] == [128, 128]
    assert metadata[1].input_ids.shape == (128, )
    assert metadata[1].positions.shape == (128, )
    assert metadata[1].input_ids.storage_offset() == 128
    assert metadata[1].positions.storage_offset() == 128
    assert metadata[1].input_ids.data_ptr() == input_ids_full[
        128:256].data_ptr()
    assert metadata[1].positions.data_ptr() == positions_full[
        128:256].data_ptr()
    assert contexts[1].ubatch_slices[1].token_slice == slice(128, 256)
    assert split_ubatch_slices[1].token_slice == slice(128, 249)
    assert torch.equal(metadata[1].input_ids[:121], torch.arange(128, 249))
    assert torch.equal(metadata[1].input_ids[121:],
                       torch.zeros(7, dtype=input_ids.dtype))
    assert torch.equal(metadata[1].positions[121:],
                       torch.zeros(7, dtype=positions.dtype))
    assert metadata[1].num_tokens == 128


def test_inplace_offset_bucket_pads_attention_metadata_to_graph_size():
    runner = object.__new__(NPUModelRunner)
    runner.uniform_decode_query_len = 1
    runner.positions = SimpleNamespace(gpu=torch.arange(256,
                                                        dtype=torch.int64))

    plan = _plan_249_bucket()
    split_ubatch_slices = [
        UBatchSlice(s.request_slice, s.token_slice)
        for s in plan.split_slices
    ]
    common = _common_decode_metadata(249, max_tokens=256)
    common_metadata_list = split_attn_metadata(split_ubatch_slices, common,
                                               256)

    padded_metadata_list = (
        NPUModelRunner._stabilize_inplace_common_attn_metadata_list(
            runner,
            common_metadata_list,
            split_mode="inplace_parallel",
            inplace_split_plan=plan,
        ))

    first, second = padded_metadata_list
    assert first.num_reqs == 128
    assert first.num_actual_tokens == 128
    assert second.num_reqs == 128
    assert second.num_actual_tokens == 128
    assert second.num_input_tokens == 128
    assert second.query_start_loc_cpu.tolist() == list(range(129))
    assert second.query_start_loc.tolist() == list(range(129))
    assert second.actual_seq_lengths_q[-1] == 128
    assert second.seq_lens_cpu.shape == (128, )
    assert second.seq_lens.shape == (128, )
    assert second.num_computed_tokens_cpu.shape == (128, )
    assert second.block_table_tensor.shape == (128, 4)
    assert second.slot_mapping.shape == (128, )
    assert torch.equal(second.seq_lens_cpu[121:],
                       torch.zeros(7, dtype=torch.int32))
    assert torch.equal(second.seq_lens[121:],
                       torch.zeros(7, dtype=torch.int32))
    assert torch.equal(second.num_computed_tokens_cpu[121:],
                       torch.zeros(7, dtype=torch.int32))
    assert torch.equal(second.block_table_tensor[121:],
                       torch.zeros((7, 4), dtype=torch.int32))
    assert torch.equal(second.slot_mapping[121:],
                       torch.zeros(7, dtype=torch.int32))
    assert torch.equal(second.positions[:121], torch.arange(128, 249))
    assert torch.equal(second.positions[121:],
                       torch.zeros(7, dtype=torch.int64))


def test_template_fia_seq_lens_list_sets_tail_to_target_t():
    attn_metadata = {
        "layer.0": SimpleNamespace(seq_lens_list=[9, 9]),
        "layer.1": SimpleNamespace(seq_lens_list=[10, 9]),
        "ignored": SimpleNamespace(seq_lens_list=[]),
    }

    updated = _template_fia_seq_lens_list(attn_metadata, target_t=32)

    assert updated == 2
    assert attn_metadata["layer.0"].seq_lens_list == [9, 32]
    assert attn_metadata["layer.1"].seq_lens_list == [10, 32]
    assert attn_metadata["ignored"].seq_lens_list == []


def test_inplace_serial_uses_fia_template_for_offset_metadata():
    runner = object.__new__(NPUModelRunner)
    runner.ascend_config = SimpleNamespace(
        split_batch_config=SimpleNamespace(enable_inplace_lazy_capture=True,
                                           inplace_validate_metadata_ptrs=True))
    runner.vllm_config = SimpleNamespace()
    runner.block_size = 256
    runner.cudagraph_dispatcher = SimpleNamespace()

    def fake_dispatch(**kwargs):
        return CUDAGraphMode.FULL, BatchDescriptor(
            num_tokens=kwargs["num_tokens"],
            num_reqs=kwargs["num_tokens"],
            uniform=kwargs["uniform_decode"],
            has_lora=kwargs["has_lora"],
            start_num_tokens=kwargs["start_num_tokens"],
            graph_variant=kwargs.get("graph_variant", ""),
            attention_backend=kwargs.get("attention_backend", ""),
            capture_metadata_mode=kwargs.get("capture_metadata_mode", ""),
        )

    runner.cudagraph_dispatcher.dispatch = fake_dispatch
    plan = _plan_416()
    split_batch_slices, split_ubatch_slices = (
        _inplace_plan_to_execution_slices("inplace_serial", plan))
    current_context = SimpleNamespace(dp_metadata=None)
    attn_metadata = [
        {"layer.0": SimpleNamespace(seq_lens_list=[9, 9])},
        {"layer.0": SimpleNamespace(seq_lens_list=[9, 9])},
    ]
    contexts = []

    def fake_create_context(cur_forward_context, **kwargs):
        context = SimpleNamespace(**kwargs)
        context.dp_metadata = kwargs.get("dp_metadata")
        contexts.append(context)
        return context

    with patch("vllm_ascend.worker.model_runner_v3.get_forward_context",
               return_value=current_context), patch(
                   "vllm_ascend.worker.model_runner_v3.create_ascend_forward_context",
                   side_effect=fake_create_context):
        NPUModelRunner._make_split_batch_metadata_inplace_serial(
            runner,
            split_ubatch_slices,
            split_batch_slices,
            attn_metadata=attn_metadata,
            input_ids=torch.arange(416),
            positions=torch.arange(416),
            inputs_embeds=None,
            intermediate_tensors=None,
            batch_descriptor=BatchDescriptor(num_tokens=416,
                                             num_reqs=416,
                                             uniform=False,
                                             has_lora=False),
            aclgraph_runtime_mode=CUDAGraphMode.FULL,
            inplace_attention_backend="fia",
        )

    assert attn_metadata[0]["layer.0"].seq_lens_list == [9, 9]
    assert attn_metadata[1]["layer.0"].seq_lens_list == [9, 256]
    assert contexts[1].forced_attention_backend == "fia"
    assert contexts[1].batch_descriptor.attention_backend == "fia"
    assert contexts[1].batch_descriptor.capture_metadata_mode == "template"


def test_inplace_serial_uses_pa_for_offset_metadata_when_forced():
    runner = object.__new__(NPUModelRunner)
    runner.ascend_config = SimpleNamespace(
        split_batch_config=SimpleNamespace(enable_inplace_lazy_capture=True,
                                           inplace_validate_metadata_ptrs=True,
                                           inplace_force_pa_for_offset=True))
    runner.vllm_config = SimpleNamespace()
    runner.block_size = 256
    runner.cudagraph_dispatcher = SimpleNamespace()

    def fake_dispatch(**kwargs):
        return CUDAGraphMode.FULL, BatchDescriptor(
            num_tokens=kwargs["num_tokens"],
            num_reqs=kwargs["num_tokens"],
            uniform=kwargs["uniform_decode"],
            has_lora=kwargs["has_lora"],
            start_num_tokens=kwargs["start_num_tokens"],
            graph_variant=kwargs.get("graph_variant", ""),
            attention_backend=kwargs.get("attention_backend", ""),
            capture_metadata_mode=kwargs.get("capture_metadata_mode", ""),
        )

    runner.cudagraph_dispatcher.dispatch = fake_dispatch
    plan = _plan_416()
    split_batch_slices, split_ubatch_slices = (
        _inplace_plan_to_execution_slices("inplace_serial", plan))
    current_context = SimpleNamespace(dp_metadata=None)
    attn_metadata = [
        {"layer.0": SimpleNamespace(seq_lens_list=[9, 9])},
        {"layer.0": SimpleNamespace(seq_lens_list=[9, 9])},
    ]
    contexts = []

    def fake_create_context(cur_forward_context, **kwargs):
        context = SimpleNamespace(**kwargs)
        context.dp_metadata = kwargs.get("dp_metadata")
        contexts.append(context)
        return context

    with patch("vllm_ascend.worker.model_runner_v3.get_forward_context",
               return_value=current_context), patch(
                   "vllm_ascend.worker.model_runner_v3.create_ascend_forward_context",
                   side_effect=fake_create_context):
        NPUModelRunner._make_split_batch_metadata_inplace_serial(
            runner,
            split_ubatch_slices,
            split_batch_slices,
            attn_metadata=attn_metadata,
            input_ids=torch.arange(416),
            positions=torch.arange(416),
            inputs_embeds=None,
            intermediate_tensors=None,
            batch_descriptor=BatchDescriptor(num_tokens=416,
                                             num_reqs=416,
                                             uniform=False,
                                             has_lora=False),
            aclgraph_runtime_mode=CUDAGraphMode.FULL,
            inplace_attention_backend="fia",
        )

    assert attn_metadata[0]["layer.0"].seq_lens_list == [9, 9]
    assert attn_metadata[1]["layer.0"].seq_lens_list == [9, 9]
    assert contexts[1].forced_attention_backend == "pa"
    assert contexts[1].batch_descriptor.attention_backend == "pa"
    assert contexts[1].batch_descriptor.capture_metadata_mode == ""


def test_inplace_serial_metadata_uses_split_dispatch_when_outer_is_none():
    runner = object.__new__(NPUModelRunner)
    runner.ascend_config = SimpleNamespace(
        split_batch_config=SimpleNamespace(enable_inplace_lazy_capture=True,
                                           inplace_validate_metadata_ptrs=True))
    runner.vllm_config = SimpleNamespace()
    runner.cudagraph_dispatcher = SimpleNamespace()

    def fake_dispatch(**kwargs):
        return CUDAGraphMode.FULL, BatchDescriptor(
            num_tokens=kwargs["num_tokens"],
            num_reqs=kwargs["num_tokens"],
            uniform=kwargs["uniform_decode"],
            has_lora=kwargs["has_lora"],
            start_num_tokens=kwargs["start_num_tokens"],
            graph_variant=kwargs.get("graph_variant", ""),
            attention_backend=kwargs.get("attention_backend", ""),
            capture_metadata_mode=kwargs.get("capture_metadata_mode", ""),
        )

    runner.cudagraph_dispatcher.dispatch = fake_dispatch
    plan = _plan_416()
    split_batch_slices, split_ubatch_slices = (
        _inplace_plan_to_execution_slices("inplace_serial", plan))
    current_context = SimpleNamespace(dp_metadata=None)
    contexts = []

    def fake_create_context(cur_forward_context, **kwargs):
        context = SimpleNamespace(**kwargs)
        context.dp_metadata = kwargs.get("dp_metadata")
        contexts.append(context)
        return context

    input_ids = torch.arange(416)
    positions = torch.arange(416)

    with patch("vllm_ascend.worker.model_runner_v3.get_forward_context",
               return_value=current_context), patch(
                   "vllm_ascend.worker.model_runner_v3.create_ascend_forward_context",
                   side_effect=fake_create_context):
        NPUModelRunner._make_split_batch_metadata_inplace_serial(
            runner,
            split_ubatch_slices,
            split_batch_slices,
            attn_metadata=None,
            input_ids=input_ids,
            positions=positions,
            inputs_embeds=None,
            intermediate_tensors=None,
            batch_descriptor=BatchDescriptor(num_tokens=416,
                                             num_reqs=416,
                                             uniform=True,
                                             has_lora=False),
            aclgraph_runtime_mode=CUDAGraphMode.NONE,
            inplace_attention_backend="fia",
        )

    assert [c.cudagraph_runtime_mode for c in contexts] == [
        CUDAGraphMode.FULL,
        CUDAGraphMode.FULL,
    ]
    assert contexts[1].allow_inplace_lazy_capture is True


def test_inplace_serial_preserves_offset_input_views():
    runner = object.__new__(NPUModelRunner)
    input_ids = torch.arange(416)
    positions = torch.arange(416)

    sliced_input_ids, sliced_positions, _ = (
        NPUModelRunner._slice_split_batch_inputs(
            runner,
            slice(384, 416),
            input_ids,
            positions,
            None,
            None,
        )[:3])
    assert sliced_input_ids.storage_offset() == 384
    assert sliced_positions.storage_offset() == 384
    assert sliced_input_ids.data_ptr() == input_ids[384:].data_ptr()
    assert sliced_positions.data_ptr() == positions[384:].data_ptr()
    assert torch.equal(sliced_input_ids, input_ids[384:416])
    assert torch.equal(sliced_positions, positions[384:416])


def test_inplace_serial_preserves_offset_mrope_position_views():
    runner = object.__new__(NPUModelRunner)
    input_ids = torch.arange(416)
    positions = torch.arange(3 * 416).view(3, 416)

    sliced_input_ids, sliced_positions, _ = (
        NPUModelRunner._slice_split_batch_inputs(
            runner,
            slice(384, 416),
            input_ids,
            positions,
            None,
            None,
        )[:3])
    expected_positions = positions[:, 384:416]

    assert sliced_input_ids.storage_offset() == 384
    assert sliced_positions.shape == (3, 32)
    assert sliced_positions.storage_offset() == expected_positions.storage_offset()
    assert sliced_input_ids.data_ptr() == input_ids[384:].data_ptr()
    assert sliced_positions.data_ptr() == expected_positions.data_ptr()
    assert torch.equal(sliced_positions, expected_positions)


def test_offset_capture_replays_and_updates_before_return():
    runner = object.__new__(NPUModelRunner)
    runner.compilation_config = SimpleNamespace(cudagraph_num_of_warmups=1)
    runner.stream_main = SimpleNamespace(synchronize=lambda: None)

    calls = []
    graph_exists = {"value": False}

    class FakeStream:

        def __call__(self, stream):
            return nullcontext()

    def fake_model(**kwargs):
        mode = metadata.context.cudagraph_runtime_mode
        capturing = bool(metadata.context.capturing)
        calls.append(("model", mode, capturing))
        if mode == CUDAGraphMode.FULL and not graph_exists["value"]:
            graph_exists["value"] = True
            return torch.tensor([100, 101])
        if mode == CUDAGraphMode.FULL:
            return torch.tensor([200, 201])
        return torch.tensor([0, 1])

    def fake_has_graph(context):
        return graph_exists["value"]

    def fake_update(context, num_tokens, parallel_streams=False):
        calls.append(("update", num_tokens, parallel_streams))

    runner.model = fake_model
    runner._has_aclgraph_for_context = fake_has_graph
    runner._update_attn_params_for_split_ubatch = fake_update

    descriptor = BatchDescriptor(num_tokens=32,
                                 num_reqs=32,
                                 uniform=True,
                                 has_lora=False,
                                 start_num_tokens=384,
                                 graph_variant="inplace_serial",
                                 attention_backend="fia",
                                 capture_metadata_mode="template")
    metadata = SimpleNamespace(
        context=SimpleNamespace(cudagraph_runtime_mode=CUDAGraphMode.FULL,
                                capturing=False,
                                batch_descriptor=descriptor),
        input_ids=torch.arange(32),
        positions=torch.arange(32),
        inputs_embeds=None,
        intermediate_tensors=None,
    )
    split_slice = SimpleNamespace(num_tokens=32, graph_num_tokens=32)

    with patch("vllm_ascend.worker.model_runner_v3.torch.npu.stream",
               new=FakeStream()), patch(
                   "vllm_ascend.worker.model_runner_v3.override_forward_context",
                   return_value=nullcontext()):
        result = NPUModelRunner._run_inplace_serial_offset_capture(
            runner,
            metadata,
            split_slice,
            model_kwargs={},
        )

    assert torch.equal(result, torch.tensor([200, 201]))
    assert calls == [
        ("model", CUDAGraphMode.NONE, False),
        ("model", CUDAGraphMode.FULL, False),
        ("model", CUDAGraphMode.FULL, False),
        ("update", 32, False),
    ]
    assert metadata.context.cudagraph_runtime_mode == CUDAGraphMode.FULL
    assert metadata.context.capturing is False


def test_offset_capture_uses_parallel_stream_and_update_when_requested():
    runner = object.__new__(NPUModelRunner)
    runner.compilation_config = SimpleNamespace(cudagraph_num_of_warmups=0)

    stream_calls = []

    class FakeSyncStream:

        def __init__(self, name):
            self.name = name

        def synchronize(self):
            stream_calls.append(("sync", self.name))

    runner.stream_main = FakeSyncStream("main")
    runner.stream_parallel = FakeSyncStream("parallel")

    calls = []
    graph_exists = {"value": False}

    class FakeStream:

        def __call__(self, stream):
            stream_calls.append(("enter", stream.name))
            return nullcontext()

    def fake_model(**kwargs):
        mode = metadata.context.cudagraph_runtime_mode
        calls.append(("model", mode))
        if mode == CUDAGraphMode.FULL and not graph_exists["value"]:
            graph_exists["value"] = True
            return torch.tensor([100, 101])
        return torch.tensor([200, 201])

    def fake_has_graph(context):
        return graph_exists["value"]

    def fake_update(context, num_tokens, parallel_streams=False):
        calls.append(("update", num_tokens, parallel_streams))

    runner.model = fake_model
    runner._has_aclgraph_for_context = fake_has_graph
    runner._update_attn_params_for_split_ubatch = fake_update

    descriptor = BatchDescriptor(num_tokens=32,
                                 num_reqs=32,
                                 uniform=True,
                                 has_lora=False,
                                 start_num_tokens=384,
                                 graph_variant="inplace_parallel",
                                 attention_backend="fia",
                                 capture_metadata_mode="")
    metadata = SimpleNamespace(
        context=SimpleNamespace(cudagraph_runtime_mode=CUDAGraphMode.FULL,
                                capturing=False,
                                batch_descriptor=descriptor),
        input_ids=torch.arange(32),
        positions=torch.arange(32),
        inputs_embeds=None,
        intermediate_tensors=None,
    )
    split_slice = SimpleNamespace(num_tokens=32, graph_num_tokens=32)

    with patch("vllm_ascend.worker.model_runner_v3.torch.npu.stream",
               new=FakeStream()), patch(
                   "vllm_ascend.worker.model_runner_v3.override_forward_context",
                   return_value=nullcontext()):
        result = NPUModelRunner._run_inplace_serial_offset_capture(
            runner,
            metadata,
            split_slice,
            model_kwargs={},
            parallel_streams=True,
        )

    assert torch.equal(result, torch.tensor([200, 201]))
    assert calls == [
        ("model", CUDAGraphMode.FULL),
        ("model", CUDAGraphMode.FULL),
        ("update", 32, True),
    ]
    assert stream_calls == [
        ("enter", "parallel"),
        ("sync", "parallel"),
        ("enter", "parallel"),
        ("sync", "parallel"),
    ]
