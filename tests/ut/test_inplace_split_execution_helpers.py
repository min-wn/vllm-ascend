import numpy as np
import torch
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch
from vllm.config import CUDAGraphMode
from vllm.forward_context import BatchDescriptor
from vllm.v1.worker.ubatch_utils import UBatchSlice

from vllm_ascend.worker.model_runner_v3 import (
    NPUModelRunner,
    _inplace_plan_to_execution_slices,
    _template_fia_seq_lens_list,
)
from vllm_ascend.worker.ubatch_utils import create_inplace_split_batch_slices


def _plan_416():
    plan, reason = create_inplace_split_batch_slices(
        np.ones(416, dtype=np.int32),
        total_num_tokens=416,
        uniform_decode_query_len=1,
        cudagraph_capture_sizes={256, 384, 512},
    )
    assert plan is not None, reason
    return plan


def test_inplace_serial_plan_enables_execution_slices():
    plan = _plan_416()

    split_batch_slices, split_ubatch_slices = (
        _inplace_plan_to_execution_slices("inplace_serial", plan))

    assert split_batch_slices == plan.split_slices
    assert split_ubatch_slices == [
        UBatchSlice(slice(0, 384), slice(0, 384)),
        UBatchSlice(slice(384, 416), slice(384, 416)),
    ]


def test_inplace_parallel_plan_stays_dry_run():
    plan = _plan_416()

    split_batch_slices, split_ubatch_slices = (
        _inplace_plan_to_execution_slices("inplace_parallel", plan))

    assert split_batch_slices is None
    assert split_ubatch_slices is None


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


def test_inplace_serial_templates_only_offset_fia_metadata():
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

    def fake_create_context(cur_forward_context, **kwargs):
        context = SimpleNamespace(**kwargs)
        context.dp_metadata = kwargs.get("dp_metadata")
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
    assert attn_metadata[1]["layer.0"].seq_lens_list == [9, 32]


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
