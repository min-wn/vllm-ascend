"""Load Qwen3.5 weights for the isolated TP communication/QKV benchmark.

The official external-launcher executor initializes the worker, TP group and
checkpoint weights. Stopping before engine initialization avoids KV allocation,
model profiling, prefill and full-model graph capture. This is not a decoder
loader and does not expand the supported models in multi_graph_replay.py.
"""

import hashlib
import os
from contextlib import nullcontext
from pathlib import Path


def run_qwen35(args, workload, *, experiment):
    """Call a boundary experiment with the real first full-attention QKV shard.

    The callback follows multi_graph_tp_boundary.experiment and accepts an
    additional keyword-only ``module``. Its inputs/context callbacks carry no
    attention metadata; only the checked local unquantized QKV may use them.
    """
    if int(os.environ.get("WORLD_SIZE", "1")) != args.tensor_parallel_size or args.tensor_parallel_size < 2:
        raise ValueError("launch with torchrun and matching multi-rank tensor parallelism")
    if getattr(args, "batch_invariant", False) or getattr(args, "compiled", False):
        raise ValueError("this loader qualifies only eager BF16 QKV weights")
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"

    import torch
    from vllm.config import CUDAGraphMode, set_current_vllm_config
    from vllm.distributed.parallel_state import get_tp_group, get_world_group
    from vllm.engine.arg_utils import EngineArgs
    from vllm.v1.executor.uniproc_executor import ExecutorWithExternalLauncher

    from vllm_ascend.ops.linear import AscendQKVParallelLinear, AscendUnquantizedLinearMethod

    config = EngineArgs(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        distributed_executor_backend="external_launcher",
        dtype="bfloat16",
        enforce_eager=True,
        max_model_len=512,
        max_num_seqs=8,
        max_num_batched_tokens=512,
        gpu_memory_utilization=0.6,
        enable_prefix_caching=False,
        limit_mm_per_prompt={"image": 0, "video": 0},
        seed=42,
        compilation_config={"mode": 0, "cudagraph_mode": "NONE", "cudagraph_capture_sizes": []},
        additional_config={"ascend_compilation_config": {"enable_npugraph_ex": False}},
    ).create_engine_config()
    if config.model_config.hf_config.model_type != "qwen3_5" or config.quant_config is not None:
        raise ValueError("this benchmark loader requires unquantized Qwen3.5")
    if config.use_v2_model_runner or config.compilation_config.cudagraph_mode != CUDAGraphMode.NONE:
        raise ValueError("this benchmark loader requires eager model runner V1")
    text_config = config.model_config.hf_text_config
    layer_types = text_config.layer_types
    layer_index = layer_types.index("full_attention")
    module_path = f"language_model.model.layers.{layer_index}.self_attn.qkv_proj"

    executor = ExecutorWithExternalLauncher(config)
    try:
        runner = executor.driver_worker.model_runner
        model = runner.get_model()
        module = model.get_submodule(module_path)
        group = get_tp_group()
        if group.world_size != args.tensor_parallel_size or get_world_group().world_size != group.world_size:
            raise ValueError("the loaded TP group does not match the benchmark world")
        if (
            type(module) is not AscendQKVParallelLinear
            or type(module.quant_method) is not AscendUnquantizedLinearMethod
            or module.custom_op is not None
            or module.gather_output
            or not module.return_bias
            or module.bias is not None
            or module.weight.dtype != torch.bfloat16
        ):
            raise ValueError("QKV must be a local, bias-free, unquantized Ascend projection")
        attention = model.get_submodule(module_path.rsplit(".", 1)[0])
        expected_width = (
            attention.num_heads * (1 + attention.attn_output_gate) + 2 * attention.num_kv_heads
        ) * attention.head_dim
        if tuple(module.weight.shape) != (expected_width, text_config.hidden_size):
            raise ValueError("loaded QKV shard disagrees with the model's gated-Q/head dimensions")
        if getattr(runner, "kv_caches", None):
            raise ValueError("unexpected KV cache allocation in the weight-only loader")

        args.boundary_module_path = module_path
        args.loader_metadata = dict(
            loader="official_external_launcher_executor_weights_only",
            model_type=config.model_config.hf_config.model_type,
            layer_index=layer_index,
            layer_type=layer_types[layer_index],
            module_path=module_path,
            module_type=type(module).__qualname__,
            quant_method=type(module.quant_method).__qualname__,
            hidden_size=text_config.hidden_size,
            num_hidden_layers=text_config.num_hidden_layers,
            full_attention_layers=layer_types.count("full_attention"),
            linear_attention_layers=layer_types.count("linear_attention"),
            attention_output_gate=attention.attn_output_gate,
            local_qkv_shape=list(module.weight.shape),
            local_query_heads=attention.num_heads,
            local_kv_heads=attention.num_kv_heads,
            head_dim=attention.head_dim,
            full_model_forward=False,
            kv_cache_initialized=False,
            attention_metadata_initialized=False,
            forward_context="none_local_unquantized_qkv_only",
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        )

        def inputs(start, stop):
            if not 0 <= start <= stop <= workload.capacity:
                raise ValueError("input slice exceeds the selected graph capacity")
            return None

        def context(data):
            # Checked QKV forward calls only quant_method.apply -> F.linear.
            # No custom parallel op, output gather or attention reads context.
            return nullcontext()

        def reset_kv():
            raise RuntimeError("this operator-only loader has no initialized KV cache")

        with torch.inference_mode(), set_current_vllm_config(config):
            return experiment(args, workload, runner, model, inputs, context, reset_kv, module=module)
    finally:
        executor.shutdown()
