# SPDX-License-Identifier: Apache-2.0
"""Weight-only TP2 launcher for the complete compact-state hybrid experiment.

Resolve the installed default graph dispatcher before loading an eager worker.
The benchmark, rather than the production raw-page allocator, then owns every
real layer cache. No capture capacity is supplied or manufactured here.
"""

import hashlib
import os
from pathlib import Path


def engine_options(model, tensor_parallel_size, *, eager):
    if tensor_parallel_size != 2:
        raise ValueError("the compact native kernels require exactly TP2")
    options = dict(
        model=model,
        tensor_parallel_size=tensor_parallel_size,
        distributed_executor_backend="external_launcher",
        dtype="bfloat16",
        enforce_eager=eager,
        max_model_len=2048,
        max_num_seqs=256,
        max_num_batched_tokens=2048,
        speculative_config={"method": "ngram", "num_speculative_tokens": 3},
        gpu_memory_utilization=0.6,
        enable_prefix_caching=False,
        limit_mm_per_prompt={"image": 0, "video": 0},
        seed=42,
        additional_config={"ascend_compilation_config": {"enable_npugraph_ex": False}},
    )
    if eager:
        options["compilation_config"] = {"mode": 0, "cudagraph_mode": "NONE", "cudagraph_capture_sizes": []}
    return options


def resolve_default_catalog(config, trace):
    from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher

    dispatcher = CudagraphDispatcher(config)
    dispatcher.initialize_cudagraph_keys(config.compilation_config.cudagraph_mode, 4)
    sizes = tuple(config.compilation_config.cudagraph_capture_sizes)
    capacities = []
    for step in trace.steps:
        mode, descriptor = dispatcher.dispatch(step["batch"] * 4, uniform_decode=True)
        if str(mode) != "FULL" or descriptor.num_tokens not in sizes:
            raise ValueError("fixed trajectory must fit the installed default FULL graph catalog")
        if descriptor.num_reqs * 4 != descriptor.num_tokens:
            raise ValueError("default dispatcher did not preserve Q4 request boundaries")
        capacities.append(descriptor.num_tokens)
    return dict(
        capture_sizes=list(sizes),
        dispatched_tokens=capacities,
        required_tokens=sorted(set(capacities)),
        config_source="installed EngineArgs defaults; no capture sizes supplied",
        query=4,
    )


def run_full_model(args, trace, *, experiment):
    if int(os.environ.get("WORLD_SIZE", "1")) != 2 or args.tensor_parallel_size != 2:
        raise ValueError("launch this benchmark through torchrun with exactly two ranks")
    # These are the same launcher flags as the established weight-only fixture.
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"

    import torch
    from vllm.config import CUDAGraphMode, set_current_vllm_config
    from vllm.distributed.parallel_state import get_tp_group, get_world_group
    from vllm.engine.arg_utils import EngineArgs
    from vllm.v1.executor.uniproc_executor import ExecutorWithExternalLauncher

    default_config = EngineArgs(**engine_options(args.model, 2, eager=False)).create_engine_config()
    args.selection = resolve_default_catalog(default_config, trace)
    config = EngineArgs(**engine_options(args.model, 2, eager=True)).create_engine_config()
    text = config.model_config.hf_text_config
    if (
        config.model_config.hf_config.model_type != "qwen3_5"
        or config.quant_config is not None
        or text.layer_types.count("linear_attention") != 48
        or text.layer_types.count("full_attention") != 16
        or text.num_hidden_layers != 64
        or config.use_v2_model_runner
        or config.compilation_config.cudagraph_mode != CUDAGraphMode.NONE
    ):
        raise ValueError("full replay currently requires the unquantized Qwen3.5-27B V1 TP2 layout")
    executor = ExecutorWithExternalLauncher(config)
    try:
        runner = executor.driver_worker.model_runner
        model = runner.get_model()
        if get_tp_group().world_size != 2 or get_world_group().world_size != 2:
            raise ValueError("the initialized process group is not the requested TP2 world")
        if getattr(runner, "kv_caches", None):
            raise ValueError("weight-only loader unexpectedly allocated production raw KV pages")
        args.loader_metadata = dict(
            loader="official_external_launcher_weights_only_direct_hybrid_caches",
            model_type="qwen3_5",
            tensor_parallel_size=2,
            decoder_layers=64,
            gdn_layers=48,
            fia_layers=16,
            query_tokens=4,
            weight_dtype="bfloat16",
            ssm_dtype="float32",
            max_model_len=config.model_config.max_model_len,
            max_num_batched_tokens=config.scheduler_config.max_num_batched_tokens,
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        )
        with torch.inference_mode(), set_current_vllm_config(config):
            return experiment(args, trace, runner, model, config)
    finally:
        executor.shutdown()
