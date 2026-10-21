# SPDX-License-Identifier: Apache-2.0
"""Replay existing ledger through installed dispatcher and GDN builder on CPU."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    output = args.root / "metadata-replay.json"
    if output.exists():
        raise FileExistsError("preserve the metadata-only observation")
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
    import torch
    from vllm.config import set_current_vllm_config
    from vllm.engine.arg_utils import EngineArgs
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher
    from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec, MambaSpec

    from vllm_ascend.attention.utils import AscendCommonAttentionMetadata
    from vllm_ascend.ops.gdn_attn_builder import AscendGDNAttentionMetadataBuilder

    native = json.loads((args.root / "native.rank0.json").read_bytes())
    source = Path("/workspace/gdn-ready-generation-v1/pair0.ready/result.rank0.json")
    ledger = json.loads(source.read_bytes())
    config = EngineArgs(**native["engine_options"]).create_engine_config()
    dispatcher = CudagraphDispatcher(config)
    dispatcher.initialize_cudagraph_keys(config.compilation_config.cudagraph_mode, 4)
    spec = MambaSpec(
        block_size=2048,
        shapes=((6, 5120), (24, 128, 128)),
        dtypes=(torch.bfloat16, torch.float32),
        page_size_padded=1835008,
        mamba_cache_mode="none",
        num_speculative_blocks=3,
    )
    report = dict(
        schema="gdn-ready-installed-metadata-replay-v1",
        native_generation_observation=False,
        model_executed=False,
        physical_block_ids_observed=False,
        block_ids="synthetic fixed quartets solely for metadata semantics; no allocation claim",
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        workload_sha256=ledger["workload_sha256"],
        effective=dict(
            graph_mode=str(config.compilation_config.cudagraph_mode),
            capture_sizes=list(config.compilation_config.cudagraph_capture_sizes),
            async_scheduling=config.scheduler_config.async_scheduling,
            cache_mode=config.cache_config.mamba_cache_mode,
        ),
        scheduler_assumptions=(
            "synchronous ngram; existing compact token histories, drafts and EOS; no new admissions, "
            "preemption or cache hits; query width=1+actual drafts; not a native generation trace"
        ),
        steps=[],
    )
    native_groups = native["cache_config"]["groups"]
    fia_spec = FullAttentionSpec(block_size=896, num_kv_heads=2, head_size=256, dtype=torch.bfloat16)
    kv_config = KVCacheConfig(
        num_blocks=native["cache_config"]["num_blocks"],
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(layer_names=g["layers"], kv_cache_spec=fia_spec if i == 0 else spec)
            for i, g in enumerate(native_groups)
        ],
    )
    scheduler_block_size = math.lcm(*(g.kv_cache_spec.block_size for g in kv_config.kv_cache_groups))
    manager = KVCacheManager(
        kv_config,
        max_model_len=2048,
        scheduler_block_size=scheduler_block_size,
        hash_block_size=128,
        enable_caching=False,
    )
    required = manager.coordinator.get_num_blocks_to_allocate(
        request_id="metadata-only-owner",
        num_tokens=297,
        new_computed_blocks=tuple([] for _ in native_groups),
        num_encoder_tokens=0,
        total_computed_tokens=0,
        num_local_computed_tokens=0,
        num_tokens_main_model=297,
    )
    free = manager.block_pool.get_num_free_blocks()
    report["native_allocation_budget"] = dict(
        cpu_bookkeeping_only=True,
        live_requests=56,
        max_query_end=297,
        scheduler_block_size=scheduler_block_size,
        hash_block_size=128,
        total_blocks=kv_config.num_blocks,
        free_blocks=free,
        blocks_per_request=required,
        all_56_required=56 * required,
        resident_upper_bound_without_other_reservations=free // required,
        physical_blocks_allocated=False,
    )
    with set_current_vllm_config(config):
        builder = AscendGDNAttentionMetadataBuilder(spec, ["diagnostic.linear_attn"], config, torch.device("cpu"))
        for step in ledger["generation"]["steps"]:
            ids = step["request_ids"]
            live = len(ids)
            drafts = [len(p) for p in step["proposals"]]
            widths = [1 + d for d in drafts]
            rows = sum(widths)
            uniform = max(widths) == 4 and rows == 4 * live
            mode, descriptor = dispatcher.dispatch(rows, uniform_decode=uniform)
            graph_requests = descriptor.num_reqs if str(mode) == "FULL" else live
            padding = graph_requests - live
            boundaries = [0]
            for width in widths:
                boundaries.append(boundaries[-1] + width)
            boundaries.extend([rows] * padding)
            qsl = torch.tensor(boundaries, dtype=torch.int32)
            lengths = torch.tensor(
                [s - 4 + q for s, q in zip(step["seq_lengths"], widths, strict=True)] + [0] * padding,
                dtype=torch.int32,
            )
            computed = lengths - torch.diff(qsl)
            common = AscendCommonAttentionMetadata(
                query_start_loc=qsl,
                query_start_loc_cpu=qsl,
                seq_lens=lengths,
                _seq_lens_cpu=lengths,
                seq_lens_cpu=lengths,
                seq_lens_cpu_upper_bound=lengths,
                _num_computed_tokens_cpu=computed,
                num_computed_tokens_cpu=computed,
                num_reqs=graph_requests,
                num_actual_tokens=rows,
                num_input_tokens=descriptor.num_tokens,
                max_query_len=max(widths),
                max_seq_len=int(lengths.max()),
                block_table_tensor=torch.tensor(
                    [[1 + 4 * owner + k for k in range(4)] for owner in ids] + [[0] * 4] * padding,
                    dtype=torch.int32,
                ),
                slot_mapping=torch.full((descriptor.num_tokens,), -1, dtype=torch.int64),
                positions=torch.zeros(descriptor.num_tokens, dtype=torch.int64),
                causal=True,
                is_prefilling=torch.zeros(graph_requests, dtype=torch.bool),
            )
            metadata = builder.build(
                0,
                common,
                num_accepted_tokens=torch.tensor(step["accepted_from_previous"] + [1] * padding, dtype=torch.int32),
                num_decode_draft_tokens_cpu=torch.tensor(
                    [d if d else -1 for d in drafts] + [-1] * padding, dtype=torch.int32
                ),
                num_actual_reqs=live,
            )
            report["steps"].append(
                dict(
                    step=step["step"],
                    owners=ids,
                    draft_counts=drafts,
                    query_lengths=widths,
                    head_rows=rows,
                    graph_mode=str(mode),
                    capacity=descriptor.num_tokens,
                    graph_requests=graph_requests,
                    gdn=dict(
                        num_spec=metadata.num_spec_decodes,
                        num_spec_tokens=metadata.num_spec_decode_tokens,
                        num_decodes=metadata.num_decodes,
                        num_decode_tokens=metadata.num_decode_tokens,
                        num_prefills=metadata.num_prefills,
                        num_prefill_tokens=metadata.num_prefill_tokens,
                        spec_query_start_loc=(
                            metadata.spec_query_start_loc.tolist()
                            if metadata.spec_query_start_loc is not None
                            else None
                        ),
                        non_spec_query_start_loc=(
                            metadata.non_spec_query_start_loc.tolist()
                            if metadata.non_spec_query_start_loc is not None
                            else None
                        ),
                    ),
                )
            )
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(steps=len(report["steps"]), effective=report["effective"]), indent=2))


if __name__ == "__main__":
    main()
