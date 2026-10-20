"""CPU-only capacity estimates for hybrid Q4 replay; never allocate NPU memory.

Checkpoint headers, model configuration and the immutable workload are evidence.
Allocator residency, TP loading and graph-pool overhead still require measurement.
"""

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path

from multi_graph_dynamic_trace import trace_digest


def ceil_div(value, divisor):
    return (value + divisor - 1) // divisor


def weight_group(name):
    if "visual." in name or "vision_model." in name:
        return "vision"
    if "mtp." in name or "mtp_" in name:
        return "mtp"
    if "lm_head." in name:
        return "lm_head"
    if "embed_tokens." in name:
        return "text_embedding"
    if "language_model." in name:
        return "text_decoder"
    return "other"


def checkpoint_inventory(model_dir):
    """Read only JSON and safetensors headers, never checkpoint tensor payloads."""
    model_dir = Path(model_dir)
    config_bytes = (model_dir / "config.json").read_bytes()
    index_path = model_dir / "model.safetensors.index.json"
    index_bytes = index_path.read_bytes() if index_path.exists() else None
    index = json.loads(index_bytes) if index_bytes is not None else {}
    shards = sorted(set(index.get("weight_map", {}).values())) or sorted(
        p.name for p in model_dir.glob("*.safetensors")
    )
    if not shards:
        raise ValueError("no safetensors checkpoint shards found")
    groups, dtypes, files, names = {}, {}, [], set()
    header_read_bytes = 0
    for shard in shards:
        path = model_dir / shard
        if path.resolve().parent != model_dir.resolve():
            raise ValueError("checkpoint index shard must be inside model_dir")
        with path.open("rb") as source:
            prefix = source.read(8)
            if len(prefix) != 8:
                raise ValueError("truncated safetensors length prefix")
            header_size = struct.unpack("<Q", prefix)[0]
            if header_size > path.stat().st_size - 8:
                raise ValueError("safetensors header exceeds file size")
            header_bytes = source.read(header_size)
        header_read_bytes += 8 + header_size
        header = json.loads(header_bytes)
        payload_bytes = 0
        for name, tensor in header.items():
            if name == "__metadata__":
                continue
            if name in names:
                raise ValueError("duplicate tensor in checkpoint shards")
            names.add(name)
            start, stop = tensor["data_offsets"]
            if not 0 <= start <= stop <= path.stat().st_size - 8 - header_size:
                raise ValueError("safetensors offsets exceed payload")
            count = stop - start
            group = weight_group(name)
            groups[group] = groups.get(group, 0) + count
            dtypes[tensor["dtype"]] = dtypes.get(tensor["dtype"], 0) + count
            payload_bytes += count
        files.append(
            dict(
                name=shard,
                file_bytes=path.stat().st_size,
                tensor_payload_bytes=payload_bytes,
                header_sha256=hashlib.sha256(header_bytes).hexdigest(),
            )
        )
    payload = sum(groups.values())
    if index.get("metadata", {}).get("total_size", payload) != payload:
        raise ValueError("index total_size differs from tensor header payload sum")
    if index.get("weight_map") and set(index["weight_map"]) != names:
        raise ValueError("checkpoint index tensor names differ from shard headers")
    return dict(
        model_dir=str(model_dir),
        config=json.loads(config_bytes),
        config_sha256=hashlib.sha256(config_bytes).hexdigest(),
        index_sha256=hashlib.sha256(index_bytes).hexdigest() if index_bytes is not None else None,
        index_total_size_bytes=index.get("metadata", {}).get("total_size"),
        tensor_payload_bytes=payload,
        tensor_count=len(names),
        payload_bytes_by_group=groups,
        payload_bytes_by_dtype=dtypes,
        checkpoint_file_bytes=sum(f["file_bytes"] for f in files),
        bytes_read_from_safetensors=header_read_bytes,
        tensor_payloads_read=False,
        files=files,
    )


def trace_residency(trace, block_size):
    query = trace["query"]
    prefix = {int(request): int(length) for request, length in trace["initial_prefix_by_request"].items()}
    if query < 1 or block_size < 1 or not prefix or any(length < 0 for length in prefix.values()):
        raise ValueError("positive query/block size and nonnegative initial prefix required")
    maximum = dict(prefix)
    for record in trace["steps"]:
        requests, lengths = record["request_ids"], record["seq_lengths"]
        if len(requests) != record["batch"] or len(requests) != len(lengths) or len(set(requests)) != len(requests):
            raise ValueError("trace request and length arrays disagree")
        for request, length in zip(requests, lengths, strict=True):
            if request not in maximum or length < query:
                raise ValueError("trace contains unknown request or incomplete query")
            maximum[request] = max(maximum[request], length)
    return dict(
        logical_requests=len(prefix),
        candidate_slots_including_null=1 + len(prefix) * query,
        initial_prefix_tokens=sum(prefix.values()),
        maximum_retained_tokens=sum(maximum.values()),
        initial_fia_blocks=sum(ceil_div(length, block_size) for length in prefix.values()),
        maximum_fia_blocks=sum(ceil_div(length, block_size) for length in maximum.values()),
        max_sequence_length=max(maximum.values()),
        absent_requests_keep_storage=True,
    )


def estimate_budget(inventory, trace_document, *, tensor_parallel_sizes=(2, 4, 8), block_size=128, capacity_gib=64):
    trace = trace_document["trace"]
    digest = trace_digest(trace)
    if trace_document.get("trace_sha256") != digest:
        raise ValueError("workload manifest digest mismatch")
    config = inventory["config"].get("text_config", inventory["config"])
    layer_types = config["layer_types"]
    if len(layer_types) != config["num_hidden_layers"] or set(layer_types) - {"linear_attention", "full_attention"}:
        raise ValueError("hybrid layer_types must describe every supported decoder layer")
    if capacity_gib <= 0 or block_size < 1:
        raise ValueError("positive device capacity and FIA block size required")
    dtypes = {"float32": 4, "bfloat16": 2, "float16": 2}
    activation_dtype = config.get("dtype", "bfloat16")
    state_dtype = config.get("mamba_ssm_dtype", activation_dtype)
    if activation_dtype not in dtypes or state_dtype not in dtypes:
        raise ValueError("unsupported activation or SSM dtype")
    residency = trace_residency(trace, block_size)
    linear_layers, full_layers = layer_types.count("linear_attention"), layer_types.count("full_attention")
    groups = inventory["payload_bytes_by_group"]
    text_weights = sum(groups.get(group, 0) for group in ("text_decoder", "text_embedding", "lm_head"))
    rows = []
    for tp in tensor_parallel_sizes:
        if isinstance(tp, bool) or not isinstance(tp, int) or tp < 1:
            raise ValueError("positive integer TP sizes required")
        for heads in (config["linear_num_key_heads"], config["linear_num_value_heads"], config["num_attention_heads"]):
            if heads % tp:
                raise ValueError("TP must divide GDN key/value and query head counts")
        kv_heads = config["num_key_value_heads"]
        if kv_heads % tp and tp % kv_heads:
            raise ValueError("FIA KV heads require exact sharding or integral replication")
        local_kv = max(1, kv_heads // tp)
        ssm_page = (
            config["linear_num_value_heads"]
            // tp
            * config["linear_value_head_dim"]
            * config["linear_key_head_dim"]
            * dtypes[state_dtype]
        )
        conv_channels = (
            2 * config["linear_num_key_heads"] * config["linear_key_head_dim"]
            + config["linear_num_value_heads"] * config["linear_value_head_dim"]
        ) // tp
        conv_width = config["linear_conv_kernel_dim"] - 1 + trace["query"] - 1
        conv_page = conv_channels * conv_width * dtypes[activation_dtype]
        logical_page = ssm_page + conv_page
        fia_page = block_size * 2 * local_kv * config["head_dim"] * dtypes[activation_dtype]
        aligned_page = ceil_div(logical_page, fia_page) * fia_page
        slots = residency["candidate_slots_including_null"]
        state_logical = slots * logical_page * linear_layers
        state_raw = slots * aligned_page * linear_layers
        fia_initial = (1 + residency["initial_fia_blocks"]) * fia_page * full_layers
        fia_maximum = (1 + residency["maximum_fia_blocks"]) * fia_page * full_layers
        ideal_text_weights = math.ceil(text_weights / tp)
        common = dict(
            tensor_parallel_size=tp,
            local_fia_kv_heads=local_kv,
            fia_kv_heads_replicated=tp > kv_heads,
            local_gdn_value_heads=config["linear_num_value_heads"] // tp,
            ssm_bytes_per_slot_per_layer=ssm_page,
            conv_bytes_per_slot_per_layer=conv_page,
            logical_gdn_bytes_per_slot_per_layer=logical_page,
            estimated_aligned_gdn_page_bytes=aligned_page,
            ssm_state_bytes=slots * ssm_page * linear_layers,
            conv_state_bytes=slots * conv_page * linear_layers,
            logical_gdn_state_bytes=state_logical,
            estimated_raw_gdn_page_bytes=state_raw,
            fia_128_token_page_bytes=fia_page,
            initial_fia_bytes=fia_initial,
            maximum_fia_bytes=fia_maximum,
            ideal_text_checkpoint_shard_bytes=ideal_text_weights,
            ideal_full_checkpoint_shard_bytes=math.ceil(inventory["tensor_payload_bytes"] / tp),
            ideal_text_weights_plus_logical_state_and_max_fia_bytes=ideal_text_weights + state_logical + fia_maximum,
            ideal_text_weights_plus_raw_state_and_max_fia_bytes=ideal_text_weights + state_raw + fia_maximum,
        )
        common["estimated_remaining_bytes_before_unknowns"] = (
            int(capacity_gib * 1024**3) - common["ideal_text_weights_plus_raw_state_and_max_fia_bytes"]
        )
        compact_slots = residency["logical_requests"] + 1
        local_value_heads = config["linear_num_value_heads"] // tp
        compact_base = compact_slots * ssm_page * linear_layers
        log_record_elements = trace["query"] * (config["linear_key_head_dim"] + config["linear_value_head_dim"] + 8)
        log_bank = compact_slots * local_value_heads * log_record_elements * 4 * linear_layers
        valid_bank = compact_slots * local_value_heads * 8 * 4 * linear_layers
        compact_conv = slots * conv_page * linear_layers
        products = (
            residency["logical_requests"]
            * local_value_heads
            * config["linear_value_head_dim"]
            * (2 * trace["query"])
            * 4
            * linear_layers
        )
        compact_total = ideal_text_weights + compact_base + 2 * log_bank + 2 * valid_bank + compact_conv + fia_maximum
        common["compact_direct_allocation_estimate"] = dict(
            base_slots_including_null=compact_slots,
            base_bytes=compact_base,
            decode_conv_bytes=compact_conv,
            log_bank_bytes=log_bank,
            padded_valid_bank_bytes=valid_bank,
            two_log_and_valid_banks_bytes=2 * (log_bank + valid_bank),
            products_scratch_all_layers_max_capacity_bytes=products,
            ideal_text_weights_plus_compact_state_and_max_fia_bytes=compact_total,
            remaining_bytes_before_products_and_unknowns=int(capacity_gib * 1024**3) - compact_total,
            assumption="direct separate conv/base/log allocation; obsolete dense SSM pages released",
            production_cache_manager_support=False,
        )
        rows.append(common)
    return dict(
        status="cpu_capacity_estimate_not_allocation_or_oom_evidence",
        trace_sha256=digest,
        checkpoint=inventory,
        workload_residency=residency,
        assumptions=dict(
            activation_dtype=activation_dtype,
            ssm_dtype=state_dtype,
            query=trace["query"],
            gdn_layers=linear_layers,
            full_attention_layers=full_layers,
            fia_block_size=block_size,
            device_capacity_gib=capacity_gib,
            state_cache_mode="none; four complete candidate slots per stable request plus shared null slot",
            page_alignment="ceil(logical_GDN_page / independent_FIA_page) * independent_FIA_page; estimate",
            cache_layout="separate GDN pools and 128-token FIA pages; unified padded FIA groups may cost more",
            weights="ideal text checkpoint / TP only; loaded residency, replication, padding and NZ copies unknown",
            baseline_and_split="one state trajectory resident at a time; no duplicate correctness snapshots",
        ),
        unknown_additional_device_bytes={
            name: None
            for name in (
                "graph_pools_per_bucket_and_execution_mode",
                "q4_products_and_qk_cast_buffers",
                "attention_and_communication_workspaces",
                "hidden_residual_and_projection_activations",
                "prefix_initialization_peak",
                "correctness_state_snapshots",
                "checkpoint_load_and_nz_weight_repacking",
                "vision_or_mtp_weights_if_loaded",
                "replicated_text_weights_and_vocab_padding",
                "allocator_fragmentation_and_runtime_reservations",
                "hybrid_group_padding_and_page_overallocation",
            )
        },
        estimates=rows,
        limitations=[
            "No torch/vLLM imports, model loading, device allocation or NPU execution performed.",
            "Actual full-model TP2/TP4/TP8 execution support and available free memory are unverified.",
            "A negative estimate is a planning warning, not measured OOM; a positive estimate does not prove fit.",
            "TP8 replicates the four FIA KV heads and therefore does not halve FIA residency relative to TP4.",
        ],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--tensor-parallel-sizes", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--capacity-gib", type=float, default=64)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if bool(args.model_dir) == bool(args.inventory) or (not args.inventory_only and args.trace is None):
        parser.error("choose exactly one model-dir/inventory, and supply trace unless inventory-only")
    if args.output.exists():
        parser.error("refuse to overwrite existing evidence")
    inventory = checkpoint_inventory(args.model_dir) if args.model_dir else json.loads(args.inventory.read_text())
    report = (
        inventory
        if args.inventory_only
        else estimate_budget(
            inventory,
            json.loads(args.trace.read_text()),
            tensor_parallel_sizes=args.tensor_parallel_sizes,
            block_size=args.block_size,
            capacity_gib=args.capacity_gib,
        )
    )
    report["source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="ascii")


if __name__ == "__main__":
    main()
