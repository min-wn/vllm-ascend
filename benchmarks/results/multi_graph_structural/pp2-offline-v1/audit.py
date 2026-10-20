# SPDX-License-Identifier: Apache-2.0
"""Offline PP2 shape, residency and closed-step pipeline screening.

No torch imports, NPU execution, fitted timings or predicted speedups. Normalized
pipeline costs assume identical linear-time stages, independent resources and
zero transport/update cost. They are necessary-screen scenarios, not bounds on
arbitrary kernels or schedules. Whole requests use the existing default sizes.
"""

import hashlib
import json
import math
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
RESULTS = HERE.parent
QUERY = 4
GROUP_COUNTS = (2, 4, 8)
TRACE_SHA256 = "0519bc80410bd15d0893e0283189425ec285d53003198c163183b79e6249e524"


def read(path):
    return json.loads(path.read_bytes())


def digest(path):
    # Repository checkout filters may change CRLF without changing the source.
    # Checkpoint safetensors header hashes remain raw bytes, as collected.
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def linear_pipeline(stage0, stage1, transfer=None):
    """Two serial compute stages and one independent serial transfer resource."""
    if transfer is None:
        transfer = [0] * len(stage0)
    a = c = b = 0
    for ai, bi, ci in zip(stage0, stage1, transfer, strict=True):
        if min(ai, bi, ci) < 0:
            raise ValueError("resource times must be nonnegative")
        a += ai
        c = max(a, c) + ci
        b = max(c, b) + bi
    return b


def capacity_table(catalog, maximum_batch):
    legal = sorted(n for n in catalog if n % QUERY == 0)
    return [0, *(next(n for n in legal if n >= QUERY * b) for b in range(1, maximum_batch + 1))]


def partition_table(capacities, maximum_groups):
    """Keep minimum capacity sum for every (groups, live rows, max capacity).

    This dynamic program permits all ordered positive whole-request partitions.
    For identical linear stages, the zero-link makespan is sum(cap)+max(cap).
    Pruning by minimum sum at fixed max preserves that specific objective.
    """
    maximum_batch = len(capacities) - 1
    # Additive squared-size tie breaks avoid selecting tiny tail groups merely
    # because several partitions have identical sum+max capacity.
    previous = {0: {0: (0, 0, 0, ())}}
    answer = {}
    for groups in range(1, maximum_groups + 1):
        current = {}
        for used, maxima in previous.items():
            for live in range(1, maximum_batch - used + 1):
                cap = capacities[live]
                destinations = current.setdefault(used + live, {})
                for old_max, (old_sum, old_cap_square, old_live_square, old_partition) in maxima.items():
                    key = max(old_max, cap)
                    candidate = (
                        old_sum + cap,
                        old_cap_square + cap**2,
                        old_live_square + live**2,
                        (*old_partition, live),
                    )
                    incumbent = destinations.get(key)
                    if incumbent is None or candidate < incumbent:
                        destinations[key] = candidate
        answer[groups] = {
            batch: min(
                maxima.values(),
                key=lambda entry: (entry[0] + max(capacities[n] for n in entry[3]), entry[1:]),
            )[3]
            for batch, maxima in current.items()
        }
        previous = current
    return answer


def partition_record(batch, groups, capacities, selected):
    quotient, remainder = divmod(batch, groups)
    balanced = [quotient + (index < remainder) for index in range(groups)]
    partition = sorted(selected, reverse=True)
    caps = [capacities[n] for n in partition]
    parent = capacities[batch]
    normalized = linear_pipeline(caps, caps) / parent
    assert linear_pipeline(caps, caps) == sum(caps) + max(caps)
    assert len(partition) == groups and sum(partition) == batch and min(partition) > 0
    return dict(
        batch=batch,
        padding_tokens=parent,
        live_requests_per_group=partition,
        group_capacity_tokens=caps,
        capacity_sum=sum(caps),
        maximum_group_capacity=max(caps),
        balanced_group_capacity_tokens=[capacities[n] for n in balanced],
        balanced_zero_link_linear_span=(sum(capacities[n] for n in balanced) + max(capacities[n] for n in balanced))
        / parent,
        selected_live_max_to_min_ratio=max(partition) / min(partition),
        normalized_zero_link_linear_stage_makespan=normalized,
        minimum_exposed_tp_comm_fraction_if_equal_compute_efficiency=1 - 1 / normalized,
        boundary_hidden_and_residual_bytes=sum(caps) * 5120 * 2 * 2,
        group_replays_across_two_stages=2 * groups,
        logical_fia_parameter_updates_across_two_stages=16 * groups,
    )


def checkpoint_checks(headers, old):
    for key in ("config_sha256", "index_sha256", "tensor_count"):
        assert headers[key] == old[key]
    assert headers["total_payload_bytes"] == old["tensor_payload_bytes"]
    assert not headers["tensor_payloads_read"] and headers["safetensors_bytes_read"] == 149632
    assert {f["name"]: f["header_sha256"] for f in headers["files"]} == {
        f["name"]: f["header_sha256"] for f in old["files"]
    }
    for key, schema in headers["layer_schemas"].items():
        canonical = json.dumps(schema, sort_keys=True, separators=(",", ":")).encode("ascii")
        assert hashlib.sha256(canonical).hexdigest() == key
        for tensor in schema.values():
            size = {"BF16": 2, "F32": 4}[tensor["dtype"]]
            assert math.prod(tensor["shape"]) * size == tensor["bytes"]
    assert [layer["index"] for layer in headers["layers"]] == list(range(64))
    for layer in headers["layers"]:
        assert sum(t["bytes"] for t in headers["layer_schemas"][layer["schema_sha256"]].values()) == layer["bytes"]
    assert (
        sum(layer["bytes"] for layer in headers["layers"]) + headers["other_group_bytes"]["text_nonlayer"]
        == (old["payload_bytes_by_group"]["text_decoder"])
    )


def residency(headers, budget, config):
    counts = budget["workload_residency"]
    # reserve_owner_pages starts at one: page_count is null + all owner pages.
    pages = 1 + counts["maximum_fia_blocks"]
    owners, candidates = counts["logical_requests"] + 1, counts["candidate_slots_including_null"]
    value_heads = config["linear_num_value_heads"]
    key_heads = config["linear_num_key_heads"]
    kd, vd = config["linear_key_head_dim"], config["linear_value_head_dim"]
    conv_width = config["linear_conv_kernel_dim"] - 1 + QUERY - 1
    base = owners * value_heads * kd * vd * 4
    conv = candidates * (2 * key_heads * kd + value_heads * vd) * conv_width * 2
    logs_and_valid = 2 * owners * value_heads * (QUERY * (kd + vd + 8) + 8) * 4
    fia = pages * 128 * 2 * config["num_key_value_heads"] * config["head_dim"] * 2
    layer_types = config["layer_types"]
    per_cut = []
    for cut in (28, 30, 31, 32, 33, 34, 36):
        stages = []
        for stage, (begin, end) in enumerate(((0, cut), (cut, 64))):
            gdn = layer_types[begin:end].count("linear_attention")
            full = end - begin - gdn
            decoder = sum(layer["bytes"] for layer in headers["layers"][begin:end])
            endpoints = headers["other_group_bytes"]["text_embedding" if stage == 0 else "lm_head"]
            if stage == 1:
                endpoints += headers["other_group_bytes"]["text_nonlayer"]
            subtotal = decoder + endpoints + gdn * (base + conv + logs_and_valid) + full * fia
            stages.append(
                dict(
                    stage=stage,
                    layer_range_half_open=[begin, end],
                    gdn_layers=gdn,
                    fia_layers=full,
                    decoder_weights_bytes=decoder,
                    endpoint_weights_bytes=endpoints,
                    compact_base_bytes=gdn * base,
                    candidate_conv_bytes=gdn * conv,
                    two_log_valid_banks_bytes=gdn * logs_and_valid,
                    maximum_fia_bytes_including_null=full * fia,
                    subtotal_before_products_activations_graphs_and_runtime=subtotal,
                    remaining_of_assumed_64_gib_before_unknowns=64 * 1024**3 - subtotal,
                )
            )
        per_cut.append(dict(cut=cut, stages=stages))
    # A balanced PP2 cut and TP2 have the same ideal per-card state footprint.
    balanced = next(row for row in per_cut if row["cut"] == 32)
    tp = budget["estimates"][0]
    assert tp["tensor_parallel_size"] == 2
    compact = tp["compact_direct_allocation_estimate"]
    for stage in balanced["stages"]:
        assert stage["compact_base_bytes"] == compact["base_bytes"]
        assert stage["candidate_conv_bytes"] == compact["decode_conv_bytes"]
        assert stage["two_log_valid_banks_bytes"] == compact["two_log_and_valid_banks_bytes"]
        assert stage["maximum_fia_bytes_including_null"] == tp["maximum_fia_bytes"]
    return dict(
        actual_fia_page_count_including_null=pages,
        old_residency_counter_excludes_null=True,
        old_fia_byte_estimate_already_includes_null=True,
        assumptions=(
            "direct compact FP32 state generalized to full heads; not current kernel support or allocated bytes"
        ),
        excluded_weights="MTP and vision; no new weight replicas; no assertion that the current loader omits these",
        per_cut=per_cut,
    )


def projection_shapes(config, partition_rows):
    h, intermediate = config["hidden_size"], config["intermediate_size"]
    qkvz = 2 * config["linear_num_key_heads"] * config["linear_key_head_dim"]
    value = config["linear_num_value_heads"] * config["linear_value_head_dim"]
    attention = config["num_attention_heads"] * config["head_dim"]
    kv = config["num_key_value_heads"] * config["head_dim"]
    full_specs = {
        "gdn_qkvz": (qkvz + 2 * value, h, "column"),
        "gdn_ba": (2 * config["linear_num_value_heads"], h, "column"),
        "gdn_output": (h, value, "row"),
        "fia_qgate_kv": (2 * attention + 2 * kv, h, "column"),
        "fia_output": (h, attention, "row"),
        "gate_up": (2 * intermediate, h, "column"),
        "down": (h, intermediate, "row"),
    }
    result = {}
    for name, (n, k, sharding) in full_specs.items():
        tp_n, tp_k = (n // 2, k) if sharding == "column" else (n, k // 2)
        variants = []
        for groups, record in partition_rows.items():
            variants.append(
                dict(
                    groups=groups,
                    pp_gemm_m_n_k=[[m, n, k] for m in record["group_capacity_tokens"]],
                    pp_weight_bytes_per_gemm=2 * n * k,
                    pp_weight_only_arithmetic_intensity_flops_per_byte=record["group_capacity_tokens"],
                )
            )
        result[name] = dict(
            full_weight_n_k=[n, k],
            tp2_gemm_m_n_k=[400, tp_n, tp_k],
            tp2_weight_bytes_per_gemm=2 * tp_n * tp_k,
            tp2_projection_flops=2 * 400 * tp_n * tp_k,
            pp2_variants=variants,
            caveat="B98 only; FLOPs and weight bytes do not model cache, fusion, nonlinears or kernel timing",
        )
    return result


def compute_balance(headers, config):
    per_layer = []
    for layer in headers["layers"]:
        schema = headers["layer_schemas"][layer["schema_sha256"]]
        # Includes only linear projections, not the conv weight or 1-D norms.
        per_layer.append(sum(math.prod(t["shape"]) for t in schema.values() if len(t["shape"]) == 2))
    decoder = sum(per_layer)
    head = math.prod(headers["text_nonlayer_tensors"]["lm_head.weight"]["shape"])
    rows = []
    for cut in (30, 31, 32, 33, 34, 36):
        a, b = sum(per_layer[:cut]), sum(per_layer[cut:])
        rows.append(
            dict(
                cut=cut,
                decoder_projection_elements=[a, b],
                decoder_max_to_mean=max(a, b) / (decoder / 2),
                including_all_q4_logits_projection_elements=[a, b + head],
                including_all_q4_logits_max_to_mean=max(a, b + head) / ((decoder + head) / 2),
            )
        )
    assert config["hidden_size"] * config["vocab_size"] == head
    return dict(
        endpoint_of_historical_full_probe="all 64 decoders and final norm; not lm_head or service latency",
        dense_balance_only_not_stage_time=rows,
        caveat="logits row selection and sampling must be defined before claiming a complete inference endpoint",
    )


def historical_capture_residency():
    path = RESULTS / "fia-sink-catalog-bridge-v1/deployed-tools/q4-tile-full-replay-v8-queue0.rank0.json"
    report = read(path)
    assert report["trace_sha256"] == TRACE_SHA256
    memory = {record["phase"]: record for record in report["memory"]}
    before, after = memory["compact_decode_transition"], memory["all_required_default_graphs_captured"]
    declared_sha = report["sources"]["multi_graph_gdn_q4_full_probe.py"]
    return dict(
        file=str(path.relative_to(ROOT)).replace("\\", "/"),
        sha256=digest(path),
        raw_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        scope=report["scope"],
        original_status=report["status"],
        rank=report["rank"],
        declared_source_sha256=declared_sha,
        declared_source_matches_current_probe=declared_sha
        == digest(ROOT / "benchmarks/multi_graph_gdn_q4_full_probe.py"),
        graph_count=report["capture"]["graphs"],
        bucket_count=len(report["capture"]["required_default_buckets"]),
        before=before,
        after=after,
        allocated_phase_delta=after["allocated_bytes"] - before["allocated_bytes"],
        reserved_phase_delta=after["reserved_bytes"] - before["reserved_bytes"],
        final_reserved=memory["complete_dynamic_trials"]["reserved_bytes"],
        limitation=(
            "Historical rank0 report only; phase delta includes warmup and workspace changes, is not pure graph bytes; "
            "runtime allocations outside torch and per-mode attribution unknown; source version not recovered"
        ),
    )


def build():
    trace_path = RESULTS / "structural-dynamic-trace-v1.json"
    trace_doc = read(trace_path)
    trace = trace_doc["trace"]
    canonical = dict(trace)
    canonical["initial_prefix_by_request"] = {int(k): v for k, v in trace["initial_prefix_by_request"].items()}
    assert (
        hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("ascii")).hexdigest()
        == (trace_doc["trace_sha256"])
        == TRACE_SHA256
    )
    budget_path = RESULTS / "structural-hybrid-memory-budget-v2-compact.json"
    budget = read(budget_path)
    headers = read(HERE / "checkpoint-headers.json")
    checkpoint_checks(headers, budget["checkpoint"])
    config = budget["checkpoint"]["config"]["text_config"]
    selection_path = RESULTS / "two-layer-fused-v1/analysis.json"
    catalog = read(selection_path)["selection"]["capture_sizes"]
    capacities = capacity_table(catalog, max(step["batch"] for step in trace["steps"]))
    partitions = partition_table(capacities, max(GROUP_COUNTS))
    candidates, b98 = [], {}
    for groups in GROUP_COUNTS:
        rows = []
        for step in trace["steps"]:
            row = partition_record(step["batch"], groups, capacities, partitions[groups][step["batch"]])
            row["step"] = step["step"]
            rows.append(row)
        total_parent = sum(row["padding_tokens"] for row in rows)
        total_pp = sum(row["capacity_sum"] for row in rows)
        total_span = sum(row["capacity_sum"] + row["maximum_group_capacity"] for row in rows)
        total_balanced_span = sum(
            sum(row["balanced_group_capacity_tokens"]) + max(row["balanced_group_capacity_tokens"]) for row in rows
        )
        b98[groups] = partition_record(98, groups, capacities, partitions[groups][98])
        signatures = sorted({cap for row in rows for cap in row["group_capacity_tokens"]})
        candidates.append(
            dict(
                groups=groups,
                capacity_sum=total_pp,
                padding_capacity_sum=total_parent,
                capacity_reduction_pct=100 * (1 - total_pp / total_parent),
                normalized_capacity_weighted_linear_span=total_span / total_parent,
                normalized_capacity_weighted_balanced_linear_span=total_balanced_span / total_parent,
                required_exposed_tp_comm_fraction_if_equal_efficiency=1 - total_parent / total_span,
                steps_with_selected_live_ratio_above_two=sum(row["selected_live_max_to_min_ratio"] > 2 for row in rows),
                stage_capacity_signatures_before_address_roles=signatures,
                caveat="proportional-compute screen, not measured latency, capture count or allocated memory",
                rows=rows,
            )
        )
    return dict(
        status="offline_architecture_screen_not_performance_prediction_or_runtime_qualification",
        repository_hash_policy="SHA256 after CRLF-to-LF normalization; checkpoint header hashes are raw bytes",
        inputs={
            str(path.relative_to(ROOT)).replace("\\", "/"): digest(path)
            for path in (trace_path, budget_path, selection_path, HERE / "checkpoint-headers.json")
        },
        trace_sha256=TRACE_SHA256,
        all_steps=len(trace["steps"]),
        batch_histogram=dict(sorted(Counter(step["batch"] for step in trace["steps"]).items())),
        capture_sizes=catalog,
        protocol="positive whole Q4 request groups; current step fully drains; no request postponement",
        group_assignment="sorted group live counts, contiguous slices of the original step request order",
        partitions=candidates,
        batch98=b98,
        pp2_residency=residency(headers, budget, config),
        batch98_projection_shapes=projection_shapes(config, b98),
        stage_dense_balance=compute_balance(headers, config),
        historical_capture_residency=historical_capture_residency(),
        source_sha256={
            path: digest(ROOT / path)
            for path in (
                "benchmarks/multi_graph_gdn_q4_full_metadata.py",
                "benchmarks/multi_graph_gdn_q4_log_backend.py",
                "benchmarks/multi_graph_gdn_q4_full_loader.py",
                "benchmarks/multi_graph_gdn_q4_full_probe.py",
                "benchmarks/multi_graph_gdn_q4_tile_pipeline_kernel/tile_pipeline_kernel.cpp",
                "vllm_ascend/worker/multi_graph/distributed.py",
            )
        },
        kernel_qualification_gap=dict(
            current_tp2_key_and_value_heads=[8, 24],
            required_pp2_key_and_value_heads=[16, 48],
            currently_supported=False,
            unchanged_loader_and_coordinator_guards=True,
        ),
        conditional_sensitivity=[
            dict(
                groups=groups,
                normalized_b98_linear_span=b98[groups]["normalized_zero_link_linear_stage_makespan"],
                compute_time_inflation_factor=alpha,
                required_exposed_tp_comm_fraction=1
                - 1 / (alpha * b98[groups]["normalized_zero_link_linear_stage_makespan"]),
            )
            for groups in GROUP_COUNTS
            for alpha in (1.0, 1.1, 1.25, 1.5)
        ],
        omitted_from_zero_link_scenarios=[
            "capacity-only oracle may prefer tiny tail groups; balanced constructions are reported separately",
            "actual full-width kernel timings and interference",
            "stage imbalance from attention sequence lengths and non-projection work",
            "real FIA host update, metadata H2D, compact state commits and graph submission",
            "P2P latency/bandwidth, packing, concurrent transport and final restoration",
            "buffer lifetimes, address roles, captured workspaces and allocator residency",
            "model parallel change contribution vs independent graph contribution",
        ],
    )


if __name__ == "__main__":
    output = HERE / "audit.json"
    output.write_text(json.dumps(build(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {output.name}: offline screening only")
