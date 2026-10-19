# SPDX-License-Identifier: Apache-2.0
"""CPU audits for the bounded consumer diagnostic and native replay grouping."""

import copy
import hashlib
import importlib
import json
import math
import struct
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]


@pytest.fixture(scope="module")
def module():
    sys.path.insert(0, str(ROOT / "benchmarks"))
    loaded = importlib.import_module("aclgraph_fia_sink_consumer_analyze")
    yield loaded
    sys.path.remove(str(ROOT / "benchmarks"))


@pytest.fixture(scope="module")
def expected(module):
    return module.expected_contract()


def identity(shape, label, dtype="torch.bfloat16"):
    return dict(shape=shape, dtype=dtype, sha256=hashlib.sha256(label.encode()).hexdigest())


def native_plan(record):
    helper = sys.modules["aclgraph_fia_sink_stage4_whole_plan"]
    maximum, batch = max(record["kv_lengths"]), record["physical_batch"]
    words = [0] * 1024
    words[576:586] = [512, min(512, maximum), 8, 0, 0, 0, 4, maximum, batch, batch]
    words = helper.whole_plan(record, words, 20)
    return dict(
        words=words,
        sha256=hashlib.sha256(struct.pack("<1024i", *words)).hexdigest(),
        whole_coverage=helper.validate_whole_plan(record, words, cores=20),
        current_exact_base_verified=True,
    )


def base_identities(module, expected):
    shapes = dict(query=[512, 12, 256], key=[expected["schedule"]["owner_pages"]["page_count"], 128, 2, 256])
    shapes["value"] = shapes["key"]
    inputs, ranges, cursor = [], [], 100_000
    for occurrence in range(16):
        inputs.append({name: identity(shape, f"{occurrence}/{name}") for name, shape in shapes.items()})
        for name, shape in shapes.items():
            size = math.prod(shape) * 2
            ranges.append(dict(occurrence=occurrence, name=name, start=cursor, end=cursor + size))
            cursor += size + 4096
    return dict(occurrences=inputs, sha256=module.evidence.digest(inputs)), ranges


def descriptors(expected, capacity, mode):
    sink = mode != "legacy"
    shapes = dict(
        q_lengths=[capacity // 4],
        kv_lengths=[capacity // 4],
        block_table=[capacity // 4, expected["schedule"]["owner_pages"]["width"]],
        atten_mask=[2048, 2048],
    )
    shapes["metadata" if sink else "workspace0"] = [1024]
    shapes.update({f"{name}{index}": [capacity, 12, 256] for name in ("query", "output") for index in range(16)})
    addresses = {
        name: dict(
            pointer=(index + 1) * 1_000_000, shape=shape, stride=[math.prod(shape[i + 1 :]) for i in range(len(shape))]
        )
        for index, (name, shape) in enumerate(shapes.items())
    }
    prepared = dict(
        mode="sink" if sink else "legacy",
        physical_tokens=capacity,
        physical_requests=capacity // 4,
        host_pinned=dict.fromkeys(("q_lengths", "kv_lengths", "block_table") if sink else ("block_table",), True),
        stable_host_allocations=True,
        stable_device_allocations=True,
        host_allocation_count=3 if sink else 1,
        device_allocation_count=0,
        setup_copies=dict(q_lengths=int(sink), kv_lengths=0, block_table=0),
        step_copies=dict(q_lengths=0, kv_lengths=20 if sink else 0, block_table=20),
        prepared_steps=20,
        completed_steps=20,
        inflight=False,
        q4_invariant=dict(query_length=4, cumulative_q=list(range(4, capacity + 1, 4))),
    )
    return addresses, prepared


def legacy_contract(record, count=2):
    return dict(
        task_count=16,
        workspace_bytes=[1024],
        current_query_lengths=record["actual_seq_qlen"],
        current_kv_lengths=record["kv_lengths"],
        update_calls=count,
        updated_tasks=count * 16,
        instrumentation_enabled=True,
    )


def qualification(record, mode, count):
    capacity, rows = record["physical_tokens"], []
    for occurrence in (0, 7, 15):
        for row in (0, record["batch"] // 2, record["batch"] - 1):
            kv = record["kv_lengths"][row]
            pages = record["block_table"][row][: (kv + 127) // 128]
            rows.append(
                dict(
                    occurrence=occurrence,
                    row=row,
                    kv_length=kv,
                    original_page_ids=pages,
                    local_page_ids=list(range(len(pages))),
                    max_abs_error=0.0003,
                )
            )
    value = dict(
        capacity=capacity,
        step=record["step"],
        private_occurrences=16,
        finite=True,
        all_outputs_finite=True,
        exact_zero_padding=True,
        stable_addresses=True,
        exact_descriptors_read_only=True,
        fp32_sampled_rows=rows,
        output_hashes=[identity([capacity, 12, 256], f"output/{capacity}/{index}") for index in range(16)],
    )
    if mode == "legacy":
        value.update(legacy=legacy_contract(record, count), legacy_update_calls_delta=1, legacy_updated_tasks_delta=16)
    else:
        value["plan"] = native_plan(record)
    return value


def native_identity(module, mode):
    controls, policy = module.controls, "upstream" if mode == "legacy" else mode
    library, digest = controls.LIBRARIES[policy]
    return dict(
        policy_provenance=dict(
            policy=policy,
            library="/vendor/op_impl/cpu/aicpu_kernel/impl/" + library,
            library_sha256=digest,
            opapi_sha256=controls.OPAPI_SHA256,
        ),
        opp_artifacts={
            "op_api/lib/libcust_opapi.so": controls.OPAPI_SHA256,
            "op_impl/cpu/aicpu_kernel/impl/" + library: digest,
            "op_impl/cpu/config/cust_aicpu_kernel.json": controls.DISPATCH_CONFIGS[policy],
        },
        final_native_provenance=dict(
            alternate_sink_symbols_absent=True,
            selected_library_handle_retained=True,
            libraries=[dict(path="/vendor/op_api/lib/libcust_opapi.so", sha256=controls.OPAPI_SHA256)],
            sink_symbols=[
                dict(symbol=name, source="/vendor/op_api/lib/libcust_opapi.so", address="0x1234")
                for name in controls.REQUIRED_SYMBOLS
            ],
        ),
    )


def sample_rows(module, records, trial, mode):
    rows = []
    for repeat in range(10):
        offset = (repeat + trial) % 2
        for record in records[offset:] + records[:offset]:
            rows.append(
                dict(
                    trial=trial,
                    mode=mode,
                    policy=mode,
                    repeat=repeat,
                    step=record["step"],
                    batch=record["batch"],
                    capacity=record["physical_tokens"],
                    wall_us=200 + (0 if mode == "whole" else 10),
                    preparation_enqueue_us=5,
                    submit_join_cpu_us=100,
                    queue_span_us=180,
                    legacy_update_calls_delta=None if mode == "legacy" else 0,
                    legacy_updated_tasks_delta=None if mode == "legacy" else 0,
                    legacy_counter_instrumented=False,
                )
            )
    return rows


def make_report(module, expected, trial, mode, child):
    """Construct all evidence independently so mutations exercise strict gates."""
    inputs, ranges = base_identities(module, expected)
    mask = identity([2048, 2048], "mask", "torch.bool")
    frames = {str(cap): [identity([cap, 12, 256], f"frame{cap}/{index}") for index in range(16)] for cap in (400, 512)}
    addresses, prepared = {}, {}
    for capacity in (400, 512):
        addresses[str(capacity)], prepared[str(capacity)] = descriptors(expected, capacity, mode)
    report = dict(
        state="completed",
        stage="completed",
        returncode=0,
        npu_work_executed=True,
        performance_measured=True,
        full_trace_measured=False,
        full_model_qualified=False,
        end_to_end_gain_established=False,
        mode=mode,
        policy=mode,
        trial=trial,
        repeats=10,
        occurrences=16,
        warmup_replays=5,
        profile_repeats=3,
        child_pid=child,
        device_name="Ascend910B3",
        torch_version="2.10.0+cpu",
        torch_npu_version="2.10.0.post4",
        canonical_digest=module.evidence.CANONICAL_TRACE_SHA256,
        selected_records=expected["selected"],
        trace_contract=expected["trace_contract"],
        core_limit=dict(cube_core_num=20, vector_core_num=40),
        independent_graph_pool_per_bucket=True,
        separate_qualification=True,
        frame_queries_match_base_prefixes=True,
        timed_samples_individually_qualified=False,
        actual_task_audit_scope="pre/post qualification only; timing and profiles use original graph.update",
        warmups=[dict(capacity=cap, replays=5, outside_timing=True) for cap in (400, 512)],
        input_identity=inputs,
        post_input_identity=copy.deepcopy(inputs),
        private_storage_ranges=ranges,
        mask_identity=mask,
        post_mask_identity=copy.deepcopy(mask),
        frame_query_identity=frames,
        post_frame_query_identity=copy.deepcopy(frames),
        captured_addresses=addresses,
        prepared_descriptors=prepared,
        pre_checks=[qualification(record, mode, 1) for record in expected["selected"]],
        post_checks=[qualification(record, mode, 2) for record in expected["selected"]],
        scripts=copy.deepcopy(expected["scripts"]),
        extension_sha256=module.controls.EXTENSION_SHA256,
        installed_catalog=dict(
            installed_catalog_revalidated=True,
            selection=dict(
                capture_sizes=expected["schedule"]["default_capture_sizes"],
                dispatched_tokens=expected["independent"]["dispatched_tokens"],
                config_source="installed EngineArgs defaults; no capture sizes supplied",
            ),
        ),
        samples=sample_rows(module, expected["selected"], trial, mode),
        **native_identity(module, mode),
    )
    if mode == "legacy":
        report["legacy_contract"] = {
            str(record["physical_tokens"]): legacy_contract(record) for record in expected["selected"]
        }
    else:
        report["metadata_preflight"] = [
            dict(
                capacity=record["physical_tokens"],
                record=record,
                verified_before_attention_capture=True,
                **native_plan(record),
            )
            for record in expected["selected"]
        ]
    return report


def event(name, ts, dur, task_type=None, task_id=0, model=7, stream=9):
    args = (
        {"Task Type": task_type, "Task Id": task_id, "Subtask Id": 0, "Model Id": model, "Physic Stream Id": stream}
        if task_type
        else {}
    )
    return dict(name=name, ts=str(ts), dur=dur, ph="X", cat="cpu_op", args=args)


def profile_events(mode, capacity):
    events = []
    for repeat in range(3):
        start = 1000 * repeat
        events.extend(
            [
                event(f"consumer/{mode}/cap{capacity}/repeat{repeat}", start, 500),
                event("MODEL_EXECUTE", start + 10, 1, "MODEL_EXECUTE", model=4294967295, stream=4),
            ]
        )
        if mode != "legacy":
            events.extend(
                [
                    event("ZerosLike", start + 12, 1, "KERNEL_AIVEC", 0),
                    event("FusedInferAttentionScoreV2SinkMetadata", start + 14, 2, "KERNEL_AICPU", 1),
                ]
            )
        for index in range(16):
            ts = start + 20 + index * 20
            events.append(
                event(
                    "FusedInferAttentionScore" if mode == "legacy" else "FusedInferAttentionScoreV2Sink",
                    ts,
                    15,
                    "KERNEL_MIX_AIC",
                    index + 2,
                )
            )
            if mode == "legacy":
                events.extend(
                    [
                        event("npu::npu_fused_infer_attention_score", ts - 5, 2),
                        event("aclnnFusedInferAttentionScoreV3", ts - 4, 1),
                        event("MEMCPY_ASYNC", ts - 2, 1, "PCIE_DMA_SQE", 100 + index, 4294967295, 14),
                        event("MEM_WAIT_VALUE", ts - 3, 1, "MEM_WAIT_VALUE", 80),
                        event("MEM_WAIT_VALUE", ts - 2, 1, "MEM_WAIT_VALUE", 80),
                    ]
                )
        events.append(event("NOTIFY_RECORD", start + 400, 0.02, "NOTIFY_RECORD", 18))
    return events


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="ascii")


def make_bundle(module, expected, root):
    bundle = dict(
        state="completed",
        returncode=0,
        performance_measured=True,
        full_trace_measured=False,
        full_model_qualified=False,
        end_to_end_gain_established=False,
        design=module.expected_design(),
        started_unix=1,
        finished_unix=100,
        runs=[],
    )
    for trial, order in enumerate(module.ORDERS):
        for mode in order:
            index = len(bundle["runs"])
            path = root / f"run{index}" / "consumer-probe.json"
            report = make_report(module, expected, trial, mode, 1000 + index)
            report["profiles"] = []
            for capacity in (400, 512):
                trace = path.parent / "profiles" / f"cap{capacity}" / "trace.json"
                write_json(trace, profile_events(mode, capacity))
                report["profiles"].append(
                    dict(
                        capacity=capacity,
                        path=trace.relative_to(path.parent).as_posix(),
                        sha256=module.evidence.sha256(trace),
                        repeats=3,
                        outside_timing=True,
                        labels=[f"consumer/{mode}/cap{capacity}/repeat{r}" for r in range(3)],
                        legacy_update_calls_delta=None if mode == "legacy" else 0,
                        legacy_updated_tasks_delta=None if mode == "legacy" else 0,
                        legacy_counter_instrumented=False,
                        preparation_in_trace_outside_label=True,
                        label_scope="native plan/update + replay + final join; prepare/H2D completed before label",
                    )
                )
            if mode == "legacy":
                snapshot = path.parent / "runtime" / "updatable_graph.py"
                snapshot.parent.mkdir()
                snapshot.write_bytes((ROOT / "vllm_ascend/compilation/updatable_graph.py").read_bytes())
                report["runtime_sources"] = {
                    "vllm_ascend/compilation/updatable_graph.py": dict(
                        sha256=expected["updatable_graph_sha256"],
                        snapshot_sha256=expected["updatable_graph_sha256"],
                        snapshot_path="runtime/updatable_graph.py",
                    )
                }
            write_json(path, report)
            bundle["runs"].append(
                dict(
                    trial=trial,
                    mode=mode,
                    process_order_index=index,
                    pid=100 + index,
                    launched_unix=2 + index * 10,
                    finished_unix=9 + index * 10,
                    returncode=0,
                    report_path=path.relative_to(root).as_posix(),
                    report_sha256=module.evidence.sha256(path),
                )
            )
    return bundle


def test_complete_bounded_diagnostic(module, expected, tmp_path):
    result = module.analyze(make_bundle(module, expected, tmp_path), base_dir=tmp_path)
    assert result["diagnostic_qualified"], result["issues"] + result["profile"]["issues"]
    assert result["coverage"]["valid_sample_count"] == 120
    assert result["profile"]["attention_tasks"] == 576
    assert result["profile"]["metadata_tasks"] == 24
    assert result["profile"]["legacy_cpu_update_calls"] == result["profile"]["legacy_pcie_dma_tasks"] == 192
    assert all(
        not result[key] for key in ("full_trace_measured", "full_model_qualified", "end_to_end_gain_established")
    )


@pytest.mark.parametrize("mode", ("upstream", "whole", "legacy"))
def test_native_replay_metrics_exclude_waits_and_nested_cpu(module, mode):
    rows = module.parse_profile(profile_events(mode, 400), mode, 400)
    assert len(rows) == 3
    assert rows[0]["attention_duration_sum_us"] == 240
    assert rows[0]["attention_first_to_last_envelope_us"] == 315
    assert rows[0]["between_attention_noncompute_us"] == 75
    assert rows[0]["cpu_legacy_update_calls"] == (16 if mode == "legacy" else 0)
    assert rows[0]["cpu_legacy_update_duration_sum_us"] == (32 if mode == "legacy" else 0)


@pytest.mark.parametrize(
    "mutation",
    (
        "missing_notify",
        "duplicate_notify",
        "foreign_model",
        "wrong_count",
        "duplicate_task",
        "overlap",
        "missing_execute",
        "cross_stream",
        "outside_completion",
        "relabel",
    ),
)
def test_profile_ambiguity_is_rejected(module, mutation):
    events = profile_events("whole", 400)
    notify = next(row for row in events if row["name"] == "NOTIFY_RECORD")
    kernel = next(row for row in events if row["args"].get("Task Type") == "KERNEL_MIX_AIC")
    if mutation == "missing_notify":
        events.remove(notify)
    elif mutation == "duplicate_notify":
        events.append(copy.deepcopy(notify))
    elif mutation == "foreign_model":
        kernel["args"]["Model Id"] = 99
    elif mutation == "wrong_count":
        events.remove(kernel)
    elif mutation == "duplicate_task":
        events.append(copy.deepcopy(kernel))
    elif mutation == "overlap":
        kernel["dur"] = 40
    elif mutation == "missing_execute":
        events.remove(next(row for row in events if row["name"] == "MODEL_EXECUTE"))
    elif mutation == "cross_stream":
        kernel["args"]["Physic Stream Id"] = 77
    elif mutation == "outside_completion":
        kernel["dur"] = 450
    else:
        events[0]["name"] = "consumer/whole/cap400/repeat1"
    with pytest.raises(ValueError):
        module.parse_profile(events, "whole", 400)


def test_decimal_clock_preserves_small_durations(module):
    start, end = module.interval(dict(ts="1792345551073407.076", dur=0.02))
    assert str(end - start) == "0.020"
    assert module.union_duration([(start, end), (start, end)]) == 0.02


def test_committed_native_trace_handles_repeated_wait_fragments(module):
    events = json.loads((ROOT / "benchmarks/results/multi_graph_structural/fia-sink-stage3-profile.json").read_text())
    for old_mode, mode in (("shared", "upstream"), ("legacy", "legacy")):
        for repeat in range(3):
            label = next(row for row in events if row.get("name") == f"stage3/{old_mode}/{repeat}")
            parsed = module.profile_replay(events, label, mode)
            assert parsed["attention_tasks"] == 16
            assert parsed["completion_notifications"] == 1
            if mode == "legacy":
                assert parsed["cpu_legacy_update_calls"] == parsed["legacy_pcie_dma_tasks"] == 16
                assert parsed["mem_wait_value_tasks"] > 16
                assert parsed["mem_wait_value_duration_union_us"] <= parsed["mem_wait_value_duration_sum_us"] + 1e-6


@pytest.mark.parametrize("key,value", (("repeats", True), ("occurrences", 1), ("step_order", "altered")))
def test_design_or_sample_scope_cannot_drift(module, expected, tmp_path, key, value):
    bundle = make_bundle(module, expected, tmp_path)
    if key == "step_order":
        bundle["runs"][0]["mode"] = "legacy"
    else:
        bundle["design"][key] = value
    result = module.analyze(bundle, base_dir=tmp_path)
    assert not result["protocol_qualified"]


def test_raw_profile_sha_and_path_traversal_fail_closed(module, expected, tmp_path):
    bundle = make_bundle(module, expected, tmp_path)
    path = tmp_path / "run0" / "profiles" / "cap400" / "trace.json"
    path.write_text("[]", encoding="ascii")
    result = module.analyze(bundle, base_dir=tmp_path)
    assert result["protocol_qualified"]
    assert not result["diagnostic_qualified"]
    assert any("SHA256" in issue for issue in result["profile"]["issues"])
    with pytest.raises(ValueError):
        module.controls.resolve_report(dict(report_path="../outside.json"), tmp_path)


@pytest.mark.parametrize(
    "mutation", ("copy_count", "oracle_pages", "host_lengths", "timed_instrumentation", "source", "input_bytes")
)
def test_report_contract_mutations_fail(module, expected, mutation):
    report = make_report(module, expected, 0, "legacy", 1000)
    if mutation == "copy_count":
        report["prepared_descriptors"]["400"]["step_copies"]["kv_lengths"] = 20
    elif mutation == "oracle_pages":
        report["pre_checks"][0]["fp32_sampled_rows"][0]["original_page_ids"] = [999]
    elif mutation == "host_lengths":
        report["pre_checks"][0]["legacy"]["current_kv_lengths"] = [0]
    elif mutation == "source":
        report["scripts"]["aclgraph_fia_sink_stage7_probe.py"] = "0" * 64
    elif mutation == "input_bytes":
        report["post_input_identity"]["sha256"] = "0" * 64
    else:
        report["legacy_contract"]["400"]["update_calls"] = 20
    assert module.report_issues(report, dict(trial=0, mode="legacy"), expected)


def test_sample_matrix_rejects_order_duplicates_and_fabricated_update_counts(module, expected):
    report = make_report(module, expected, 0, "legacy", 1000)
    report["samples"][0]["legacy_update_calls_delta"] = 1
    report["samples"][0]["legacy_counter_instrumented"] = True
    report["samples"].append(report["samples"][0])
    _, coverage = module.collect_samples([(dict(trial=0, mode="legacy"), report)], expected)
    assert not coverage["complete"]
    assert any("duplicate" in issue for issue in coverage["issues"])
    assert any("instrumentation" in issue or "uninstrumented" in issue for issue in coverage["issues"])
