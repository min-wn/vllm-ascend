"""CPU contract tests for the complete dynamic attention-trace analyzer."""

import copy
import hashlib
import importlib
import json
import math
import struct
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]


@pytest.fixture(scope="module")
def module():
    sys.path.insert(0, str(ROOT / "benchmarks"))
    loaded = importlib.import_module("aclgraph_fia_sink_stage7_analyze")
    yield loaded
    sys.path.remove(str(ROOT / "benchmarks"))


@pytest.fixture(scope="module")
def expected(module):
    return module.expected_contract()


def identity(shape, label, dtype="torch.bfloat16"):
    return dict(shape=shape, dtype=dtype, sha256=hashlib.sha256(label.encode()).hexdigest())


def make_inputs(module, expected):
    shapes = dict(query=[512, 12, 256], key=[expected["schedule"]["owner_pages"]["page_count"], 128, 2, 256])
    shapes["value"] = shapes["key"]
    entries, ranges, start = [], [], 1_000_000
    for occurrence in range(16):
        entries.append({name: identity(shape, f"{occurrence}{name}") for name, shape in shapes.items()})
        for name, shape in shapes.items():
            end = start + math.prod(shape) * 2
            ranges.append(dict(occurrence=occurrence, name=name, start=start, end=end))
            start = end + 4096
    return dict(occurrences=entries, sha256=module.evidence.digest(entries)), ranges


def plan(module, record):
    helper = sys.modules["aclgraph_fia_sink_stage4_whole_plan"]
    words = [0] * 1024
    maximum, batch = max(record["kv_lengths"]), record["physical_batch"]
    words[576:586] = [512, min(512, maximum), 8, 0, 0, 0, 4, maximum, batch, batch]
    words = helper.whole_plan(record, words, 20)
    return dict(
        words=words,
        sha256=hashlib.sha256(struct.pack("<1024i", *words)).hexdigest(),
        metadata={},
        whole_coverage=helper.validate_whole_plan(record, words, cores=20),
        current_exact_base_verified=True,
    )


def descriptor(expected, capacity, count):
    shapes = dict(
        q_lengths=[capacity // 4],
        kv_lengths=[capacity // 4],
        block_table=[capacity // 4, expected["schedule"]["owner_pages"]["width"]],
        atten_mask=[2048, 2048],
        metadata=[1024],
    )
    shapes.update({f"{name}{index}": [capacity, 12, 256] for name in ("query", "output") for index in range(16)})
    addresses = {
        name: dict(
            pointer=1000 + index * 1_000_000, shape=shape, stride=[math.prod(shape[i + 1 :]) for i in range(len(shape))]
        )
        for index, (name, shape) in enumerate(shapes.items())
    }
    summary = dict(
        mode="sink",
        physical_tokens=capacity,
        physical_requests=capacity // 4,
        host_pinned=dict.fromkeys(("q_lengths", "kv_lengths", "block_table"), True),
        stable_host_allocations=True,
        stable_device_allocations=True,
        host_allocation_count=3,
        device_allocation_count=0,
        setup_copies=dict(q_lengths=1, kv_lengths=0, block_table=0),
        step_copies=dict(q_lengths=0, kv_lengths=count, block_table=count),
        prepared_steps=count,
        completed_steps=count,
        inflight=False,
        q4_invariant=dict(query_length=4, cumulative_q=list(range(4, capacity + 1, 4))),
    )
    return addresses, summary


@pytest.fixture(scope="module")
def reference_report(module, expected):
    """Build complete temporal evidence so mutations test individual audit gates."""
    checks, preflight = [], []
    for record in expected["records"]:
        occurrence, capacity = record["step"] % 16, record["physical_tokens"]
        rows = []
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
        checks.append(
            dict(
                step=record["step"],
                batch=record["batch"],
                capacity=capacity,
                private_occurrences=16,
                oracle_occurrence=occurrence,
                fp32_sampled_rows=rows,
                output_hashes=[identity([capacity, 12, 256], f"output{record['step']}{index}") for index in range(16)],
                finite=True,
                all_outputs_finite=True,
                exact_zero_padding=True,
                stable_addresses=True,
                exact_descriptors_read_only=True,
                plan=plan(module, record),
            )
        )
    for record in expected["captures"]:
        preflight.append(
            dict(
                capacity=record["physical_tokens"],
                record=record,
                verified_before_attention_capture=True,
                **copy.deepcopy(checks[record["step"]]["plan"]),
            )
        )
    inputs, ranges = make_inputs(module, expected)
    frames = {
        str(capacity): [identity([capacity, 12, 256], f"frame{capacity}{index}") for index in range(16)]
        for capacity in expected["capacities"]
    }
    mask = identity([2048, 2048], "mask", "torch.bool")
    addresses, prepared = {}, {}
    counts = Counter(record["physical_tokens"] for record in expected["records"])
    for capacity in expected["capacities"]:
        addresses[str(capacity)], prepared[str(capacity)] = descriptor(expected, capacity, 8 * counts[capacity])
    c = module.controls
    library, library_hash = c.LIBRARIES["whole"]
    return dict(
        state="completed",
        stage="completed",
        returncode=0,
        npu_work_executed=True,
        performance_measured=True,
        full_trace_measured=True,
        full_model_qualified=False,
        end_to_end_gain_established=False,
        policy="whole",
        trial=0,
        repeats=3,
        occurrences=16,
        started_unix=2,
        finished_unix=8,
        child_pid=1000,
        device_name="Ascend910B3",
        torch_version="2.10.0+cpu",
        torch_npu_version="2.10.0.post4",
        canonical_digest=module.evidence.CANONICAL_TRACE_SHA256,
        selected_records=expected["records"],
        capture_records=expected["captures"],
        trace_contract=expected["trace_contract"],
        core_limit=dict(cube_core_num=20, vector_core_num=40),
        independent_graph_pool_per_bucket=True,
        frame_queries_match_base_prefixes=True,
        separate_qualification=True,
        timed_samples_individually_qualified=False,
        warmup_passes=[dict(repeat=r, steps=list(range(128)), completed=True) for r in range(3)],
        input_identity=inputs,
        post_input_identity=copy.deepcopy(inputs),
        private_storage_ranges=ranges,
        mask_identity=mask,
        post_mask_identity=copy.deepcopy(mask),
        frame_query_identity=frames,
        post_frame_query_identity=copy.deepcopy(frames),
        captured_addresses=addresses,
        prepared_descriptors=prepared,
        metadata_preflight=preflight,
        pre_checks=checks,
        post_checks=copy.deepcopy(checks),
        installed_catalog=dict(
            installed_catalog_revalidated=True,
            selection=dict(
                capture_sizes=expected["schedule"]["default_capture_sizes"],
                dispatched_tokens=expected["independent"]["dispatched_tokens"],
                config_source="installed EngineArgs defaults; no capture sizes supplied",
            ),
        ),
        scripts=expected["scripts"],
        extension_sha256=c.EXTENSION_SHA256,
        policy_provenance=dict(
            policy="whole",
            library="/vendor/op_impl/cpu/aicpu_kernel/impl/" + library,
            library_sha256=library_hash,
            opapi_sha256=c.OPAPI_SHA256,
        ),
        opp_artifacts={
            "op_api/lib/libcust_opapi.so": c.OPAPI_SHA256,
            "op_impl/cpu/aicpu_kernel/impl/" + library: library_hash,
            "op_impl/cpu/config/cust_aicpu_kernel.json": c.DISPATCH_CONFIGS["whole"],
            "op_impl/ai_core/consumer.o": "f" * 64,
        },
        final_native_provenance=dict(
            alternate_sink_symbols_absent=True,
            selected_library_handle_retained=True,
            libraries=[dict(path="/vendor/op_api/lib/libcust_opapi.so", sha256=c.OPAPI_SHA256)],
            sink_symbols=[
                dict(symbol=name, source="/vendor/op_api/lib/libcust_opapi.so", address="0x1234")
                for name in c.REQUIRED_SYMBOLS
            ],
        ),
    )


def samples(module, expected, trial, policy):
    result = []
    for repeat in range(3):
        for record in expected["records"]:
            step, capacity = record["step"], record["physical_tokens"]
            base = 100 + capacity / 10 + repeat + trial
            result.append(
                dict(
                    trial=trial,
                    policy=policy,
                    repeat=repeat,
                    step=step,
                    batch=record["batch"],
                    capacity=capacity,
                    **{
                        k: v
                        for k, v in module.transition(record, expected["records"][step - 1]).items()
                        if k in ("previous_capacity", "capacity_transition", "transition_class")
                    },
                    wall_us=base if policy == "upstream" else base - 10,
                    preparation_enqueue_us=4,
                    submit_join_cpu_us=2,
                    queue_span_us=base - 20 if policy == "upstream" else base - 30,
                )
            )
    return result


@pytest.fixture
def timed_runs(module, expected):
    return [
        (dict(trial=trial, policy=policy), dict(samples=samples(module, expected, trial, policy)))
        for trial in range(4)
        for policy in ("upstream", "whole")
    ]


def test_full_matrix_frequency_and_chronological_pairing(module, expected, timed_runs):
    indexed, coverage = module.collect_samples(timed_runs, expected)
    assert coverage["complete"], coverage["issues"]
    assert coverage["valid_sample_count"] == 3072
    result = module.comparison(indexed, expected)
    assert result["overall"]["paired_count"] == 1536
    assert result["overall"]["metrics"]["wall_us"]["whole_faster_count"] == 1536
    assert len(result["per_step"]) == 128
    assert all(item["paired_count"] == 12 for item in result["per_step"])
    counts = Counter(record["physical_tokens"] for record in expected["records"])
    assert all(item["paired_count"] == 12 * counts[item["capacity"]] for item in result["per_capacity"])
    assert sum(item["paired_count"] for item in result["per_transition"]) == 1536
    assert result["per_transition"][0]["paired_count"] == 12


def test_mean_is_weighted_by_real_trace_frequency_and_keeps_outlier(module, expected, timed_runs):
    timed_runs[1][1]["samples"][0]["wall_us"] = 10000
    indexed, coverage = module.collect_samples(timed_runs, expected)
    assert coverage["complete"]
    result = module.comparison(indexed, expected)
    assert len(result["slower_counterexamples"]) == 1
    assert result["slower_counterexamples"][0]["step"] == 0
    left = [x["wall_us"] for (t, p, r, s), x in indexed.items() if p == "upstream"]
    right = [x["wall_us"] for (t, p, r, s), x in indexed.items() if p == "whole"]
    metrics = result["overall"]["metrics"]["wall_us"]
    assert metrics["upstream_reduction_percent"] == pytest.approx(100 * (sum(left) - sum(right)) / sum(left))
    assert metrics["whole"]["count"] == 1536
    assert metrics["whole"]["mean_us"] == pytest.approx(sum(right) / len(right))


@pytest.mark.parametrize(
    "mutation", ["order", "missing", "duplicate", "step_bool", "transition", "previous", "tiny_wall", "wrong_policy"]
)
def test_timing_contract_rejects_temporal_corruption(module, expected, timed_runs, mutation):
    rows = timed_runs[0][1]["samples"]
    if mutation == "order":
        rows[0], rows[1] = rows[1], rows[0]
    elif mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows.append(copy.deepcopy(rows[0]))
    elif mutation == "step_bool":
        rows[0]["step"] = False
    elif mutation == "transition":
        rows[1]["capacity_transition"] = not rows[1]["capacity_transition"]
    elif mutation == "previous":
        rows[1]["previous_capacity"] = 1
    elif mutation == "tiny_wall":
        rows[0]["wall_us"] = 0.001
    else:
        rows[0]["policy"] = "whole"
    _, coverage = module.collect_samples(timed_runs, expected)
    assert not coverage["complete"]


def test_reference_report_qualifies_without_stage6_fabrication(module, expected, reference_report):
    issues = module.report_issues(reference_report, dict(policy="whole", trial=0), expected)
    assert not issues, issues
    summary = module.numerical_summary([(dict(policy="whole", trial=0), reference_report)])
    assert summary["observed_step_checks"] == 256
    assert summary["observed_fp32_rows"] == 768
    assert summary["timed_samples_individually_qualified"] is False


@pytest.mark.parametrize(
    "mutation",
    [
        "pre_short",
        "post_duplicate",
        "occurrence",
        "row_missing",
        "kv",
        "page",
        "localpages",
        "error",
        "finite",
        "fd",
        "payload_tail",
        "verified",
    ],
)
def test_temporal_numeric_and_native_payload_gates(module, expected, reference_report, mutation):
    report = copy.deepcopy(reference_report)
    if mutation == "pre_short":
        report["pre_checks"].pop()
    elif mutation == "post_duplicate":
        report["post_checks"][6] = copy.deepcopy(report["post_checks"][5])
    else:
        item = report["pre_checks"][5]
        oracle = item["fp32_sampled_rows"][0]
        if mutation == "occurrence":
            oracle["occurrence"] = 0
        elif mutation == "row_missing":
            item["fp32_sampled_rows"].pop()
        elif mutation == "kv":
            oracle["kv_length"] += 1
        elif mutation == "page":
            oracle["original_page_ids"][0] += 1
        elif mutation == "localpages":
            oracle["local_page_ids"][0] = 1
        elif mutation == "error":
            oracle["max_abs_error"] = -1
        elif mutation == "finite":
            item["all_outputs_finite"] = False
        elif mutation == "verified":
            item["plan"]["current_exact_base_verified"] = False
        else:
            words = item["plan"]["words"]
            words[8 if mutation == "fd" else 1023] = 1
            item["plan"]["sha256"] = hashlib.sha256(struct.pack("<1024i", *words)).hexdigest()
    assert module.qualification_issues(report, expected, "whole")


def test_output_bitwise_changes_remain_diagnostic(module, expected, reference_report):
    report = copy.deepcopy(reference_report)
    report["post_checks"][50]["output_hashes"][5]["sha256"] = "e" * 64
    assert not module.qualification_issues(report, expected, "whole")
    summary = module.numerical_summary([(dict(trial=0, policy="whole"), report)])
    assert sum(item["identical"] is False for item in summary["output_bitwise_repeatability"]) == 1


def test_same_step_metadata_repeatability_is_diagnostic(module, reference_report):
    report = copy.deepcopy(reference_report)
    report["post_checks"][17]["plan"]["sha256"] = "e" * 64
    summary = module.numerical_summary([(dict(trial=0, policy="whole"), report)])
    assert len(summary["metadata_same_step_repeatability"]) == 128
    changed = [item for item in summary["metadata_same_step_repeatability"] if item["identical"] is False]
    assert len(changed) == 1
    assert changed[0]["step"] == 17
    assert summary["observed_pre_capture_payloads"] == 17
    assert summary["observed_total_payloads"] == 273


@pytest.mark.parametrize(
    "mutation",
    ["frame_bytes", "frame_missing", "alias", "counts", "warmup", "full_scope", "timed_claim", "base_prefix"],
)
def test_dynamic_lifetime_and_measurement_scope(module, expected, reference_report, mutation):
    report = copy.deepcopy(reference_report)
    if mutation == "frame_bytes":
        report["post_frame_query_identity"]["400"][0]["sha256"] = "d" * 64
    elif mutation == "frame_missing":
        report["frame_query_identity"].pop("400")
    elif mutation == "alias":
        table = report["captured_addresses"]["400"]
        table["output1"]["pointer"] = table["output0"]["pointer"]
    elif mutation == "counts":
        report["prepared_descriptors"]["400"]["step_copies"]["kv_lengths"] -= 1
    elif mutation == "warmup":
        report["warmup_passes"][2]["steps"].pop()
    elif mutation == "full_scope":
        report["full_trace_measured"] = False
    elif mutation == "timed_claim":
        report["timed_samples_individually_qualified"] = True
    else:
        report["frame_queries_match_base_prefixes"] = False
    assert module.report_issues(report, dict(trial=0, policy="whole"), expected)


def test_complete_analyze_pipeline_and_raw_protection(module, expected, reference_report, tmp_path, monkeypatch):
    orders = [["upstream", "whole"], ["whole", "upstream"], ["upstream", "whole"], ["whole", "upstream"]]
    bundle = dict(
        state="completed",
        returncode=0,
        started_unix=0,
        finished_unix=100,
        full_trace_measured=True,
        full_model_qualified=False,
        end_to_end_gain_established=False,
        design=dict(
            policies=list(module.POLICIES),
            capacities=expected["capacities"],
            occurrences=16,
            rounds=4,
            repeats=3,
            step_count=128,
            process_orders=orders,
            screen=False,
            seed=1703,
            step_order="unchanged chronological canonical trace",
            qualification_passes=2,
            reference_rows_per_step=3,
            reference_occurrence="step % occurrences",
            warmup_passes_per_repeat=1,
        ),
        runs=[],
    )
    for trial, order in enumerate(orders):
        for policy in order:
            i = len(bundle["runs"])
            r = copy.deepcopy(reference_report)
            r.update(
                policy=policy,
                trial=trial,
                child_pid=1000 + i,
                started_unix=10 * i + 2,
                finished_unix=10 * i + 8,
                samples=samples(module, expected, trial, policy),
            )
            c = module.controls
            library, digest = c.LIBRARIES[policy]
            r["policy_provenance"] = dict(
                policy=policy,
                library="/vendor/op_impl/cpu/aicpu_kernel/impl/" + library,
                library_sha256=digest,
                opapi_sha256=c.OPAPI_SHA256,
            )
            r["opp_artifacts"] = {
                "op_api/lib/libcust_opapi.so": c.OPAPI_SHA256,
                "op_impl/cpu/aicpu_kernel/impl/" + library: digest,
                "op_impl/cpu/config/cust_aicpu_kernel.json": c.DISPATCH_CONFIGS[policy],
                "op_impl/ai_core/consumer.o": "f" * 64,
            }
            path = tmp_path / f"run{i}.json"
            path.write_text(json.dumps(r), encoding="ascii")
            bundle["runs"].append(
                dict(
                    trial=trial,
                    policy=policy,
                    process_order_index=i,
                    report_path=path.name,
                    report_sha256=module.evidence.sha256(path),
                    launched_unix=10 * i + 1,
                    finished_unix=10 * i + 9,
                    pid=100 + i,
                    returncode=0,
                )
            )
    result = module.analyze(bundle, base_dir=tmp_path)
    assert result["protocol_qualified"], result["issues"]
    assert result["full_trace_measured"]
    assert result["coverage"]["valid_sample_count"] == 3072
    assert result["comparison"]["overall"]["paired_count"] == 1536
    assert result["numerical_qualification"]["observed_fp32_rows"] == 6144
    assert result["numerical_qualification"]["observed_step_checks"] == 2048
    assert result["end_to_end_gain_established"] is False
    source = tmp_path / "bundle.json"
    source.write_text(json.dumps(bundle))
    raw = tmp_path / "run0.json"
    before = raw.read_bytes()
    monkeypatch.setattr(sys, "argv", ["analyze", "--input", str(source), "--output", str(raw)])
    with pytest.raises(SystemExit):
        module.main()
    assert raw.read_bytes() == before
    bundle["runs"][0]["report_path"] = "../outside.json"
    assert not module.analyze(bundle, base_dir=tmp_path)["protocol_qualified"]
