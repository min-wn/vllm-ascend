"""CPU evidence-adversarial tests for fresh-process block pairing."""

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


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "benchmarks"))
    return importlib.import_module("aclgraph_fia_sink_stage6_analyze")


def make_plan(module, record):
    helper = sys.modules["aclgraph_fia_sink_stage4_whole_plan"]
    words = [0] * 1024
    maximum = max(record["kv_lengths"])
    batch = record["physical_batch"]
    words[576:586] = [512, min(512, maximum), 8, 0, 0, 0, 4, maximum, batch, batch]
    words = helper.whole_plan(record, words, 20)
    return dict(
        words=words,
        sha256=hashlib.sha256(struct.pack("<1024i", *words)).hexdigest(),
        metadata={},
        whole_coverage=helper.validate_whole_plan(record, words, cores=20),
    )


def make_inputs(module, expected):
    shapes = dict(
        query=[max(expected["schedule"]["required_capture_sizes"]), 12, 256],
        key=[expected["schedule"]["owner_pages"]["page_count"], 128, 2, 256],
        value=[expected["schedule"]["owner_pages"]["page_count"], 128, 2, 256],
    )
    entries, ranges, start = [], [], 1_000_000
    for occurrence in range(16):
        entry = {}
        for name, shape in shapes.items():
            entry[name] = dict(
                shape=shape, dtype="torch.bfloat16", sha256=hashlib.sha256(f"{occurrence}{name}".encode()).hexdigest()
            )
            end = start + 2 * math.prod(shape)
            ranges.append(dict(occurrence=occurrence, name=name, start=start, end=end))
            start = end + 4096
        entries.append(entry)
    return dict(occurrences=entries, sha256=module.evidence.digest(entries)), ranges


def make_descriptors(capacity):
    return dict(
        mode="sink",
        physical_tokens=capacity,
        physical_requests=capacity // 4,
        host_pinned=dict.fromkeys(("q_lengths", "kv_lengths", "block_table"), True),
        stable_host_allocations=True,
        stable_device_allocations=True,
        host_allocation_count=3,
        device_allocation_count=0,
        setup_copies=dict(q_lengths=1, kv_lengths=0, block_table=0),
        step_copies=dict(q_lengths=0, kv_lengths=8, block_table=8),
        prepared_steps=8,
        completed_steps=8,
        inflight=False,
        q4_invariant=dict(query_length=4, cumulative_q=list(range(4, capacity + 1, 4))),
    )


def make_addresses(expected, capacity):
    shapes = dict(
        q_lengths=[capacity // 4],
        kv_lengths=[capacity // 4],
        block_table=[capacity // 4, expected["schedule"]["owner_pages"]["width"]],
        atten_mask=[2048, 2048],
        metadata=[1024],
    )
    shapes.update({f"{name}{index}": [capacity, 12, 256] for name in ("query", "output") for index in range(16)})
    return {
        name: dict(
            pointer=1000 + index * 1_000_000, shape=shape, stride=[math.prod(shape[i + 1 :]) for i in range(len(shape))]
        )
        for index, (name, shape) in enumerate(shapes.items())
    }


def make_report(module, expected, trial, policy, index):
    """Construct a complete protocol fixture so each mutation breaks one gate."""
    identity, ranges = make_inputs(module, expected)
    library, library_hash = module.LIBRARIES[policy]
    record_by_capacity = {record["physical_tokens"]: record for record in expected["selected"]}
    checks, preflight = [], []
    for record in expected["selected"]:
        capacity = record["physical_tokens"]
        plan = make_plan(module, record)
        if policy == "upstream":
            plan.pop("whole_coverage")
        preflight.append(
            dict(capacity=capacity, record=record, verified_before_attention_capture=True, **copy.deepcopy(plan))
        )
        checks.append(
            dict(
                capacity=capacity,
                step=record["step"],
                private_occurrences=16,
                fp32_sampled_rows=[
                    dict(occurrence=occurrence, row=row, max_abs_error=0.0003)
                    for occurrence in range(16)
                    for row in (0, record["batch"] // 2, record["batch"] - 1)
                ],
                output_hashes=[
                    dict(shape=[capacity, 12, 256], dtype="torch.bfloat16", sha256=f"{occurrence:064x}")
                    for occurrence in range(16)
                ],
                finite=True,
                all_outputs_finite=True,
                exact_zero_padding=True,
                stable_addresses=True,
                exact_descriptors_read_only=True,
                resident_queries_unchanged=True,
                plan=copy.deepcopy(plan),
            )
        )
    samples = []
    for repeat in range(5):
        offset = (trial + repeat) % 4
        order = module.CAPACITIES[offset:] + module.CAPACITIES[:offset]
        for capacity in order:
            record = record_by_capacity[capacity]
            base = 100 + trial + repeat + capacity / 100
            samples.append(
                dict(
                    trial=trial,
                    policy=policy,
                    repeat=repeat,
                    step=record["step"],
                    batch=record["batch"],
                    capacity=capacity,
                    wall_us=base if policy == "upstream" else base - 10,
                    preparation_enqueue_us=5 if policy == "upstream" else 4,
                    submit_join_cpu_us=3 if policy == "upstream" else 2,
                    queue_span_us=80 if policy == "upstream" else 72,
                )
            )
    mask = dict(shape=[2048, 2048], dtype="torch.bool", sha256="1" * 64)
    return dict(
        state="completed",
        stage="completed",
        returncode=0,
        npu_work_executed=True,
        performance_measured=True,
        full_trace_measured=False,
        full_model_qualified=False,
        end_to_end_gain_established=False,
        started_unix=10 * index + 2,
        finished_unix=10 * index + 8,
        child_pid=1000 + index,
        occurrences=16,
        repeats=5,
        trial=trial,
        policy=policy,
        device_name="Ascend910B3",
        torch_version="2.10.0+cpu",
        torch_npu_version="2.10.0.post4",
        canonical_digest=module.evidence.CANONICAL_TRACE_SHA256,
        selected_records=expected["selected"],
        trace_contract=expected["trace_contract"],
        core_limit=dict(cube_core_num=20, vector_core_num=40),
        installed_catalog=dict(
            installed_catalog_revalidated=True,
            selection=dict(
                capture_sizes=expected["schedule"]["default_capture_sizes"],
                dispatched_tokens=expected["independent"]["dispatched_tokens"],
                config_source="installed EngineArgs defaults; no capture sizes supplied",
            ),
        ),
        input_identity=identity,
        post_input_identity=copy.deepcopy(identity),
        private_storage_ranges=ranges,
        mask_identity=mask,
        post_mask_identity=copy.deepcopy(mask),
        scripts=expected["scripts"],
        policy_provenance=dict(
            policy=policy,
            library="/vendor/op_impl/cpu/aicpu_kernel/impl/" + library,
            library_sha256=library_hash,
            opapi_sha256=module.OPAPI_SHA256,
        ),
        opp_artifacts={
            "op_api/lib/libcust_opapi.so": module.OPAPI_SHA256,
            "op_impl/cpu/aicpu_kernel/impl/" + library: library_hash,
            "op_impl/cpu/config/cust_aicpu_kernel.json": module.DISPATCH_CONFIGS[policy],
            "op_impl/ai_core/consumer.o": "f" * 64,
        },
        extension_sha256=module.EXTENSION_SHA256,
        final_native_provenance=dict(
            alternate_sink_symbols_absent=True,
            selected_library_handle_retained=True,
            libraries=[dict(path="/vendor/op_api/lib/libcust_opapi.so", sha256=module.OPAPI_SHA256)],
            sink_symbols=[
                dict(symbol=name, source="/vendor/op_api/lib/libcust_opapi.so", address="0x1234")
                for name in module.REQUIRED_SYMBOLS
            ],
        ),
        metadata_preflight=preflight,
        pre_checks=checks,
        post_checks=copy.deepcopy(checks),
        prepared_descriptors={str(capacity): make_descriptors(capacity) for capacity in module.CAPACITIES},
        captured_addresses={str(capacity): make_addresses(expected, capacity) for capacity in module.CAPACITIES},
        samples=samples,
    )


@pytest.fixture
def dataset(module, tmp_path):
    expected = module.expected_contract()
    orders = [list(module.POLICIES if trial % 2 == 0 else reversed(module.POLICIES)) for trial in range(4)]
    bundle = dict(
        state="completed",
        returncode=0,
        started_unix=0,
        finished_unix=100,
        full_trace_measured=False,
        full_model_qualified=False,
        end_to_end_gain_established=False,
        design=dict(
            policies=list(module.POLICIES),
            capacities=list(module.CAPACITIES),
            occurrences=16,
            rounds=4,
            repeats=5,
            process_orders=orders,
        ),
        runs=[],
    )
    reports = []
    for trial, order in enumerate(orders):
        for policy in order:
            index = len(reports)
            report = make_report(module, expected, trial, policy, index)
            reports.append(report)
            bundle["runs"].append(
                dict(
                    trial=trial,
                    policy=policy,
                    process_order_index=index,
                    report_path=f"run{index}.json",
                    report_sha256="",
                    launched_unix=10 * index + 1,
                    finished_unix=10 * index + 9,
                    pid=100 + index,
                    returncode=0,
                )
            )

    def save():
        for entry, report in zip(bundle["runs"], reports, strict=True):
            path = tmp_path / entry["report_path"]
            path.write_text(json.dumps(report), encoding="ascii")
            entry["report_sha256"] = module.evidence.sha256(path)
        return bundle

    save()
    return bundle, reports, save, tmp_path


def analyze(module, dataset):
    bundle, _, _, directory = dataset
    return module.analyze(bundle, base_dir=directory)


def test_complete_block_paired_protocol(module, dataset):
    result = analyze(module, dataset)
    assert result["protocol_qualified"], result["issues"]
    assert result["coverage"]["valid_sample_count"] == 160
    comparison = result["comparison"]
    assert comparison["overall"]["paired_count"] == 80
    assert len(result["samples"]) == 160
    wall = comparison["overall"]["metrics"]["wall_us"]
    assert wall["upstream_minus_whole"]["mean_us"] == 10
    assert wall["whole_faster_count"] == 80
    assert all(row["paired_count"] == 20 for row in comparison["per_capacity"])
    assert all(row["metrics"]["wall_us"]["whole_faster_count"] == 20 for row in comparison["per_capacity"])
    assert not result["absolute_causal_attribution_established"]
    assert not result["end_to_end_gain_established"]


def test_unfiltered_outlier_kept_and_weighted_reduction(module, dataset):
    _, reports, save, _ = dataset
    reports[1]["samples"][0]["wall_us"] = 10000
    save()
    result = analyze(module, dataset)
    assert result["protocol_qualified"]
    wall = result["comparison"]["overall"]["metrics"]["wall_us"]
    assert wall["whole"]["mean_us"] > 200
    assert wall["whole_slower_count"] == 1
    left = [sample["wall_us"] for report in reports if report["policy"] == "upstream" for sample in report["samples"]]
    right = [sample["wall_us"] for report in reports if report["policy"] == "whole" for sample in report["samples"]]
    assert wall["upstream_reduction_percent"] == pytest.approx(100 * (sum(left) - sum(right)) / sum(left))
    assert max(sample["wall_us"] for sample in result["samples"]) == 10000


def test_impossible_same_clock_timing_cannot_create_a_gain(module, dataset):
    _, reports, save, _ = dataset
    reports[1]["samples"][0]["wall_us"] = 0.001
    save()
    result = analyze(module, dataset)
    assert not result["protocol_qualified"]
    assert any("same-clock" in issue for issue in result["issues"])


def test_queue_and_cpu_clocks_are_not_assumed_identical(module, dataset):
    _, reports, save, _ = dataset
    reports[0]["samples"][0]["queue_span_us"] = 10000
    save()
    result = analyze(module, dataset)
    assert result["protocol_qualified"], result["issues"]
    assert len(result["queue_clock_diagnostics"]) == 1
    assert result["queue_clock_diagnostics"][0]["queue_span_us"] == 10000


@pytest.mark.parametrize(
    "field,value",
    [
        ("trial", False),
        ("repeat", []),
        ("step", False),
        ("batch", True),
        ("capacity", []),
        ("policy", []),
        ("wall_us", 0),
        ("queue_span_us", -1),
        ("preparation_enqueue_us", True),
    ],
)
def test_malformed_sample_cannot_qualify(module, dataset, field, value):
    _, reports, save, _ = dataset
    reports[0]["samples"][0][field] = value
    save()
    assert not analyze(module, dataset)["protocol_qualified"]


@pytest.mark.parametrize("mutation", ["duplicate", "missing", "order", "wrong_process"])
def test_sample_matrix_and_rotation_are_exact(module, dataset, mutation):
    _, reports, save, _ = dataset
    samples = reports[0]["samples"]
    if mutation == "duplicate":
        samples.append(copy.deepcopy(samples[0]))
    elif mutation == "missing":
        samples.pop()
    elif mutation == "order":
        samples[0], samples[1] = samples[1], samples[0]
    else:
        samples[0]["trial"] = 1
    save()
    assert not analyze(module, dataset)["protocol_qualified"]


@pytest.mark.parametrize("mutation", ["reorder", "overlap", "pid", "bool_rc", "short_design"])
def test_fresh_process_order_and_intervals(module, dataset, mutation):
    bundle, _, _, _ = dataset
    if mutation == "reorder":
        bundle["runs"][0], bundle["runs"][1] = bundle["runs"][1], bundle["runs"][0]
    elif mutation == "overlap":
        bundle["runs"][1]["launched_unix"] = 5
    elif mutation == "pid":
        bundle["runs"][1]["pid"] = bundle["runs"][0]["pid"]
    elif mutation == "bool_rc":
        bundle["runs"][0]["returncode"] = False
    else:
        bundle["design"]["rounds"] = 1
    assert not analyze(module, dataset)["protocol_qualified"]


@pytest.mark.parametrize("mutation", ["missing_post", "dup_cap", "missing_occurrence", "nan_error", "finite_false"])
def test_numeric_gate_all_capacities_all_occurrences(module, dataset, mutation):
    _, reports, save, _ = dataset
    report = reports[0]
    if mutation == "missing_post":
        report.pop("post_checks")
    elif mutation == "dup_cap":
        report["pre_checks"][1] = copy.deepcopy(report["pre_checks"][0])
    elif mutation == "missing_occurrence":
        report["pre_checks"][0]["fp32_sampled_rows"].pop()
    elif mutation == "nan_error":
        report["pre_checks"][0]["fp32_sampled_rows"][0]["max_abs_error"] = -1
    elif mutation == "finite_false":
        report["pre_checks"][0]["all_outputs_finite"] = False
    save()
    assert not analyze(module, dataset)["protocol_qualified"]


def test_output_bitwise_variation_is_diagnostic(module, dataset):
    _, reports, save, _ = dataset
    reports[0]["post_checks"][0]["output_hashes"][15]["sha256"] = "e" * 64
    save()
    result = analyze(module, dataset)
    assert result["protocol_qualified"], result["issues"]
    assert any(item["identical"] is False for item in result["output_bitwise_repeatability"])


@pytest.mark.parametrize("mutation", ["mapped_hash", "symbol_missing", "symbol_owner", "dispatch"])
def test_loaded_provenance_is_checked_beyond_boolean_flags(module, dataset, mutation):
    _, reports, save, _ = dataset
    report = reports[0]
    if mutation == "mapped_hash":
        report["final_native_provenance"]["libraries"][0]["sha256"] = "0" * 64
    elif mutation == "symbol_missing":
        report["final_native_provenance"]["sink_symbols"].pop()
    elif mutation == "symbol_owner":
        report["final_native_provenance"]["sink_symbols"][0]["source"] = "/alternate/libcust_opapi.so"
    else:
        report["opp_artifacts"]["op_impl/cpu/config/cust_aicpu_kernel.json"] = "0" * 64
    save()
    assert not analyze(module, dataset)["protocol_qualified"]


def test_cross_process_mask_identity_must_match(module, dataset):
    _, reports, save, _ = dataset
    reports[1]["mask_identity"]["sha256"] = "0" * 64
    reports[1]["post_mask_identity"]["sha256"] = "0" * 64
    save()
    assert not analyze(module, dataset)["protocol_qualified"]


@pytest.mark.parametrize(
    "mutation", ["input_changed", "post_input_changed", "mask_changed", "overlap", "alias", "copy_missing"]
)
def test_identity_and_lifetime_controls(module, dataset, mutation):
    _, reports, save, _ = dataset
    report = reports[0]
    if mutation == "input_changed":
        report["input_identity"]["occurrences"][0]["key"]["sha256"] = "d" * 64
    elif mutation == "post_input_changed":
        report["post_input_identity"]["occurrences"][15]["value"]["sha256"] = "d" * 64
    elif mutation == "mask_changed":
        report["post_mask_identity"]["sha256"] = "d" * 64
    elif mutation == "overlap":
        report["private_storage_ranges"][1]["start"] = report["private_storage_ranges"][0]["start"]
    elif mutation == "alias":
        report["captured_addresses"]["256"]["output1"]["pointer"] = report["captured_addresses"]["256"]["output0"][
            "pointer"
        ]
    else:
        report["prepared_descriptors"]["256"]["step_copies"]["kv_lengths"] = 7
    save()
    assert not analyze(module, dataset)["protocol_qualified"]


@pytest.mark.parametrize("mutation", ["library", "consumer", "scripts", "lengths", "fd", "tail"])
def test_native_provenance_and_payload_checked_independently(module, dataset, mutation):
    _, reports, save, _ = dataset
    report = reports[1]
    if mutation == "library":
        report["policy_provenance"]["library_sha256"] = "0" * 64
    elif mutation == "consumer":
        report["opp_artifacts"]["op_impl/ai_core/consumer.o"] = "0" * 64
    elif mutation == "scripts":
        report["scripts"]["aclgraph_fia_sink_stage6_probe.py"] = "0" * 64
    else:
        plan = report["metadata_preflight"][0]
        word = {"lengths": 583, "fd": 8, "tail": 1023}[mutation]
        plan["words"][word] += 1
        plan["sha256"] = hashlib.sha256(struct.pack("<1024i", *plan["words"])).hexdigest()
    save()
    assert not analyze(module, dataset)["protocol_qualified"]


@pytest.mark.parametrize("value", ["../outside.json", "/absolute.json", "C:/absolute.json", "run0\\file.json"])
def test_raw_report_path_cannot_escape(module, dataset, value):
    bundle, _, _, _ = dataset
    bundle["runs"][0]["report_path"] = value
    result = analyze(module, dataset)
    assert not result["protocol_qualified"]
    assert any("report_path" in issue for issue in result["issues"])


def test_report_hash_and_duplicate_path_rejected(module, dataset):
    bundle, reports, _, directory = dataset
    (directory / "run0.json").write_text(json.dumps(reports[0]) + " ")
    assert not analyze(module, dataset)["protocol_qualified"]
    bundle["runs"][0]["report_sha256"] = module.evidence.sha256(directory / "run0.json")
    bundle["runs"][1]["report_path"] = "run0.json"
    bundle["runs"][1]["report_sha256"] = bundle["runs"][0]["report_sha256"]
    result = analyze(module, dataset)
    assert any("duplicate report_path" in issue for issue in result["issues"])


def test_symlink_escape_rejected(module, tmp_path):
    external = tmp_path / "external.json"
    external.write_text("{}")
    inside = tmp_path / "bundle"
    inside.mkdir()
    try:
        (inside / "link.json").symlink_to(external)
    except OSError:
        pytest.skip("symlink creation is unavailable on this host")
    with pytest.raises(ValueError, match="escapes"):
        module.resolve_report(dict(report_path="link.json"), inside.resolve())


def test_cli_preserves_raw_evidence(module, dataset, monkeypatch):
    bundle, _, _, directory = dataset
    source = directory / "bundle.json"
    source.write_text(json.dumps(bundle))
    target = directory / "run0.json"
    before = target.read_bytes()
    monkeypatch.setattr(sys, "argv", ["analyzer", "--input", str(source), "--output", str(target)])
    with pytest.raises(SystemExit):
        module.main()
    assert target.read_bytes() == before


def test_cli_writes_qualified_analysis(module, dataset, monkeypatch):
    bundle, _, _, directory = dataset
    source, target = directory / "bundle.json", directory / "analysis.json"
    source.write_text(json.dumps(bundle))
    monkeypatch.setattr(sys, "argv", ["analyzer", "--input", str(source), "--output", str(target)])
    assert module.main() == 0
    result = json.loads(target.read_text())
    assert result["protocol_qualified"]
    assert result["input"]["sha256"] == module.evidence.sha256(source)
