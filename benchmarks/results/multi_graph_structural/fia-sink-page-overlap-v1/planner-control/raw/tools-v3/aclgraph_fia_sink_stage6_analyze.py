# SPDX-License-Identifier: Apache-2.0
"""Audit fresh-process, block-paired native producer measurements on CPU.

All observations are retained. The AB/BA process blocks control order drift;
they are not randomized within-process sample pairs or a causal attribution.
"""

import argparse
import contextlib
import hashlib
import json
import math
import struct
from pathlib import Path, PurePosixPath, PureWindowsPath

import aclgraph_fia_sink_stage4_whole_analyze as evidence
from aclgraph_fia_sink_stage3_trace import load_trace_schedule
from aclgraph_fia_sink_stage4_whole_plan import validate_whole_plan

POLICIES = ("upstream", "whole")
CAPACITIES = (256, 400, 496, 512)
METRICS = evidence.METRICS
ROUNDS = 4
REPEATS = 5
OCCURRENCES = 16
OPAPI_SHA256 = "2c1660f393e0e8cc9c8d935929420e3e3b61297163994ade48eec5ac14c9ad40"
EXTENSION_SHA256 = "25a41e47473a3bc9c34a50c053766181c7a764ea496b93d4d3b70180b0ecf76b"
LIBRARIES = {
    "upstream": ("libfia_v2_sink_aicpu_kernels.so", "796983827bc6326a250009a2aacc810c44f65545ec2ae56dab517a5ec4e7111b"),
    "whole": ("libthesis_stage5_whole_aicpu.so", "1c4ab9fcf2b7d2935be3e60f2a215fb65dc85859bc6fe99fe5271b4c4784ea90"),
}
DISPATCH_CONFIGS = {
    "upstream": "1127c8287069e1ab1f2a681e76a3dd9c1385eb86e2a8584dd02400333ed09203",
    "whole": "7d5c6a82a6ae43ffa7ba3fbb2d13f8ebad8b85728fbbc9ddac4a35e3df724d1b",
}
RUNTIME_SCRIPTS = (
    "aclgraph_fia_sink_stage6_probe.py",
    "aclgraph_fia_sink_stage5_gate.py",
    "aclgraph_fia_sink_stage4_whole_probe.py",
    "aclgraph_fia_sink_stage4_whole_plan.py",
    "aclgraph_fia_sink_stage4_prepare.py",
    "aclgraph_fia_sink_stage3_budget.py",
    "aclgraph_fia_sink_stage3_trace.py",
    "aclgraph_fia_sink_stage3_legacy.py",
    "aclgraph_fia_sink_stage2_graph.py",
    "aclgraph_fia_sink_stage2_validate.py",
    "aclgraph_fia_sink_stage1_smoke.py",
)
SCOPE_FLAGS = ("full_trace_measured", "full_model_qualified", "end_to_end_gain_established")
REQUIRED_SYMBOLS = (
    "aclnnFusedInferAttentionScoreV2SinkV3GetWorkspaceSize",
    "aclnnFusedInferAttentionScoreV2SinkV3",
    "aclnnFusedInferAttentionScoreV2SinkMetadataGetWorkspaceSize",
    "aclnnFusedInferAttentionScoreV2SinkMetadata",
)
REGISTRATION_LIBRARY = "/usr/local/Ascend/cann-9.1.0/opp/vendors/batch_invariant/op_api/lib/libcust_opapi.so"
REGISTRATION_SHA256 = "d1eecbbe84ac8b32af671b09a988f43d953198f671e34c379bb38c2b356670b8"


def integer(value, minimum=0):
    return type(value) is int and value >= minimum


def exact(left, right):
    """Compare JSON identities without accepting bool as an integer."""
    return json.dumps(left, sort_keys=True, separators=(",", ":")) == json.dumps(
        right, sort_keys=True, separators=(",", ":")
    )


def sha(value):
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def normalized(value):
    return json.loads(json.dumps(value))


def array(value):
    return value if isinstance(value, list) else []


def expected_contract(trace_path=None, selection_path=None):
    """Reconstruct exact records from independently pinned canonical sources."""
    independent = evidence.expected_schedule(trace_path, selection_path)
    schedule = normalized(load_trace_schedule(trace_path, selection_path))
    selected = []
    for capacity in CAPACITIES:
        record = next(item for item in schedule["records"] if item["physical_tokens"] == capacity)
        selected.append(dict(record, page_count=schedule["owner_pages"]["page_count"]))
    contract = {key: value for key, value in schedule.items() if key != "records"}
    contract["installed_catalog_revalidated"] = True
    scripts = {name: evidence.sha256(Path(__file__).with_name(name)) for name in RUNTIME_SCRIPTS}
    return dict(schedule=schedule, trace_contract=contract, selected=selected, independent=independent, scripts=scripts)


def resolve_report(entry, base_dir):
    """Keep relative evidence reads within the bundle directory, including links."""
    value = entry.get("report_path")
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("report_path must be a nonempty relative POSIX path")
    path = Path(value)
    if PurePosixPath(value).is_absolute() or PureWindowsPath(value).drive or ".." in path.parts:
        raise ValueError("report_path must not be absolute or contain traversal")
    target = (base_dir / path).resolve(strict=True)
    if not target.is_relative_to(base_dir) or not target.is_file():
        raise ValueError("report_path escapes the bundle directory or is not a file")
    return target


def load_runs(bundle, base_dir, issues):
    entries = bundle.get("runs")
    if not isinstance(entries, list):
        issues.append("runs must be a list")
        return [], []
    runs, paths = [], []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            issues.append(f"run {index}: entry must be an object")
            continue
        try:
            path = resolve_report(entry, base_dir)
            if path in paths:
                raise ValueError("duplicate report_path")
            paths.append(path)
            if not sha(entry.get("report_sha256")) or evidence.sha256(path) != entry["report_sha256"]:
                raise ValueError("raw report SHA256 mismatch")
            report = evidence.load_json(path)
            if not isinstance(report, dict):
                raise ValueError("raw report must be an object")
            runs.append((entry, report))
        except (ValueError, OSError, TypeError) as error:
            issues.append(f"run {index}: {error}")
    return runs, paths


def design_issues(bundle):
    expected = dict(
        policies=list(POLICIES),
        capacities=list(CAPACITIES),
        occurrences=OCCURRENCES,
        rounds=ROUNDS,
        repeats=REPEATS,
        process_orders=[list(POLICIES if trial % 2 == 0 else reversed(POLICIES)) for trial in range(ROUNDS)],
    )
    design = bundle.get("design")
    issues = []
    if not isinstance(design, dict) or any(not exact(design.get(key), value) for key, value in expected.items()):
        issues.append("design must declare the exact four-round/five-repeat AB/BA block protocol")
    if bundle.get("state") != "completed" or type(bundle.get("returncode")) is not int or bundle["returncode"] != 0:
        issues.append("bundle execution did not complete successfully")
    if any(bundle.get(flag) is not False for flag in SCOPE_FLAGS):
        issues.append("bundle must explicitly retain bounded scope flags")
    return issues


def process_issues(bundle, runs):
    issues, previous_finish, pids, children = [], None, set(), set()
    started, finished = bundle.get("started_unix"), bundle.get("finished_unix")
    if not evidence.finite(started) or not evidence.finite(finished) or finished <= started:
        issues.append("bundle requires a valid execution interval")
        started, finished = -math.inf, math.inf
    expected = [
        (trial, policy) for trial in range(ROUNDS) for policy in (POLICIES if trial % 2 == 0 else reversed(POLICIES))
    ]
    if len(runs) != len(expected):
        issues.append("exactly eight verified raw process reports are required")
    for index, (entry, report) in enumerate(runs):
        if (
            index >= len(expected)
            or not integer(entry.get("trial"))
            or (entry.get("trial"), entry.get("policy")) != expected[index]
            or type(entry.get("process_order_index")) is not int
            or entry["process_order_index"] != index
        ):
            issues.append(f"run {index}: process order does not match the declared AB/BA blocks")
        begin, end = entry.get("launched_unix"), entry.get("finished_unix")
        if not evidence.finite(begin) or not evidence.finite(end) or not started <= begin < end <= finished:
            issues.append(f"run {index}: invalid process interval")
        else:
            if previous_finish is not None and begin < previous_finish:
                issues.append(f"run {index}: process intervals overlap")
            previous_finish = end
            raw_begin, raw_end = report.get("started_unix"), report.get("finished_unix")
            if (
                not evidence.finite(raw_begin)
                or not evidence.finite(raw_end)
                or not begin <= raw_begin < raw_end <= end
            ):
                issues.append(f"run {index}: raw report interval is outside the launched process")
        pid = entry.get("pid")
        child = report.get("child_pid")
        if not integer(pid, 1) or pid in pids or not integer(child, 1) or child in children:
            issues.append(f"run {index}: distinct launcher and native child PID identities are required")
        if integer(pid, 1):
            pids.add(pid)
        if integer(child, 1):
            children.add(child)
        if type(entry.get("returncode")) is not int or entry["returncode"] != 0:
            issues.append(f"run {index}: process did not exit successfully")
    if pids & children:
        issues.append("launcher and native child process identities must be disjoint")
    return issues


def input_issues(report, expected):
    identity = report.get("input_identity")
    if not isinstance(identity, dict):
        return ["input_identity is required"]
    entries = identity.get("occurrences")
    if not isinstance(entries, list) or len(entries) != OCCURRENCES:
        return ["input identity must cover all sixteen private occurrences"]
    issues = []
    shapes = {
        "query": [max(expected["schedule"]["required_capture_sizes"]), 12, 256],
        "key": [expected["schedule"]["owner_pages"]["page_count"], 128, 2, 256],
        "value": [expected["schedule"]["owner_pages"]["page_count"], 128, 2, 256],
    }
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != set(shapes):
            issues.append("input occurrence must describe query, key and value")
            continue
        for name, shape in shapes.items():
            tensor = entry.get(name)
            if (
                not isinstance(tensor, dict)
                or not exact(tensor.get("shape"), shape)
                or tensor.get("dtype") != "torch.bfloat16"
                or not sha(tensor.get("sha256"))
            ):
                issues.append(f"input {name} shape/dtype/hash is invalid")
    if identity.get("sha256") != evidence.digest(entries):
        issues.append("input identity aggregate SHA256 does not match its per-tensor evidence")
    if not exact(report.get("post_input_identity"), identity):
        issues.append("complete resident Q/K/V byte identity changed after timing")
    mask = report.get("mask_identity")
    if (
        not isinstance(mask, dict)
        or not exact(mask.get("shape"), [2048, 2048])
        or mask.get("dtype") != "torch.bool"
        or not sha(mask.get("sha256"))
        or not exact(report.get("post_mask_identity"), mask)
    ):
        issues.append("complete causal mask identity is absent or changed")
    ranges = report.get("private_storage_ranges")
    wanted = {(occurrence, name) for occurrence in range(OCCURRENCES) for name in shapes}
    if not isinstance(ranges, list) or len(ranges) != len(wanted):
        issues.append("private storage ranges must cover every Q/K/V occurrence")
    else:
        seen, ordered = set(), []
        for item in ranges:
            if (
                not isinstance(item, dict)
                or not integer(item.get("occurrence"))
                or not isinstance(item.get("name"), str)
                or (item["occurrence"], item["name"]) not in wanted
                or not integer(item.get("start"), 1)
                or not integer(item.get("end"), 1)
            ):
                issues.append("private storage range identity is invalid")
                continue
            key = item["occurrence"], item["name"]
            if key in seen or item["end"] - item["start"] != 2 * math.prod(shapes[item["name"]]):
                issues.append("private storage range is duplicate or has the wrong extent")
            seen.add(key)
            ordered.append((item["start"], item["end"]))
        if seen != wanted or any(left[1] > right[0] for left, right in zip(sorted(ordered), sorted(ordered)[1:])):
            issues.append("private Q/K/V storage is missing or overlapping")
    return issues


def plan_issues(plan, record, policy):
    if not isinstance(plan, dict):
        return ["native plan evidence is required"]
    words = plan.get("words")
    if (
        not isinstance(words, list)
        or len(words) != 1024
        or any(type(x) is not int or not 0 <= x < 2**31 for x in words)
    ):
        return ["native plan must retain all 1024 nonnegative int32 words"]
    issues = []
    if hashlib.sha256(struct.pack("<1024i", *words)).hexdigest() != plan.get("sha256"):
        issues.append("native plan payload SHA256 mismatch")
    base = words[576:586]
    maximum = max(record["kv_lengths"]) or 1024
    if base[:3] != [512, min(512, maximum), 8] or base[6:] != [
        4,
        maximum,
        record["physical_batch"],
        record["physical_batch"],
    ]:
        issues.append("native plan geometry or current lengths differ from the canonical record")
    if not 1 <= base[4] <= 20:
        issues.append("native plan active core count is outside the qualified allocation")
    if policy == "whole":
        try:
            checked = validate_whole_plan(record, words, cores=20)
            if not exact(checked, plan.get("whole_coverage")):
                issues.append("independent no-FD coverage differs from the recorded validation")
        except (ValueError, KeyError, TypeError) as error:
            issues.append(f"whole-row payload rejected: {error}")
    return issues


def numeric_issues(report, expected, policy):
    issues, plans = [], {}
    records = {record["physical_tokens"]: record for record in expected["selected"]}
    for label in ("metadata_preflight", "pre_checks", "post_checks"):
        checks = report.get(label)
        if not isinstance(checks, list) or len(checks) != len(CAPACITIES):
            issues.append(f"{label} must cover all four capacities")
            continue
        seen = set()
        for check in checks:
            if not isinstance(check, dict) or type(check.get("capacity")) is not int:
                issues.append(f"{label}: invalid capacity identity")
                continue
            capacity = check["capacity"]
            if capacity not in records or capacity in seen:
                issues.append(f"{label}: duplicate or unexpected capacity")
                continue
            seen.add(capacity)
            record = records[capacity]
            if label == "metadata_preflight":
                if not exact(check.get("record"), record) or check.get("verified_before_attention_capture") is not True:
                    issues.append(f"{label}: exact current record and pre-capture validation required")
                plan = check
            else:
                flags = (
                    "finite",
                    "all_outputs_finite",
                    "exact_zero_padding",
                    "stable_addresses",
                    "exact_descriptors_read_only",
                    "resident_queries_unchanged",
                )
                if (
                    type(check.get("step")) is not int
                    or check["step"] != record["step"]
                    or type(check.get("private_occurrences")) is not int
                    or check["private_occurrences"] != OCCURRENCES
                    or any(check.get(flag) is not True for flag in flags)
                ):
                    issues.append(f"{label}: finite/read-only/address/occurrence checks failed")
                rows = check.get("fp32_sampled_rows")
                wanted = {
                    (occurrence, row)
                    for occurrence in range(OCCURRENCES)
                    for row in {0, record["batch"] // 2, record["batch"] - 1}
                }
                if (
                    not isinstance(rows, list)
                    or len(rows) != len(wanted)
                    or any(
                        not isinstance(row, dict)
                        or type(row.get("row")) is not int
                        or type(row.get("occurrence")) is not int
                        for row in rows
                    )
                    or {(row["occurrence"], row["row"]) for row in rows} != wanted
                    or any(not evidence.finite(row.get("max_abs_error")) for row in rows)
                ):
                    issues.append(f"{label}: three distinct FP32 rows for every private occurrence are required")
                outputs = check.get("output_hashes")
                if (
                    not isinstance(outputs, list)
                    or len(outputs) != OCCURRENCES
                    or any(
                        not isinstance(output, dict)
                        or not exact(output.get("shape"), [capacity, 12, 256])
                        or output.get("dtype") != "torch.bfloat16"
                        or not sha(output.get("sha256"))
                        for output in outputs
                    )
                ):
                    issues.append(f"{label}: all private output identities are required")
                plan = check.get("plan")
            issues.extend(f"{label}/{capacity}: {issue}" for issue in plan_issues(plan, record, policy))
            if isinstance(plan, dict):
                digest = plan.get("sha256")
                if capacity in plans and digest != plans[capacity]:
                    issues.append(f"{label}/{capacity}: fixed-record metadata changed across qualification")
                plans[capacity] = digest
    return issues


def preparation_issues(report):
    summaries = report.get("prepared_descriptors")
    if not isinstance(summaries, dict) or set(summaries) != {str(capacity) for capacity in CAPACITIES}:
        return ["prepared_descriptors must cover all four capacities"]
    issues = []
    for capacity in CAPACITIES:
        item = summaries[str(capacity)]
        if not isinstance(item, dict):
            issues.append("prepared descriptor summary must be an object")
            continue
        wanted = dict(
            mode="sink",
            physical_tokens=capacity,
            physical_requests=capacity // 4,
            host_pinned=dict.fromkeys(("q_lengths", "kv_lengths", "block_table"), True),
            stable_host_allocations=True,
            stable_device_allocations=True,
            host_allocation_count=3,
            device_allocation_count=0,
            setup_copies=dict(q_lengths=1, kv_lengths=0, block_table=0),
            step_copies=dict(q_lengths=0, kv_lengths=REPEATS + 3, block_table=REPEATS + 3),
            prepared_steps=REPEATS + 3,
            completed_steps=REPEATS + 3,
            inflight=False,
            q4_invariant=dict(query_length=4, cumulative_q=list(range(4, capacity + 1, 4))),
        )
        if any(not exact(item.get(key), value) for key, value in wanted.items()):
            issues.append(f"capacity {capacity}: pinned preparation/copy/lifetime contract failed")
    return issues


def allocation_issues(report, expected):
    addresses = report.get("captured_addresses")
    if not isinstance(addresses, dict) or set(addresses) != {str(capacity) for capacity in CAPACITIES}:
        return ["capture-origin addresses must cover every default bucket"]
    issues = []
    for capacity in CAPACITIES:
        shapes = dict(
            q_lengths=[capacity // 4],
            kv_lengths=[capacity // 4],
            block_table=[capacity // 4, expected["schedule"]["owner_pages"]["width"]],
            atten_mask=[2048, 2048],
            metadata=[1024],
        )
        shapes.update(
            {f"{name}{index}": [capacity, 12, 256] for name in ("query", "output") for index in range(OCCURRENCES)}
        )
        captured = addresses[str(capacity)]
        if not isinstance(captured, dict) or set(captured) != set(shapes):
            issues.append(f"capacity {capacity}: captured descriptor coverage is incomplete")
            continue
        pointers = []
        for name, shape in shapes.items():
            item = captured[name]
            stride = [math.prod(shape[index + 1 :]) for index in range(len(shape))]
            if (
                not isinstance(item, dict)
                or not integer(item.get("pointer"), 1)
                or not exact(item.get("shape"), shape)
                or not exact(item.get("stride"), stride)
            ):
                issues.append(f"capacity {capacity}: invalid exact captured {name} descriptor")
                continue
            pointers.append(item["pointer"])
        if len(set(pointers)) != len(pointers):
            issues.append(f"capacity {capacity}: captured query/output/descriptor pointers alias")
    return issues


def provenance_issues(report, policy, expected):
    issues = []
    provenance = report.get("policy_provenance")
    library, digest = LIBRARIES[policy]
    if (
        not isinstance(provenance, dict)
        or provenance.get("policy") != policy
        or provenance.get("library_sha256") != digest
        or provenance.get("opapi_sha256") != OPAPI_SHA256
        or Path(str(provenance.get("library"))).name != library
    ):
        issues.append("selected native policy provenance differs from the qualified library")
    artifacts = report.get("opp_artifacts")
    required = {
        "op_api/lib/libcust_opapi.so": OPAPI_SHA256,
        "op_impl/cpu/aicpu_kernel/impl/" + library: digest,
        "op_impl/cpu/config/cust_aicpu_kernel.json": DISPATCH_CONFIGS[policy],
    }
    if not isinstance(artifacts, dict) or any(artifacts.get(key) != value for key, value in required.items()):
        issues.append("raw OPP inventory does not contain the selected policy and common consumer")
    scripts = report.get("scripts")
    if not isinstance(scripts, dict) or any(scripts.get(key) != value for key, value in expected["scripts"].items()):
        issues.append("measured runtime scripts differ from the locally audited sources")
    final = report.get("final_native_provenance")
    if (
        not isinstance(final, dict)
        or final.get("alternate_sink_symbols_absent") is not True
        or final.get("selected_library_handle_retained") is not True
    ):
        issues.append("final loaded native provenance is not qualified")
    else:
        libraries, symbols = final.get("libraries"), final.get("sink_symbols")
        library_path = (
            PurePosixPath(str(provenance.get("library", ""))) if isinstance(provenance, dict) else PurePosixPath()
        )
        selected = str(library_path.parents[4] / "op_api/lib/libcust_opapi.so") if len(library_path.parents) > 4 else ""
        allowed = {selected: OPAPI_SHA256, REGISTRATION_LIBRARY: REGISTRATION_SHA256}
        if (
            not isinstance(libraries, list)
            or not libraries
            or len(libraries) > 2
            or any(not isinstance(item, dict) or not isinstance(item.get("path"), str) for item in libraries)
            or len({item.get("path") for item in libraries}) != len(libraries)
            or not any(item.get("path") == selected and item.get("sha256") == OPAPI_SHA256 for item in libraries)
            or any(allowed.get(item.get("path")) != item.get("sha256") for item in libraries)
        ):
            issues.append("final mapped libraries differ from the selected OpAPI and allowed registration owner")
        if (
            not isinstance(symbols, list)
            or len(symbols) != len(REQUIRED_SYMBOLS)
            or any(not isinstance(item, dict) or not isinstance(item.get("symbol"), str) for item in symbols)
            or {item.get("symbol") for item in symbols} != set(REQUIRED_SYMBOLS)
            or any(item.get("source") != selected or not isinstance(item.get("address"), str) for item in symbols)
        ):
            issues.append("final sink symbols do not all resolve from the selected OpAPI")
    if report.get("extension_sha256") != EXTENSION_SHA256:
        issues.append("extension differs from the common qualified consumer")
    return issues


def report_issues(report, entry, expected):
    policy, trial = entry.get("policy"), entry.get("trial")
    if policy not in POLICIES or not integer(trial) or trial >= ROUNDS:
        return ["invalid report policy/trial identity"]
    issues = []
    wanted = dict(
        state="completed",
        stage="completed",
        returncode=0,
        npu_work_executed=True,
        performance_measured=True,
        occurrences=OCCURRENCES,
        repeats=REPEATS,
        trial=trial,
        policy=policy,
        canonical_digest=evidence.CANONICAL_TRACE_SHA256,
        selected_records=expected["selected"],
        trace_contract=expected["trace_contract"],
        core_limit=dict(cube_core_num=20, vector_core_num=40),
    )
    wanted.update(dict.fromkeys(SCOPE_FLAGS, False))
    for key, value in wanted.items():
        if not exact(report.get(key), value):
            issues.append(f"report {key} differs from its strict execution contract")
    installed = report.get("installed_catalog")
    if not isinstance(installed, dict) or installed.get("installed_catalog_revalidated") is not True:
        issues.append("installed default capture catalog was not revalidated")
    else:
        selection = installed.get("selection", {})
        if not isinstance(selection, dict) or (
            not exact(selection.get("capture_sizes"), expected["schedule"]["default_capture_sizes"])
            or not exact(selection.get("dispatched_tokens"), expected["independent"]["dispatched_tokens"])
            or selection.get("config_source") != "installed EngineArgs defaults; no capture sizes supplied"
        ):
            issues.append("installed default dispatch differs from the archived canonical catalog")
    issues.extend(input_issues(report, expected))
    issues.extend(provenance_issues(report, policy, expected))
    issues.extend(numeric_issues(report, expected, policy))
    issues.extend(preparation_issues(report))
    issues.extend(allocation_issues(report, expected))
    return issues


def collect_samples(runs, expected):
    indexed, issues = {}, []
    records = {item["physical_tokens"]: item for item in expected["selected"]}
    raw_count = 0
    for entry, report in runs:
        samples = report.get("samples")
        if not isinstance(samples, list):
            issues.append("samples must be a list")
            continue
        raw_count += len(samples)
        sequence = []
        for sample in samples:
            if (
                not isinstance(sample, dict)
                or any(type(sample.get(name)) is not int for name in ("trial", "repeat", "step", "batch", "capacity"))
                or not isinstance(sample.get("policy"), str)
            ):
                issues.append("sample identities require exact integers and a policy string")
                continue
            trial, repeat, policy, capacity = (sample[name] for name in ("trial", "repeat", "policy", "capacity"))
            key = trial, policy, capacity, repeat
            if (
                trial != entry.get("trial")
                or policy != entry.get("policy")
                or not 0 <= trial < ROUNDS
                or not 0 <= repeat < REPEATS
                or capacity not in records
                or (sample["step"], sample["batch"]) != (records[capacity]["step"], records[capacity]["batch"])
            ):
                issues.append("sample identity differs from its process or canonical record")
                continue
            if key in indexed:
                issues.append("duplicate sample identity")
                continue
            if any(not evidence.finite(sample.get(metric)) for metric in METRICS) or sample["wall_us"] <= 0:
                issues.append("all sample timings must be finite/nonnegative and wall must be positive")
                continue
            if sample["preparation_enqueue_us"] + sample["submit_join_cpu_us"] > sample["wall_us"] + 1e-6:
                issues.append("same-clock preparation and submission intervals exceed wall time")
                continue
            indexed[key] = sample
            sequence.append((repeat, capacity))
        trial = entry.get("trial")
        wanted = []
        if integer(trial):
            for repeat in range(REPEATS):
                offset = (trial + repeat) % len(CAPACITIES)
                wanted.extend((repeat, capacity) for capacity in CAPACITIES[offset:] + CAPACITIES[:offset])
        if sequence != wanted:
            issues.append("recorded sample order differs from the declared rotating bucket order")
    wanted = {
        (trial, policy, capacity, repeat)
        for trial in range(ROUNDS)
        for policy in POLICIES
        for capacity in CAPACITIES
        for repeat in range(REPEATS)
    }
    missing = sorted(wanted - indexed.keys())
    if missing:
        issues.append("sample matrix is incomplete")
    return indexed, dict(
        complete=not issues,
        expected_sample_count=len(wanted),
        raw_sample_count=raw_count,
        valid_sample_count=len(indexed),
        missing=[list(key) for key in missing],
        issues=issues,
    )


def common_control_issues(runs):
    """Only the dispatched AICPU policy may differ between the two vendors."""
    if not runs:
        return ["no verified raw reports"]
    issues, first = [], runs[0][1]
    keys = (
        "input_identity",
        "mask_identity",
        "scripts",
        "extension_sha256",
        "device_name",
        "torch_version",
        "torch_npu_version",
    )
    for _, report in runs:
        for key in keys:
            if key not in first or not exact(report.get(key), first[key]):
                issues.append(f"cross-process {key} identity differs or is absent")
        inventory = report.get("opp_artifacts")
        baseline = first.get("opp_artifacts")
        if not isinstance(inventory, dict) or not isinstance(baseline, dict):
            continue
        excluded = {"op_impl/cpu/config/cust_aicpu_kernel.json"}
        excluded.update("op_impl/cpu/aicpu_kernel/impl/" + item[0] for item in LIBRARIES.values())
        current = {key: value for key, value in inventory.items() if key not in excluded}
        common = {key: value for key, value in baseline.items() if key not in excluded}
        if not exact(current, common):
            issues.append("non-policy OPP artifacts differ across native processes")
    fixed_plans = {}
    for _, report in runs:
        checks = report.get("metadata_preflight")
        for check in checks if isinstance(checks, list) else []:
            if not isinstance(check, dict) or type(check.get("capacity")) is not int:
                continue
            words = check.get("words")
            if not isinstance(words, list) or len(words) != 1024:
                continue
            geometry = words[576:579] + words[582:586]
            capacity = check["capacity"]
            if capacity in fixed_plans and not exact(geometry, fixed_plans[capacity]):
                issues.append("inner tiling/current-length geometry differs between native policies")
            fixed_plans[capacity] = geometry
    return issues


def group_statistics(indexed, pairs):
    result = dict(paired_count=len(pairs), metrics={})
    for metric in METRICS:
        upstream = [indexed[row["trial"], "upstream", row["capacity"], row["repeat"]][metric] for row in pairs]
        whole = [indexed[row["trial"], "whole", row["capacity"], row["repeat"]][metric] for row in pairs]
        deltas = [left - right for left, right in zip(upstream, whole, strict=True)]
        result["metrics"][metric] = dict(
            upstream=evidence.stats(upstream),
            whole=evidence.stats(whole),
            upstream_minus_whole=evidence.stats(deltas),
            upstream_reduction_percent=100 * sum(deltas) / sum(upstream) if sum(upstream) else None,
            whole_faster_count=sum(value > 0 for value in deltas),
            whole_slower_count=sum(value < 0 for value in deltas),
            equal_count=sum(value == 0 for value in deltas),
        )
    return result


def paired_statistics(indexed):
    pairs = []
    for trial in range(ROUNDS):
        for capacity in CAPACITIES:
            for repeat in range(REPEATS):
                left = indexed.get((trial, "upstream", capacity, repeat))
                right = indexed.get((trial, "whole", capacity, repeat))
                if left is None or right is None:
                    continue
                pairs.append(
                    dict(
                        trial=trial,
                        capacity=capacity,
                        repeat=repeat,
                        step=left["step"],
                        upstream_minus_whole_us={metric: left[metric] - right[metric] for metric in METRICS},
                        upstream_reduction_percent={
                            metric: 100 * (left[metric] - right[metric]) / left[metric] if left[metric] else None
                            for metric in METRICS
                        },
                    )
                )
    return dict(
        overall=group_statistics(indexed, pairs),
        per_capacity=[
            dict(capacity=capacity, **group_statistics(indexed, [row for row in pairs if row["capacity"] == capacity]))
            for capacity in CAPACITIES
        ],
        per_round=[
            dict(trial=trial, **group_statistics(indexed, [row for row in pairs if row["trial"] == trial]))
            for trial in range(ROUNDS)
        ],
        pairs=pairs,
    )


def analyze(bundle, *, base_dir, trace_path=None, selection_path=None):
    """Validate exact evidence before qualifying the bounded statistical result."""
    if not isinstance(bundle, dict):
        raise ValueError("stage-six bundle must be an object")
    expected = expected_contract(trace_path, selection_path)
    issues = design_issues(bundle)
    runs, paths = load_runs(bundle, Path(base_dir).resolve(strict=True), issues)
    issues.extend(process_issues(bundle, runs))
    issues.extend(common_control_issues(runs))
    for index, (entry, report) in enumerate(runs):
        issues.extend(f"run {index}: {issue}" for issue in report_issues(report, entry, expected))
    indexed, coverage = collect_samples(runs, expected)
    issues.extend(coverage["issues"])
    qualified = not issues
    comparison = paired_statistics(indexed)
    return dict(
        schema_version=1,
        state="complete_bounded_comparison" if qualified else "partial_or_invalid",
        analysis_valid=qualified,
        protocol_qualified=qualified,
        issues=issues,
        coverage=coverage,
        bounded_probe_measured=qualified,
        full_trace_measured=False,
        full_model_qualified=False,
        end_to_end_gain_established=False,
        absolute_causal_attribution_established=False,
        design=bundle.get("design"),
        canonical_digest=evidence.CANONICAL_TRACE_SHA256,
        raw_reports=[dict(path=str(path), sha256=evidence.sha256(path)) for path in paths],
        runtime_script_hashes=expected["scripts"],
        comparison=comparison,
        output_bitwise_repeatability=[
            dict(
                trial=entry.get("trial"),
                policy=entry.get("policy"),
                capacity=left.get("capacity"),
                identical=exact(left.get("output_hashes"), right.get("output_hashes")),
            )
            for entry, report in runs
            for left, right in zip(array(report.get("pre_checks")), array(report.get("post_checks")))
            if isinstance(left, dict) and isinstance(right, dict)
        ],
        queue_clock_diagnostics=[
            dict(
                trial=key[0],
                policy=key[1],
                capacity=key[2],
                repeat=key[3],
                queue_span_us=sample["queue_span_us"],
                wall_us=sample["wall_us"],
            )
            for key, sample in sorted(indexed.items())
            if sample["queue_span_us"] > sample["wall_us"]
        ],
        samples=[sample for _, report in runs for sample in array(report.get("samples")) if isinstance(sample, dict)],
        statistics_contract=dict(
            filtering="none; all observations retained, including outliers",
            delta="upstream minus whole; positive means whole reduces the measured endpoint",
            reduction="100 * sum(upstream - whole) / sum(upstream); not the mean of per-pair percentages",
            percentiles="linear interpolation at (n-1)*p over unfiltered observations",
            pairing="same round, default capacity and repeat in distinct sequential fresh processes",
            independence="repeats are nested within process blocks; 80 pairs are not 80 independent process trials",
            output_repeatability="pre/post hashes are diagnostic; correctness uses the FP32 tolerance checks",
        ),
        timing_boundaries=dict(
            wall_us="Pinned preparation, required transfers, native producer, attention replay and final join.",
            preparation_enqueue_us="Host fill/enqueue interval, not completed H2D duration.",
            queue_span_us="Queue span includes submission gaps and dependency waits, not pure attention task time.",
            submit_join_cpu_us="Host submission overlaps device work and must not be added to queue time.",
        ),
        limitation="AB/BA blocks reduce order imbalance but cannot eliminate between-process drift. "
        "This bounded attention-chain comparison does not establish absolute causal attribution, "
        "full dynamic-trace performance or full-model/end-to-end speedup. Any difference concerns "
        "the combined complete-row assignment and FD removal, not isolated FD merge time.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--selection", type=Path)
    args = parser.parse_args()
    source, output = args.input.resolve(strict=True), args.output.resolve()
    bundle = evidence.load_json(source)
    protected = {
        source,
        Path(args.trace or evidence.RESULTS_DIRECTORY / "structural-dynamic-trace-v1.json").resolve(),
        Path(args.selection or evidence.RESULTS_DIRECTORY / "q4-tile-full-replay-v8-queue0.rank0.json").resolve(),
    }
    if isinstance(bundle, dict) and isinstance(bundle.get("runs"), list):
        for entry in bundle["runs"]:
            if isinstance(entry, dict):
                with contextlib.suppress(ValueError, OSError):
                    protected.add(resolve_report(entry, source.parent))
    if output in protected:
        parser.error("--output cannot overwrite bundle, raw process reports or canonical schedule evidence")
    result = analyze(bundle, base_dir=source.parent, trace_path=args.trace, selection_path=args.selection)
    result["input"] = dict(path=str(source), sha256=evidence.sha256(source))
    result["analyzer_sha256"] = evidence.sha256(__file__)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="ascii")
    print(json.dumps(dict(output=str(output), state=result["state"], protocol_qualified=result["protocol_qualified"])))
    return 0 if result["protocol_qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
