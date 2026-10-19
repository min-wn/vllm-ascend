# SPDX-License-Identifier: Apache-2.0
"""Audit the bounded A2 consumer diagnostic without equating queue time to kernels.

Two fresh-process blocks compare the same resident tensors at two original
default capacities. Profiler observations are diagnostic and separate from the
unprofiled wall endpoint; they cannot establish hardware utilization or E2E gain.
"""

import argparse
import contextlib
import hashlib
import json
import math
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path

import aclgraph_fia_sink_stage6_analyze as controls
from aclgraph_fia_sink_stage3_trace import load_trace_schedule

MODES = ("upstream", "whole", "legacy")
ORDERS = (MODES, tuple(reversed(MODES)))
CAPACITIES = (400, 512)
ROUNDS, REPEATS, WARMUPS, PROFILE_REPEATS, OCCURRENCES = 2, 10, 5, 3, 16
METRICS = controls.METRICS
evidence = controls.evidence


def expected_contract(trace_path=None, selection_path=None):
    independent = evidence.expected_schedule(trace_path, selection_path)
    schedule = controls.normalized(load_trace_schedule(trace_path, selection_path))
    records = [
        dict(
            next(row for row in schedule["records"] if row["physical_tokens"] == capacity),
            page_count=schedule["owner_pages"]["page_count"],
        )
        for capacity in CAPACITIES
    ]
    contract = {key: value for key, value in schedule.items() if key != "records"}
    contract["installed_catalog_revalidated"] = True
    scripts = set(controls.RUNTIME_SCRIPTS) | {
        "aclgraph_fia_sink_stage7_probe.py",
        "aclgraph_fia_sink_consumer_probe.py",
        "aclgraph_fia_sink_consumer_suite.py",
    }
    return dict(
        schedule=schedule,
        independent=independent,
        selected=records,
        trace_contract=contract,
        scripts={name: evidence.sha256(Path(__file__).with_name(name)) for name in sorted(scripts)},
        updatable_graph_sha256=evidence.sha256(
            Path(__file__).resolve().parents[1] / "vllm_ascend/compilation/updatable_graph.py"
        ),
        updatable_graph_lf_sha256=hashlib.sha256(
            (Path(__file__).resolve().parents[1] / "vllm_ascend/compilation/updatable_graph.py")
            .read_bytes()
            .replace(b"\r\n", b"\n")
        ).hexdigest(),
    )


def expected_design():
    return dict(
        modes=list(MODES),
        capacities=list(CAPACITIES),
        steps=[0, 29],
        occurrences=OCCURRENCES,
        rounds=ROUNDS,
        repeats=REPEATS,
        warmups=WARMUPS,
        profile_repeats=PROFILE_REPEATS,
        process_orders=[list(order) for order in ORDERS],
        seed=1703,
    )


def execution_issues(bundle, runs):
    issues = []
    if not isinstance(bundle.get("design"), dict) or any(
        not controls.exact(bundle["design"].get(key), value) for key, value in expected_design().items()
    ):
        issues.append("design differs from the fixed two-round consumer diagnostic")
    if bundle.get("state") != "completed" or not controls.exact(bundle.get("returncode"), 0):
        issues.append("suite did not complete successfully")
    if bundle.get("performance_measured") is not True or any(
        bundle.get(flag) is not False for flag in controls.SCOPE_FLAGS
    ):
        issues.append("suite must qualify bounded performance and exclude full-trace/model/E2E claims")
    start, finish = bundle.get("started_unix"), bundle.get("finished_unix")
    if not evidence.finite(start) or not evidence.finite(finish) or finish <= start:
        issues.append("suite execution interval is invalid")
        start, finish = -math.inf, math.inf
    wanted = [(trial, mode) for trial, order in enumerate(ORDERS) for mode in order]
    if len(runs) != len(wanted):
        issues.append("exactly six raw process reports are required")
    previous, pids, children = None, set(), set()
    for index, (entry, report) in enumerate(runs):
        if index >= len(wanted) or not controls.exact(
            [entry.get("trial"), entry.get("mode"), entry.get("process_order_index")], [*wanted[index], index]
        ):
            issues.append(f"run {index}: fresh-process order mismatch")
        begin, end = entry.get("launched_unix"), entry.get("finished_unix")
        if not evidence.finite(begin) or not evidence.finite(end) or not start <= begin < end <= finish:
            issues.append(f"run {index}: process interval is invalid")
        elif previous is not None and begin < previous:
            issues.append(f"run {index}: process execution overlaps")
        if evidence.finite(end):
            previous = end
        pid, child = entry.get("pid"), report.get("child_pid")
        if not controls.integer(pid, 1) or pid in pids or not controls.integer(child, 1) or child in children:
            issues.append(f"run {index}: distinct launcher and native child identities required")
        if controls.integer(pid, 1):
            pids.add(pid)
        if controls.integer(child, 1):
            children.add(child)
        if not controls.exact(entry.get("returncode"), 0):
            issues.append(f"run {index}: unsuccessful process")
    if pids & children:
        issues.append("launcher and child process identities overlap")
    return issues


def descriptor_issues(report, expected, mode):
    """Audit mode-specific copy counts without pretending host lengths use H2D."""
    issues, summaries, addresses = [], report.get("prepared_descriptors"), report.get("captured_addresses")
    capacities = {str(capacity) for capacity in CAPACITIES}
    if not isinstance(summaries, dict) or set(summaries) != capacities:
        return ["persistent preparation summaries must cover both capacities"]
    if not isinstance(addresses, dict) or set(addresses) != capacities:
        return ["captured descriptors must cover both capacities"]
    count = REPEATS + WARMUPS + PROFILE_REPEATS + 2
    for capacity in CAPACITIES:
        sink = mode != "legacy"
        wanted = dict(
            mode="sink" if sink else "legacy",
            physical_tokens=capacity,
            physical_requests=capacity // 4,
            host_pinned=dict.fromkeys(("q_lengths", "kv_lengths", "block_table") if sink else ("block_table",), True),
            stable_host_allocations=True,
            stable_device_allocations=True,
            host_allocation_count=3 if sink else 1,
            device_allocation_count=0,
            setup_copies=dict(q_lengths=int(sink), kv_lengths=0, block_table=0),
            step_copies=dict(q_lengths=0, kv_lengths=count if sink else 0, block_table=count),
            prepared_steps=count,
            completed_steps=count,
            inflight=False,
            q4_invariant=dict(query_length=4, cumulative_q=list(range(4, capacity + 1, 4))),
        )
        item = summaries[str(capacity)]
        if not isinstance(item, dict) or any(not controls.exact(item.get(key), value) for key, value in wanted.items()):
            issues.append(f"capacity {capacity}: preparation/copy/lifetime contract differs")
        shapes = dict(
            q_lengths=[capacity // 4],
            kv_lengths=[capacity // 4],
            block_table=[capacity // 4, expected["schedule"]["owner_pages"]["width"]],
            atten_mask=[2048, 2048],
        )
        if sink:
            shapes["metadata"] = [1024]
        shapes.update({f"{name}{index}": [capacity, 12, 256] for name in ("query", "output") for index in range(16)})
        saved, pointers = addresses[str(capacity)], []
        if not sink and isinstance(saved, dict):
            workspaces = {name: value for name, value in saved.items() if name.startswith("workspace")}
            if len(workspaces) != 1 or any(
                not isinstance(value, dict)
                or not isinstance(value.get("shape"), list)
                or not value["shape"]
                or any(not controls.integer(size, 1) for size in value["shape"])
                for value in workspaces.values()
            ):
                issues.append(f"capacity {capacity}: shared legacy workspace descriptor is absent or invalid")
            else:
                shapes.update({name: value["shape"] for name, value in workspaces.items()})
        if not isinstance(saved, dict) or set(saved) != set(shapes):
            issues.append(f"capacity {capacity}: exact descriptor set differs")
            continue
        for name, shape in shapes.items():
            entry = saved[name]
            if (
                not isinstance(entry, dict)
                or not controls.integer(entry.get("pointer"), 1)
                or not controls.exact(entry.get("shape"), shape)
                or not controls.exact(entry.get("stride"), [math.prod(shape[i + 1 :]) for i in range(len(shape))])
            ):
                issues.append(f"capacity {capacity}: captured {name} descriptor differs")
            else:
                pointers.append(entry["pointer"])
        if len(set(pointers)) != len(pointers):
            issues.append(f"capacity {capacity}: captured pointers alias")
    return issues


def legacy_issues(value, record, *, per_replay=False):
    if not isinstance(value, dict):
        return ["real legacy update evidence is required"]
    wanted = dict(
        task_count=16, current_query_lengths=record["actual_seq_qlen"], current_kv_lengths=record["kv_lengths"]
    )
    issues = []
    if any(not controls.exact(value.get(key), expected) for key, expected in wanted.items()):
        issues.append("legacy tasks/current true host lengths differ")
    workspace = value.get("workspace_bytes")
    if not isinstance(workspace, list) or len(workspace) != 1 or not controls.integer(workspace[0], 1):
        issues.append("legacy workspace extent must be positive")
    if per_replay and (
        not controls.exact(value.get("update_calls_delta"), 1)
        or not controls.exact(value.get("updated_tasks_delta"), 16)
    ):
        issues.append("legacy replay did not update every captured attention task")
    return issues


def qualification_issues(report, expected, mode):
    issues, plans = [], {}
    for label in ("pre_checks", "post_checks"):
        checks = report.get(label)
        if not isinstance(checks, list) or len(checks) != len(CAPACITIES):
            issues.append(f"{label}: both capacities require independent qualification")
            continue
        for check, record in zip(checks, expected["selected"], strict=True):
            capacity = record["physical_tokens"]
            if not isinstance(check, dict):
                issues.append(f"{label}/{capacity}: check must be an object")
                continue
            flags = (
                "finite",
                "all_outputs_finite",
                "exact_zero_padding",
                "stable_addresses",
                "exact_descriptors_read_only",
            )
            if (
                not controls.exact(check.get("capacity"), capacity)
                or not controls.exact(check.get("step"), record["step"])
                or not controls.exact(check.get("private_occurrences"), 16)
                or any(check.get(flag) is not True for flag in flags)
            ):
                issues.append(f"{label}/{capacity}: finite/padding/address/read-only qualification failed")
            rows = check.get("fp32_sampled_rows")
            wanted = {(occ, row) for occ in (0, 7, 15) for row in (0, record["batch"] // 2, record["batch"] - 1)}
            if (
                not isinstance(rows, list)
                or len(rows) != 9
                or any(
                    not isinstance(row, dict)
                    or not controls.integer(row.get("occurrence"))
                    or not controls.integer(row.get("row"))
                    or not evidence.finite(row.get("max_abs_error"))
                    for row in rows
                )
                or {(row["occurrence"], row["row"]) for row in rows} != wanted
            ):
                issues.append(f"{label}/{capacity}: exact nine FP32 row identities required")
            else:
                for row in rows:
                    kv = record["kv_lengths"][row["row"]]
                    pages = record["block_table"][row["row"]][: (kv + 127) // 128]
                    if (
                        not controls.exact(row.get("kv_length"), kv)
                        or not controls.exact(row.get("original_page_ids"), pages)
                        or not controls.exact(row.get("local_page_ids"), list(range(len(pages))))
                    ):
                        issues.append(f"{label}/{capacity}: current ordered-page FP32 oracle identity differs")
            outputs = check.get("output_hashes")
            if (
                not isinstance(outputs, list)
                or len(outputs) != 16
                or any(
                    not isinstance(row, dict)
                    or not controls.exact(row.get("shape"), [capacity, 12, 256])
                    or row.get("dtype") != "torch.bfloat16"
                    or not controls.sha(row.get("sha256"))
                    for row in outputs
                )
            ):
                issues.append(f"{label}/{capacity}: all private output byte identities required")
            if mode == "legacy":
                issues.extend(f"{label}/{capacity}: {item}" for item in legacy_issues(check.get("legacy"), record))
                count = 1 if label == "pre_checks" else 2
                value = check.get("legacy", {})
                if (
                    not controls.exact(check.get("legacy_update_calls_delta"), 1)
                    or not controls.exact(check.get("legacy_updated_tasks_delta"), 16)
                    or not controls.exact(value.get("update_calls"), count)
                    or not controls.exact(value.get("updated_tasks"), count * 16)
                    or value.get("instrumentation_enabled") is not True
                ):
                    issues.append(f"{label}/{capacity}: successful real update dispatch/apply audit differs")
            else:
                plan = check.get("plan")
                issues.extend(f"{label}/{capacity}: {item}" for item in controls.plan_issues(plan, record, mode))
                if isinstance(plan, dict):
                    if capacity in plans and plans[capacity] != plan.get("sha256"):
                        issues.append(f"{label}/{capacity}: fixed-record plan changed")
                    plans[capacity] = plan.get("sha256")
    if mode != "legacy":
        preflight = report.get("metadata_preflight")
        if not isinstance(preflight, list) or len(preflight) != 2:
            issues.append("both sink captures require preflight native metadata")
        else:
            for item, record in zip(preflight, expected["selected"], strict=True):
                if (
                    not isinstance(item, dict)
                    or not controls.exact(item.get("record"), record)
                    or item.get("verified_before_attention_capture") is not True
                ):
                    issues.append("preflight must retain exact record and pre-capture validation")
                    continue
                issues.extend(controls.plan_issues(item, record, mode))
                if plans.get(record["physical_tokens"]) != item.get("sha256"):
                    issues.append("preflight plan differs from fixed-record qualification")
    return issues


def report_issues(report, entry, expected):
    mode = entry.get("mode")
    if mode not in MODES:
        return ["unsupported native mode"]
    wanted = dict(
        state="completed",
        stage="completed",
        returncode=0,
        npu_work_executed=True,
        performance_measured=True,
        occurrences=16,
        repeats=10,
        warmup_replays=5,
        profile_repeats=3,
        trial=entry.get("trial"),
        mode=mode,
        policy=mode,
        canonical_digest=evidence.CANONICAL_TRACE_SHA256,
        selected_records=expected["selected"],
        trace_contract=expected["trace_contract"],
        core_limit=dict(cube_core_num=20, vector_core_num=40),
        independent_graph_pool_per_bucket=True,
        separate_qualification=True,
        timed_samples_individually_qualified=False,
        frame_queries_match_base_prefixes=True,
        actual_task_audit_scope="pre/post qualification only; timing and profiles use original graph.update",
    )
    wanted.update(dict.fromkeys(controls.SCOPE_FLAGS, False))
    issues = [
        f"report {key} differs from execution contract"
        for key, value in wanted.items()
        if not controls.exact(report.get(key), value)
    ]
    issues.extend(controls.input_issues(report, expected))
    issues.extend(controls.provenance_issues(report, "upstream" if mode == "legacy" else mode, expected))
    issues.extend(descriptor_issues(report, expected, mode))
    issues.extend(qualification_issues(report, expected, mode))
    wanted_warmups = [dict(capacity=capacity, replays=5, outside_timing=True) for capacity in CAPACITIES]
    if not controls.exact(report.get("warmups"), wanted_warmups):
        issues.append("five actual warmup replays per capacity are required")
    if mode == "legacy":
        final = report.get("legacy_contract")
        if not isinstance(final, dict) or set(final) != {str(capacity) for capacity in CAPACITIES}:
            issues.append("final legacy audited contracts must cover both capacities")
        else:
            for record in expected["selected"]:
                value = final[str(record["physical_tokens"])]
                issues.extend(legacy_issues(value, record))
                if not isinstance(value, dict):
                    continue
                if (
                    not controls.exact(value.get("update_calls"), 2)
                    or not controls.exact(value.get("updated_tasks"), 32)
                    or value.get("instrumentation_enabled") is not True
                ):
                    issues.append("only the two qualification replays may have Python update instrumentation")
    frames, after = report.get("frame_query_identity"), report.get("post_frame_query_identity")
    if (
        not isinstance(frames, dict)
        or set(frames) != {str(capacity) for capacity in CAPACITIES}
        or not controls.exact(frames, after)
    ):
        issues.append("captured query-frame byte identities are absent or changed")
    else:
        for capacity in CAPACITIES:
            rows = frames[str(capacity)]
            if (
                not isinstance(rows, list)
                or len(rows) != 16
                or any(
                    not isinstance(row, dict)
                    or not controls.exact(row.get("shape"), [capacity, 12, 256])
                    or row.get("dtype") != "torch.bfloat16"
                    or not controls.sha(row.get("sha256"))
                    for row in rows
                )
            ):
                issues.append(f"capacity {capacity}: private frame hashes are invalid")
    installed = report.get("installed_catalog", {})
    selection = installed.get("selection", {}) if isinstance(installed, dict) else {}
    if (
        not isinstance(installed, dict)
        or installed.get("installed_catalog_revalidated") is not True
        or not isinstance(selection, dict)
        or not controls.exact(selection.get("capture_sizes"), expected["schedule"]["default_capture_sizes"])
        or not controls.exact(selection.get("dispatched_tokens"), expected["independent"]["dispatched_tokens"])
        or selection.get("config_source") != "installed EngineArgs defaults; no capture sizes supplied"
    ):
        issues.append("installed default capture catalog was not independently revalidated")
    return issues


def collect_samples(runs, expected):
    indexed, issues, count = {}, [], 0
    records = {record["physical_tokens"]: record for record in expected["selected"]}
    for entry, report in runs:
        samples, sequence = report.get("samples"), []
        if not isinstance(samples, list):
            issues.append("samples must be a list")
            continue
        count += len(samples)
        for sample in samples:
            if not isinstance(sample, dict) or any(
                not controls.integer(sample.get(key)) for key in ("trial", "capacity", "repeat", "step", "batch")
            ):
                issues.append("sample requires strict integer identities")
                continue
            trial, mode, capacity, repeat = (sample.get(key) for key in ("trial", "mode", "capacity", "repeat"))
            if (
                trial != entry.get("trial")
                or mode != entry.get("mode")
                or mode not in MODES
                or sample.get("policy") != mode
                or not 0 <= trial < ROUNDS
                or not 0 <= repeat < REPEATS
                or capacity not in records
                or sample["step"] != records[capacity]["step"]
                or sample["batch"] != records[capacity]["batch"]
            ):
                issues.append("sample differs from process/canonical identity")
                continue
            key = trial, mode, capacity, repeat
            if key in indexed:
                issues.append("duplicate sample identity")
                continue
            if any(not evidence.finite(sample.get(metric)) for metric in METRICS) or sample["wall_us"] <= 0:
                issues.append("sample metrics must be finite/nonnegative with positive wall")
                continue
            if sample["preparation_enqueue_us"] + sample["submit_join_cpu_us"] > sample["wall_us"] + 1e-6:
                issues.append("same-clock timing subdivisions exceed wall")
            calls, tasks = (None, None) if mode == "legacy" else (0, 0)
            if not controls.exact(sample.get("legacy_update_calls_delta"), calls) or not controls.exact(
                sample.get("legacy_updated_tasks_delta"), tasks
            ):
                issues.append("timed replay update instrumentation contract differs")
            if mode == "legacy" and sample.get("legacy_counter_instrumented") is not False:
                issues.append("timed legacy replay must use the original uninstrumented graph.update")
            indexed[key] = sample
            sequence.append((repeat, capacity))
        if controls.integer(entry.get("trial")):
            wanted = [
                (repeat, capacity)
                for repeat in range(REPEATS)
                for capacity in (
                    CAPACITIES[(entry["trial"] + repeat) % 2 :] + CAPACITIES[: (entry["trial"] + repeat) % 2]
                )
            ]
            if sequence != wanted:
                issues.append("sample order differs from rotating original-capacity order")
    wanted = {
        (trial, mode, capacity, repeat)
        for trial in range(ROUNDS)
        for mode in MODES
        for capacity in CAPACITIES
        for repeat in range(REPEATS)
    }
    missing = sorted(wanted - indexed.keys())
    if missing:
        issues.append("sample matrix is incomplete")
    return indexed, dict(
        expected_sample_count=120,
        raw_sample_count=count,
        valid_sample_count=len(indexed),
        complete=not issues,
        missing=[list(key) for key in missing],
        issues=issues,
    )


def interval(event):
    try:
        start, duration = Decimal(str(event["ts"])), Decimal(str(event["dur"]))
    except (KeyError, TypeError, InvalidOperation) as error:
        raise ValueError("complete profiler events require numeric ts/dur") from error
    if not start.is_finite() or not duration.is_finite() or duration < 0:
        raise ValueError("invalid profiler interval")
    return start, start + duration


def duration_sum(events):
    return float(sum((interval(row)[1] - interval(row)[0] for row in events), Decimal(0)))


def union_duration(intervals):
    total, previous = Decimal(0), None
    for start, end in sorted(intervals):
        if previous is None:
            previous = start, end
        elif start <= previous[1]:
            previous = previous[0], max(previous[1], end)
        else:
            total += previous[1] - previous[0]
            previous = start, end
    if previous is not None:
        total += previous[1] - previous[0]
    return float(total)


def profile_replay(events, label, mode):
    """Require one completed native model before assigning hardware executions."""
    start, end = interval(label)
    selected = [
        row
        for row in events
        if row.get("ph") == "X" and start <= interval(row)[0] and interval(row)[1] <= end + Decimal("1")
    ]
    hardware = [row for row in selected if row.get("args", {}).get("Task Type")]
    notifies = [row for row in hardware if row.get("args", {}).get("Task Type") == "NOTIFY_RECORD"]
    executes = [row for row in hardware if row.get("args", {}).get("Task Type") == "MODEL_EXECUTE"]
    if len(notifies) != 1 or len(executes) != 1:
        raise ValueError("each label requires exactly one hardware model execute and completion notification")
    completion = notifies[0]
    args = completion["args"]
    model, stream = args.get("Model Id"), args.get("Physic Stream Id")
    if not controls.integer(model) or model == 4294967295 or not controls.integer(stream):
        raise ValueError("completion must identify an actual captured model and physical stream")
    launch, completed = interval(executes[0])[0], interval(completion)[1]
    # The CPU label selects a unique launch, while the native completion owns
    # task attribution. MODEL_EXECUTE itself carries a sentinel model ID.
    tasks = [
        row
        for row in events
        if row.get("ph") == "X"
        and row.get("args", {}).get("Model Id") == model
        and row.get("args", {}).get("Task Type")
        and launch <= interval(row)[0] <= completed
    ]
    if any(row["args"].get("Physic Stream Id") != stream for row in tasks):
        raise ValueError("captured model spans an unexpected hardware stream")
    kernels = [row for row in hardware if str(row["args"]["Task Type"]).startswith("KERNEL_")]
    if any(row not in tasks for row in kernels):
        raise ValueError("unattributed hardware kernel inside replay label")
    once = [
        row
        for row in tasks
        if str(row["args"].get("Task Type", "")).startswith("KERNEL_")
        or row["args"].get("Task Type") == "NOTIFY_RECORD"
    ]
    identities = [(row["args"].get("Task Id"), row["args"].get("Subtask Id")) for row in once]
    if len(set(identities)) != len(identities) or any(type(key[0]) is not int for key in identities):
        raise ValueError("duplicate or missing native model task identity")
    if any(interval(row)[1] > completed for row in tasks):
        raise ValueError("model task is outside execute-to-completion interval")
    attention = sorted(
        [row for row in tasks if row["args"]["Task Type"] == "KERNEL_MIX_AIC"], key=lambda row: interval(row)[0]
    )
    metadata = [row for row in tasks if row["args"]["Task Type"] == "KERNEL_AICPU"]
    zero = [row for row in tasks if row["args"]["Task Type"] == "KERNEL_AIVEC"]
    expected_native = (16, 0, 0) if mode == "legacy" else (16, 1, 1)
    if (len(attention), len(metadata), len(zero)) != expected_native or len(kernels) != sum(expected_native):
        raise ValueError("attention/metadata/zero native task counts differ from the exact captured graph")
    if any(interval(row)[0] == interval(row)[1] for row in attention):
        raise ValueError("attention hardware duration must be positive")
    if mode != "legacy" and (
        "SinkMetadata" not in metadata[0].get("name", "")
        or "ZerosLike" not in zero[0].get("name", "")
        or interval(zero[0])[1] > interval(metadata[0])[0]
        or interval(metadata[0])[1] > interval(attention[0])[0]
    ):
        raise ValueError("shared metadata/zero identity or producer-before-consumer ordering differs")
    fragment = "FusedInferAttentionScore" if mode == "legacy" else "FusedInferAttentionScoreV2Sink"
    if any(fragment not in row.get("name", "") or (mode == "legacy" and "Sink" in row["name"]) for row in attention):
        raise ValueError("attention hardware identity differs from the intended consumer family")
    if any(interval(left)[1] > interval(right)[0] for left, right in zip(attention, attention[1:])):
        raise ValueError("sequential private attention executions overlap")
    cpu_update = [
        row
        for row in selected
        if row.get("cat") == "cpu_op" and row.get("name") == "npu::npu_fused_infer_attention_score"
    ]
    dma = [row for row in hardware if row["args"]["Task Type"] == "PCIE_DMA_SQE"]
    if len(cpu_update) != (16 if mode == "legacy" else 0) or len(dma) != (16 if mode == "legacy" else 0):
        raise ValueError("real legacy CPU update/PCIe DMA task counts differ")
    if any(row["args"]["Task Type"] == "SDMA_SQE" for row in hardware):
        raise ValueError("preparation DMA leaked into the replay-only profiler label")
    first, last = interval(attention[0])[0], interval(attention[-1])[1]
    waits = [row for row in tasks if row["args"]["Task Type"] == "MEM_WAIT_VALUE"]
    overlap = union_duration(
        (max(first, interval(row)[0]), min(last, interval(row)[1]))
        for row in waits
        if interval(row)[1] > first and interval(row)[0] < last
    )
    total = duration_sum(attention)
    return dict(
        label=label["name"],
        model_id=model,
        physical_stream_id=stream,
        attention_tasks=16,
        attention_duration_sum_us=total,
        attention_task_durations_us=[float(interval(row)[1] - interval(row)[0]) for row in attention],
        attention_first_to_last_envelope_us=float(last - first),
        between_attention_noncompute_us=float(last - first) - total,
        metadata_tasks=len(metadata),
        metadata_duration_sum_us=duration_sum(metadata),
        metadata_zero_tasks=len(zero),
        metadata_zero_duration_sum_us=duration_sum(zero),
        cpu_legacy_update_calls=len(cpu_update),
        cpu_legacy_update_duration_sum_us=duration_sum(cpu_update),
        legacy_pcie_dma_tasks=len(dma),
        legacy_pcie_dma_duration_sum_us=duration_sum(dma),
        mem_wait_value_tasks=len(waits),
        mem_wait_value_duration_sum_us=duration_sum(waits),
        mem_wait_value_duration_union_us=union_duration(interval(row) for row in waits),
        mem_wait_value_overlap_attention_envelope_us=overlap,
        completion_notifications=1,
        hardware_task_type_counts=dict(sorted(Counter(row["args"]["Task Type"] for row in hardware).items())),
        profiled_label_us=float(end - start),
        preparation_dma_in_label=sum(row["args"]["Task Type"] == "SDMA_SQE" for row in hardware),
        task_identities=[
            dict(
                name=row.get("name"),
                model_id=model,
                physical_stream_id=stream,
                task_id=row["args"].get("Task Id"),
                subtask_id=row["args"].get("Subtask Id"),
            )
            for row in once
        ],
    )


def parse_profile(document, mode, capacity):
    events = document.get("traceEvents") if isinstance(document, dict) else document
    if not isinstance(events, list) or any(not isinstance(row, dict) for row in events):
        raise ValueError("Chrome profiler data must be an event list or a traceEvents object")
    # Validate intervals once, including otherwise unselected complete events.
    for event in events:
        if "args" in event and not isinstance(event["args"], dict):
            raise ValueError("profiler event args must be an object")
        if event.get("ph") == "X":
            interval(event)
    labels = [row for row in events if row.get("ph") == "X" and str(row.get("name", "")).startswith("consumer/")]
    wanted = [f"consumer/{mode}/cap{capacity}/repeat{repeat}" for repeat in range(PROFILE_REPEATS)]
    if [row["name"] for row in labels] != wanted:
        raise ValueError("exact three chronological profiler labels are required")
    if any(interval(left)[1] > interval(right)[0] for left, right in zip(labels, labels[1:])):
        raise ValueError("profiler replay labels overlap")
    return [
        dict(capacity=capacity, mode=mode, repeat=index, **profile_replay(events, label, mode))
        for index, label in enumerate(labels)
    ]


def profile_statistics(rows):
    fields = (
        "attention_duration_sum_us",
        "attention_first_to_last_envelope_us",
        "between_attention_noncompute_us",
        "metadata_duration_sum_us",
        "metadata_zero_duration_sum_us",
        "cpu_legacy_update_duration_sum_us",
        "legacy_pcie_dma_duration_sum_us",
        "mem_wait_value_duration_sum_us",
        "mem_wait_value_overlap_attention_envelope_us",
    )
    per_process = [
        dict(
            trial=trial,
            mode=mode,
            capacity=capacity,
            replays=len(selected),
            metrics={key: evidence.stats([row[key] for row in selected]) for key in fields},
        )
        for trial in range(ROUNDS)
        for mode in MODES
        for capacity in CAPACITIES
        if (
            selected := [row for row in rows if (row["trial"], row["mode"], row["capacity"]) == (trial, mode, capacity)]
        )
    ]
    per_mode = [
        dict(
            mode=mode,
            capacity=capacity,
            replays=len(selected),
            metrics={key: evidence.stats([row[key] for row in selected]) for key in fields},
        )
        for mode in MODES
        for capacity in CAPACITIES
        if (selected := [row for row in rows if (row["mode"], row["capacity"]) == (mode, capacity)])
    ]
    return dict(per_process=per_process, per_mode_capacity=per_mode)


def collect_profiles(runs, base_dir):
    rows, issues, sources, seen = [], [], [], set()
    for entry, report in runs:
        profiles = report.get("profiles")
        if not isinstance(profiles, list) or len(profiles) != 2:
            issues.append("each process requires two separately exported profiles")
            continue
        report_path = controls.resolve_report(entry, base_dir)
        for item, capacity in zip(profiles, CAPACITIES, strict=True):
            try:
                if not isinstance(item, dict) or not controls.exact(item.get("capacity"), capacity):
                    raise ValueError("profile capacities are not in original declared order")
                mode = entry["mode"]
                wanted = [f"consumer/{mode}/cap{capacity}/repeat{repeat}" for repeat in range(PROFILE_REPEATS)]
                calls, tasks = (None, None) if mode == "legacy" else (0, 0)
                if (
                    item.get("outside_timing") is not True
                    or not controls.exact(item.get("repeats"), 3)
                    or not controls.exact(item.get("labels"), wanted)
                    or not controls.exact(item.get("legacy_update_calls_delta"), calls)
                    or not controls.exact(item.get("legacy_updated_tasks_delta"), tasks)
                    or item.get("legacy_counter_instrumented") is not False
                    or item.get("preparation_in_trace_outside_label") is not True
                    or item.get("label_scope")
                    != "native plan/update + replay + final join; prepare/H2D completed before label"
                ):
                    raise ValueError("profile execution boundaries/update counters differ")
                path = controls.resolve_report(dict(report_path=item.get("path")), report_path.parent)
                if path in seen:
                    raise ValueError("duplicate profiler evidence path")
                seen.add(path)
                if not controls.sha(item.get("sha256")) or evidence.sha256(path) != item["sha256"]:
                    raise ValueError("profiler raw byte SHA256 mismatch")
                parsed = parse_profile(evidence.load_json(path), mode, capacity)
                rows.extend(dict(trial=entry["trial"], **row) for row in parsed)
                sources.append(dict(path=str(path), sha256=evidence.sha256(path)))
            except (ValueError, OSError, KeyError, TypeError) as error:
                issues.append(f"trial {entry.get('trial')}/{entry.get('mode')}/cap{capacity}: {error}")
    if len(rows) != 36:
        issues.append("exactly 36 qualified diagnostic replays are required")
    return dict(
        qualified=not issues,
        issues=issues,
        replays=rows,
        sources=sources,
        summary=profile_statistics(rows),
        attention_tasks=sum(row["attention_tasks"] for row in rows),
        metadata_tasks=sum(row["metadata_tasks"] for row in rows),
        legacy_cpu_update_calls=sum(row["cpu_legacy_update_calls"] for row in rows),
        legacy_pcie_dma_tasks=sum(row["legacy_pcie_dma_tasks"] for row in rows),
    )


def runtime_source_issues(runs, base_dir, expected):
    issues = []
    for entry, report in runs:
        if entry.get("mode") != "legacy":
            continue
        try:
            sources = report.get("runtime_sources")
            name = "vllm_ascend/compilation/updatable_graph.py"
            if not isinstance(sources, dict) or name not in sources or not isinstance(sources[name], dict):
                raise ValueError("runtime UpdatableGraph source evidence is required")
            item = sources[name]
            wanted = item.get("sha256")
            if not controls.sha(wanted) or item.get("snapshot_sha256") != wanted:
                raise ValueError("runtime UpdatableGraph source and snapshot byte identities differ")
            parent = controls.resolve_report(entry, base_dir).parent
            snapshot = controls.resolve_report(dict(report_path=item.get("snapshot_path")), parent)
            if evidence.sha256(snapshot) != wanted:
                raise ValueError("runtime UpdatableGraph snapshot bytes differ")
            if (
                hashlib.sha256(snapshot.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
                != expected["updatable_graph_lf_sha256"]
            ):
                raise ValueError(
                    "runtime UpdatableGraph source differs from the audited checkout after LF normalization"
                )
        except (ValueError, OSError, TypeError) as error:
            issues.append(f"trial {entry.get('trial')}/legacy: {error}")
    return issues


def comparisons(indexed):
    output = []
    for capacity in CAPACITIES:
        for left_mode, right_mode in (("upstream", "whole"), ("legacy", "upstream"), ("legacy", "whole")):
            pairs = [
                (indexed.get((trial, left_mode, capacity, repeat)), indexed.get((trial, right_mode, capacity, repeat)))
                for trial in range(ROUNDS)
                for repeat in range(REPEATS)
            ]
            pairs = [(left, right) for left, right in pairs if left is not None and right is not None]
            metrics = {}
            for metric in METRICS:
                left, right = [row[0][metric] for row in pairs], [row[1][metric] for row in pairs]
                delta = [a - b for a, b in zip(left, right, strict=True)]
                metrics[metric] = dict(
                    left=evidence.stats(left),
                    right=evidence.stats(right),
                    left_minus_right=evidence.stats(delta),
                    left_reduction_percent=100 * sum(delta) / sum(left) if sum(left) else None,
                    right_faster_count=sum(value > 0 for value in delta),
                )
            output.append(
                dict(
                    capacity=capacity,
                    left_mode=left_mode,
                    right_mode=right_mode,
                    paired_count=len(pairs),
                    metrics=metrics,
                )
            )
    return output


def analyze(bundle, *, base_dir, trace_path=None, selection_path=None):
    if not isinstance(bundle, dict):
        raise ValueError("consumer suite bundle must be an object")
    expected = expected_contract(trace_path, selection_path)
    issues = []
    base_dir = Path(base_dir).resolve(strict=True)
    runs, paths = controls.load_runs(bundle, base_dir, issues)
    issues.extend(execution_issues(bundle, runs))
    issues.extend(runtime_source_issues(runs, base_dir, expected))
    for index, (entry, report) in enumerate(runs):
        issues.extend(f"run {index}: {issue}" for issue in report_issues(report, entry, expected))
    for _, report in runs[1:]:
        for key in (
            "input_identity",
            "mask_identity",
            "frame_query_identity",
            "scripts",
            "extension_sha256",
            "device_name",
            "torch_version",
            "torch_npu_version",
        ):
            if key not in runs[0][1] or not controls.exact(report.get(key), runs[0][1][key]):
                issues.append(f"cross-process {key} differs or is absent")
    sink_runs = [(entry, report) for entry, report in runs if entry.get("mode") in ("upstream", "whole")]
    issues.extend(controls.common_control_issues(sink_runs))
    indexed, coverage = collect_samples(runs, expected)
    issues.extend(coverage["issues"])
    profiles = collect_profiles(runs, base_dir)
    valid = not issues
    return dict(
        schema_version=1,
        state="complete_bounded_diagnostic" if valid and profiles["qualified"] else "partial_or_invalid",
        protocol_qualified=valid,
        diagnostic_qualified=valid and profiles["qualified"],
        issues=issues,
        coverage=coverage,
        profile=profiles,
        comparison=comparisons(indexed),
        bounded_probe_measured=valid,
        full_trace_measured=False,
        full_model_qualified=False,
        end_to_end_gain_established=False,
        absolute_causal_attribution_established=False,
        hardware_utilization_measured=False,
        design=bundle.get("design"),
        canonical_digest=evidence.CANONICAL_TRACE_SHA256,
        raw_reports=[dict(path=str(path), sha256=evidence.sha256(path)) for path in paths],
        runtime_script_hashes=expected["scripts"],
        timing_boundaries=dict(
            wall_us="Actual preparation, required H2D, producer/update, replay, final join.",
            queue_span_us="Queue span includes submission gaps and waits; not kernel time.",
            preparation_enqueue_us="Host fill/enqueue; not completed H2D duration.",
            submit_join_cpu_us="Submission and join overlaps device work; do not add to queue.",
        ),
        limitations=[
            "Two process blocks are a bounded screen, not a stable speedup estimate.",
            "Profiler labels and task durations include profiler effects and are separate from wall trials.",
            "MEM_WAIT, CPU updates and DMA overlap; do not add them as an endpoint decomposition.",
            "No AIC metrics: MM1/MM2, bandwidth, core imbalance and FD cost are not localized.",
            "Fixed resident attention inputs; no model layers, token generation or serving.",
        ],
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
    if isinstance(bundle, dict):
        for entry in controls.array(bundle.get("runs")):
            if isinstance(entry, dict):
                with contextlib.suppress(ValueError, OSError):
                    path = controls.resolve_report(entry, source.parent)
                    protected.add(path)
                    report = evidence.load_json(path)
                    if not isinstance(report, dict):
                        continue
                    for item in controls.array(report.get("profiles")):
                        if isinstance(item, dict):
                            with contextlib.suppress(ValueError, OSError):
                                protected.add(controls.resolve_report(dict(report_path=item.get("path")), path.parent))
                    sources = report.get("runtime_sources")
                    for item in sources.values() if isinstance(sources, dict) else []:
                        if isinstance(item, dict):
                            with contextlib.suppress(ValueError, OSError):
                                protected.add(
                                    controls.resolve_report(dict(report_path=item.get("snapshot_path")), path.parent)
                                )
    if output in protected:
        parser.error("output cannot overwrite suite, raw reports, profiles or canonical sources")
    result = analyze(bundle, base_dir=source.parent, trace_path=args.trace, selection_path=args.selection)
    result["input"] = dict(path=str(source), sha256=evidence.sha256(source))
    result["analyzer_sha256"] = evidence.sha256(__file__)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="ascii")
    print(json.dumps(dict(output=str(output), diagnostic_qualified=result["diagnostic_qualified"])))
    return 0 if result["diagnostic_qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
