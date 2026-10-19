# SPDX-License-Identifier: Apache-2.0
"""Read collected stage-two evidence and produce an auditable compact JSON.

This imports no execution tools or Torch and never contacts the remote machine.
Missing, malformed, running and failed inputs remain explicit. A successful
attempt count cannot substitute for distinct expected case IDs. ``--output``
may be regenerated, but must never overwrite one of the original evidence files.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath

ROOT_FILES = (
    "build-status.json",
    "d256-patch-manifest.json",
    "deployed-tools.json",
    "staged-opp-manifest.json",
    "pipeline-status.json",
    "extension-status.json",
    "final-artifact-inventory.json",
)
GATES = ("b1", "b4", "shared")
CONSTANT_CASES = ("official_layout_d128_gqa4", "target_q4_d256_gqa6")
PIPELINE_STEPS = (
    "stage_opp",
    "eager_first",
    "eager_remaining",
    "d128_and_d256_constant",
    "graph_b1",
    "graph_b4",
    "graph_shared",
)
EXECUTION_TOOLS = (
    "aclgraph_fia_sink_stage1_smoke.py",
    "aclgraph_fia_sink_stage2_validate.py",
    "aclgraph_fia_sink_stage2_graph.py",
)
MAX_CUBE_CORES = 20
MAX_VECTOR_CORES = 40
MAX_FD_SLOTS = 24
METADATA_M_BASE = 512
METADATA_FD_M_BASE = 8
RTOL = 0.03
ATOL = 0.003
MIN_MASK_WITNESS = 10.0


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def pick(value, names):
    return {name: value[name] for name in names if name in value}


def valid_hash(value):
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def object_or_empty(value):
    return value if isinstance(value, dict) else {}


def nonnegative_integer(value):
    return type(value) is int and value >= 0


def finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def reject_json_constant(value):
    raise ValueError(f"nonfinite JSON constant: {value}")


def load_artifact(evidence, candidates):
    present = [evidence / relative for relative in candidates if (evidence / relative).exists()]
    record = dict(expected_paths=candidates, state="missing")
    if len(present) != 1:
        if present:
            record.update(state="invalid", error="ambiguous collection layout: multiple candidate files exist")
        return None, record
    path = present[0]
    record["relative_path"] = path.relative_to(evidence).as_posix()
    try:
        if not path.resolve(strict=True).is_relative_to(evidence) or not path.is_file():
            raise ValueError("evidence path is not a regular file inside the collection")
        raw = path.read_bytes()
        record.update(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        payload = json.loads(raw, parse_constant=reject_json_constant)
        if not isinstance(payload, dict):
            raise ValueError("status/manifest JSON must be an object")
        record["state"] = "present"
        return payload, record
    except (OSError, ValueError) as error:
        record.update(state="invalid", error=str(error))
        return None, record


def status_fields(data):
    return pick(
        data,
        (
            "state",
            "returncode",
            "checkpoint",
            "error",
            "npu_work_executed",
            "selected_group_complete",
            "selected_graph_gate_qualified",
            "smoke_passed",
        ),
    )


def finished(data):
    return data.get("state") == "completed" and type(data.get("returncode")) is int and data["returncode"] == 0


def compact_fingerprint(fingerprint, inventories):
    compact = {key: value for key, value in fingerprint.items() if key not in ("cases", "opp_artifacts")}
    compact["fingerprint_sha256"] = digest(fingerprint)
    artifacts = fingerprint.get("opp_artifacts")
    if isinstance(artifacts, dict):
        inventory_hash = digest(artifacts)
        inventories[inventory_hash] = dict(file_count=len(artifacts), inventory_sha256=inventory_hash)
        compact["opp_inventory_ref"] = inventory_hash
    return compact


def eager_cases():
    """Independent qualification contract; an input cannot shrink its own scope."""
    cases = []
    for causal in (False, True):
        suffix = "causal" if causal else "full"
        cases.extend(
            dict(id=f"b1-q4-kv{length}-{suffix}", q_lengths=[4], kv_lengths=[length], causal=causal, group="positive")
            for length in (127, 128, 129, 511, 512, 513, 1025, 4097)
        )
        cases.append(
            dict(
                id=f"ragged-{suffix}", q_lengths=[4, 1, 3], kv_lengths=[129, 511, 1025], causal=causal, group="positive"
            )
        )
    cases.extend(
        dict(
            id=f"b1-q4-kv{length}-causal-short",
            q_lengths=[4],
            kv_lengths=[length],
            causal=True,
            group="positive",
            purpose="short-context right-down causal semantic gate",
        )
        for length in (4, 7)
    )
    cases.append(
        dict(
            id="b1-q97-kv513-full-internal-loop-boundary",
            q_lengths=[97],
            kv_lengths=[513],
            causal=False,
            group="positive",
            purpose="internal M=582 multi-loop boundary only; no model/mixed-request support claim",
        )
    )
    cases.extend(
        [
            dict(id="zero-kv-only", q_lengths=[4], kv_lengths=[0], causal=False, group="zero-kv"),
            dict(id="zero-kv-mixed", q_lengths=[4, 4, 4], kv_lengths=[0, 129, 0], causal=False, group="zero-kv"),
            dict(id="zero-q-only", q_lengths=[0], kv_lengths=[129], causal=False, group="zero-q"),
            dict(id="zero-q-mixed", q_lengths=[0, 4, 0], kv_lengths=[129, 511, 0], causal=False, group="zero-q"),
        ]
    )
    return cases


def graph_cases(gate):
    if gate == "b1":
        return [
            dict(id=f"b1-{index}-kv{length}", q_lengths=[4], kv_lengths=[length], causal=True)
            for index, length in enumerate((4, 7, 127, 129, 513, 1025, 127, 4))
        ]
    return [
        dict(id=f"{gate}-{name}", q_lengths=query, kv_lengths=kv, causal=True)
        for name, query, kv in (
            ("capture", [4, 4, 4, 4], [127] * 4),
            ("holes", [4, 0, 4, 4], [129, 0, 513, 1025]),
            ("resume", [0, 4, 0, 4], [0, 1025, 0, 129]),
            ("shorten", [4, 4, 4, 0], [513, 127, 129, 0]),
            ("zero-kv", [4, 4, 0, 4], [0, 129, 0, 513]),
            ("empty", [0, 0, 0, 0], [0, 0, 0, 0]),
            ("all-live", [4, 4, 4, 4], [127, 129, 513, 1025]),
        )
    ]


def expected_cases(fingerprint, contract):
    issues = (
        []
        if digest(fingerprint.get("cases")) == digest(contract)
        else ["fingerprint.cases differs from fixed qualification contract"]
    )
    if fingerprint.get("rtol") != RTOL or fingerprint.get("atol") != ATOL:
        issues.append("fingerprint tolerances differ from qualification contract")
    return contract, issues


def matches_case(actual, expected):
    return isinstance(actual, dict) and all(
        key in actual and digest(actual[key]) == digest(value) for key, value in expected.items()
    )


def valid_errors(errors):
    return isinstance(errors, dict) and all(
        finite_number(errors.get(key)) and errors[key] >= 0 for key in ("max_abs_error", "rms_error")
    )


def valid_mask_witness(witness):
    witness = object_or_empty(witness)
    return (
        finite_number(witness.get("max_normalized_full_vs_causal"))
        and witness["max_normalized_full_vs_causal"] > MIN_MASK_WITNESS
        and witness.get("required_normalized_difference") == MIN_MASK_WITNESS
        and finite_number(witness.get("max_abs_full_vs_causal"))
        and witness["max_abs_full_vs_causal"] > 0
    )


def valid_core_limit(limit):
    limit = object_or_empty(limit)
    cube, vector = limit.get("cube_core_num"), limit.get("vector_core_num")
    return (
        nonnegative_integer(cube)
        and 0 < cube <= MAX_CUBE_CORES
        and nonnegative_integer(vector)
        and 0 < vector <= MAX_VECTOR_CORES
        and vector == 2 * cube
    )


def valid_metadata(plan, case, core_limit):
    plan = object_or_empty(plan)
    base = object_or_empty(plan.get("base"))
    fields = (
        "mBase",
        "sInner",
        "mFdBase",
        "numFd",
        "usedCoreNum",
        "usedFdVecNum",
        "maxS1",
        "maxS2",
        "actualQDims",
        "actualKVDims",
    )
    if not valid_core_limit(core_limit) or not all(nonnegative_integer(base.get(key)) for key in fields):
        return False
    counts, groups = plan.get("fd_s2_split_count_by_aic_metadata_slot"), plan.get("fd_group_s2_split_count")
    return (
        base["mBase"] == METADATA_M_BASE
        and base["mFdBase"] == METADATA_FD_M_BASE
        and 0 < base["sInner"] <= METADATA_M_BASE
        and base["usedCoreNum"] <= core_limit["cube_core_num"]
        and base["usedFdVecNum"] <= core_limit["vector_core_num"]
        and base["numFd"] <= min(MAX_FD_SLOTS, base["usedCoreNum"])
        and base["maxS1"] == max(case["q_lengths"])
        and base["maxS2"] == (max(case["kv_lengths"]) or 1024)
        and base["actualQDims"] == base["actualKVDims"] == len(case["q_lengths"])
        and isinstance(counts, list)
        and len(counts) == base["usedCoreNum"]
        and all(nonnegative_integer(count) and count <= MAX_FD_SLOTS for count in counts)
        and groups == counts[: base["numFd"]]
        and all(count > 0 for count in groups)
        and all(count == 0 for count in counts[base["numFd"] :])
        and plan.get("fd_present") is (base["numFd"] > 0)
    )


def zero_checks_match(rows, case):
    expected = {
        index
        for index, (qlen, kvlen) in enumerate(zip(case.get("q_lengths", []), case.get("kv_lengths", [])))
        if qlen and not kvlen
    }
    return (
        isinstance(rows, list)
        and len(rows) == len(expected)
        and all(isinstance(row, dict) and nonnegative_integer(row.get("row")) for row in rows)
        and {row["row"] for row in rows} == expected
        and all(row.get("nonzero_output_elements") == 0 for row in rows)
    )


def eager_summary(data, inventories):
    if data is None:
        return dict(complete=False)
    fingerprint = object_or_empty(data.get("fingerprint"))
    expected, issues = expected_cases(fingerprint, eager_cases())
    if fingerprint.get("phase") != "eager" or fingerprint.get("case_group") != "all":
        issues.append("eager phase/group differs from qualification contract")
    if not valid_core_limit(data.get("core_limit")):
        issues.append("missing or unsafe device core limits")
    attempts = data.get("case_attempts", [])
    if not isinstance(attempts, list):
        attempts, issues = [], issues + ["case_attempts is not a list"]
    if any(not isinstance(attempt, dict) for attempt in attempts):
        issues.append("malformed eager attempt records")
    attempts = [attempt for attempt in attempts if isinstance(attempt, dict)]
    known_ids = {case["id"] for case in expected}
    unexpected = sorted(
        {
            str(attempt.get("id"))
            for attempt in attempts
            if not isinstance(attempt.get("id"), str) or attempt["id"] not in known_ids
        }
    )
    if unexpected:
        issues.append("unexpected attempted case IDs")
    cases, passed_ids, fd_ids = [], set(), []
    for case in expected:
        history = [attempt for attempt in attempts if attempt.get("id") == case["id"]]
        successful = [
            attempt for attempt in history if attempt.get("state") == "passed" and matches_case(attempt, case)
        ]
        selected = successful[-1] if successful else (history[-1] if history else {})
        checks = dict(
            case_contract=bool(successful),
            final_checkpoint=selected.get("checkpoint") == "reference_and_canaries_passed",
            finite_error_metrics=valid_errors(selected.get("errors"))
            and matches_case(selected.get("errors"), dict(rtol=RTOL, atol=ATOL)),
            inputs_unchanged=selected.get("changed_read_only_inputs") == [],
            exact_zero_kv=zero_checks_match(selected.get("zero_kv_rows"), case),
            metadata_plan=valid_metadata(selected.get("metadata_plan"), case, data.get("core_limit")),
            metadata_hash=valid_hash(selected.get("metadata_sha256")),
        )
        if case.get("causal"):
            checks["mask_witness"] = valid_mask_witness(selected.get("mask_sensitivity"))
        passed = bool(successful) and all(checks.values())
        if passed:
            passed_ids.add(case["id"])
            if selected["metadata_plan"]["fd_present"]:
                fd_ids.append(case["id"])
        cases.append(
            dict(
                case=case,
                passed=passed,
                checks=checks,
                attempt_count=len(history),
                evidence=pick(
                    selected,
                    (
                        "checkpoint",
                        "errors",
                        "metadata_plan",
                        "metadata_nonzero",
                        "metadata_sha256",
                        "mask_sensitivity",
                        "zero_kv_rows",
                        "canaries",
                        "shapes",
                    ),
                ),
                unsuccessful_attempts=[
                    pick(attempt, ("state", "checkpoint", "error", "errors", "shapes"))
                    for attempt in history
                    if attempt.get("state") != "passed"
                ],
            )
        )
    complete = (
        finished(data)
        and data.get("selected_group_complete") is True
        and data.get("npu_work_executed") is True
        and not issues
        and bool(fd_ids)
        and len(passed_ids) == len(expected)
    )
    return dict(
        **status_fields(data),
        complete=complete,
        expected_case_count=len(expected),
        distinct_passed_case_count=len(passed_ids),
        attempt_count=len(attempts),
        missing_or_unqualified_ids=[case["id"] for case in expected if case["id"] not in passed_ids],
        unexpected_ids=unexpected,
        issues=issues,
        cases=cases,
        fd_present_ids=fd_ids,
        actual_fd_path_covered=bool(fd_ids),
        core_limit=data.get("core_limit"),
        fingerprint=compact_fingerprint(fingerprint, inventories),
    )


def valid_capture_addresses(addresses, copies):
    inputs, outputs, metadata = (
        addresses.get(key) for key in ("input_addresses", "output_pointers", "metadata_pointer")
    )
    if not (
        isinstance(inputs, list)
        and len(inputs) == copies
        and isinstance(outputs, list)
        and len(outputs) == copies
        and all(nonnegative_integer(pointer) and pointer > 0 for pointer in [metadata] + outputs)
    ):
        return False
    names = ("query_storage", "key", "value", "block_table", "q_lengths", "kv_lengths", "atten_mask")
    for storage in inputs:
        if not isinstance(storage, dict):
            return False
        for name in names:
            item = object_or_empty(storage.get(name))
            shape, stride, pointer = (item.get(key) for key in ("shape", "stride", "pointer"))
            if not (
                nonnegative_integer(pointer)
                and pointer > 0
                and isinstance(shape, list)
                and shape
                and isinstance(stride, list)
                and len(shape) == len(stride)
                and all(nonnegative_integer(value) for value in shape + stride)
            ):
                return False
    if copies > 1:
        if any(inputs[0][name]["pointer"] != inputs[1][name]["pointer"] for name in ("q_lengths", "kv_lengths")):
            return False
        if any(
            inputs[0][name]["pointer"] == inputs[1][name]["pointer"]
            for name in ("query_storage", "key", "value", "block_table")
        ):
            return False
    return len(set(outputs)) == copies and metadata not in outputs


def graph_summary(data, gate, inventories):
    if data is None:
        return dict(gate=gate, complete=False)
    fingerprint = object_or_empty(data.get("fingerprint"))
    expected, issues = expected_cases(fingerprint, graph_cases(gate))
    result = object_or_empty(data.get("result"))
    if not valid_core_limit(data.get("core_limit")):
        issues.append("missing or unsafe device core limits")
    records = result.get("results", data.get("results", []))
    if not isinstance(records, list):
        records, issues = [], issues + ["graph results is not a list"]
    if any(not isinstance(record, dict) for record in records):
        issues.append("malformed graph result records")
    if "results" in result and "results" in data and result["results"] != data["results"]:
        issues.append("final and progress result records disagree")
    copies = 2 if gate == "shared" else 1
    cases, passed_ids, fd_ids = [], set(), []
    known_ids = {case["id"] for case in expected}
    unexpected = [
        str(object_or_empty(record.get("case")).get("id"))
        for record in records
        if isinstance(record, dict)
        and (
            not isinstance(object_or_empty(record.get("case")).get("id"), str) or record["case"]["id"] not in known_ids
        )
    ]
    if unexpected:
        issues.append("unexpected graph step IDs")
    for case in expected:
        history = [
            record
            for record in records
            if isinstance(record, dict) and object_or_empty(record.get("case")).get("id") == case["id"]
        ]
        if len(history) > 1:
            issues.append(f"duplicate graph step: {case['id']}")
        record = history[-1] if history else {}
        errors = record.get("errors", [])
        checks = dict(
            record_passed=record.get("passed") is True,
            case_contract=matches_case(record.get("case", {}), case),
            addresses=record.get("stable_addresses") is True,
            input_storage_canaries=record.get("input_storage_exact") is True,
            zero_output_tail=record.get("zero_tail_exact") is True,
            error_metrics=isinstance(errors, list)
            and len(errors) == copies
            and all(valid_errors(error) for error in errors),
            exact_zero_kv=isinstance(errors, list)
            and len(errors) == copies
            and all(zero_checks_match(object_or_empty(error).get("zero_kv_outputs_exact"), case) for error in errors),
            mask_witness=not sum(case["q_lengths"])
            or isinstance(errors, list)
            and len(errors) == copies
            and all(valid_mask_witness(object_or_empty(error).get("mask_sensitivity")) for error in errors),
            metadata=valid_metadata(record.get("metadata_summary"), case, data.get("core_limit")),
            metadata_hash=valid_hash(record.get("metadata_sha256")),
        )
        if all(checks.values()):
            passed_ids.add(case["id"])
            if record["metadata_summary"]["fd_present"]:
                fd_ids.append(case["id"])
        cases.append(
            dict(
                case=case,
                passed=all(checks.values()),
                checks=checks,
                errors=errors,
                metadata_fd=record.get("metadata_summary"),
                metadata_sha256=record.get("metadata_sha256"),
                addresses_reference="capture_addresses; runner records per-step equality, not per-step raw pointers",
            )
        )
    captured = dict(
        metadata=result.get("captured_metadata_occurrences", data.get("metadata_occurrences")),
        attention=result.get("captured_attention_occurrences", data.get("attention_occurrences")),
    )
    addresses = pick(data, ("input_addresses", "metadata_pointer", "output_pointers"))
    metadata_hashes = {
        record.get("metadata_sha256")
        for record in records
        if isinstance(record, dict) and valid_hash(record.get("metadata_sha256"))
    }
    complete = (
        finished(data)
        and data.get("selected_graph_gate_qualified") is True
        and data.get("npu_work_executed") is True
        and fingerprint.get("gate") == gate
        and result.get("gate") == gate
        and result.get("state") == "passed"
        and result.get("graph_contains_planner") is True
        and captured == dict(metadata=1, attention=copies)
        and valid_capture_addresses(addresses, copies)
        and len(metadata_hashes) > 1
        and not issues
        and bool(fd_ids)
        and len(passed_ids) == len(expected)
    )
    return dict(
        **status_fields(data),
        gate=gate,
        complete=complete,
        expected_step_count=len(expected),
        distinct_passed_step_count=len(passed_ids),
        missing_or_unqualified_ids=[case["id"] for case in expected if case["id"] not in passed_ids],
        issues=issues,
        unexpected_ids=unexpected,
        steps=cases,
        capture_addresses=addresses,
        captured_occurrences=captured,
        distinct_metadata_hashes=len(metadata_hashes),
        fd_present_ids=fd_ids,
        actual_fd_path_covered=bool(fd_ids),
        core_limit=data.get("core_limit"),
        shared_input_independence_evidence=(
            "Distinct Q/K/V storage pointers are persisted. Distinct contents and per-copy FP32 reference checks "
            "are assertions in the executed runner, conditional on provenance checks; "
            "raw tensor hashes are not persisted."
            if gate == "shared"
            else None
        ),
        fingerprint=compact_fingerprint(fingerprint, inventories),
    )


def constant_summary(data):
    if data is None:
        return dict(complete=False)
    cases = data.get("cases", [])
    cases = [case for case in cases if isinstance(case, dict)] if isinstance(cases, list) else []
    shapes = {CONSTANT_CASES[0]: (8, 4, 1, 128), CONSTANT_CASES[1]: (4, 12, 2, 256)}
    passed = set()
    for case in cases:
        contract = shapes.get(case.get("name")) if isinstance(case.get("name"), str) else None
        if contract is None:
            continue
        query, heads, kv_heads, dimension = contract
        if (
            case.get("status") == "passed"
            and case.get("output_finite") is True
            and matches_case(
                case,
                dict(
                    query=query,
                    query_heads=heads,
                    kv_heads=kv_heads,
                    head_dim=dimension,
                    query_shape=[query, heads, dimension],
                    output_shape=[query, heads, dimension],
                    metadata_shape=[1024],
                    metadata_dtype="torch.int32",
                    rtol=ATOL,
                    atol=ATOL,
                ),
            )
            and finite_number(case.get("max_abs_error_from_one"))
            and 0 <= case["max_abs_error_from_one"] <= 2 * ATOL
            and valid_hash(case.get("metadata_sha256"))
        ):
            passed.add(case["name"])
    return dict(
        **status_fields(data),
        complete=finished(data)
        and data.get("smoke_passed") is True
        and data.get("npu_work_executed") is True
        and valid_core_limit(data.get("core_limit"))
        and len(cases) == len(CONSTANT_CASES)
        and set(CONSTANT_CASES).issubset(passed),
        expected_ids=list(CONSTANT_CASES),
        distinct_expected_passed_count=len(set(CONSTANT_CASES) & passed),
        missing_or_unqualified_ids=sorted(set(CONSTANT_CASES) - passed),
        cases=[
            pick(
                case,
                (
                    "name",
                    "status",
                    "query",
                    "query_heads",
                    "kv_heads",
                    "head_dim",
                    "metadata_shape",
                    "metadata_nonzero",
                    "metadata_sha256",
                    "output_shape",
                    "max_abs_error_from_one",
                    "output_finite",
                ),
            )
            for case in cases
        ],
        current_case=data.get("current_case"),
        opapi_library_sha256=data.get("opapi_library_sha256"),
    )


def foundation_summary(data):
    build, patch, deployed, staged, pipeline, extension = [data[name] or {} for name in ROOT_FILES[:6]]
    steps = pipeline.get("steps", [])
    steps = steps if isinstance(steps, list) else []
    successful_steps = {step.get("name") for step in steps if isinstance(step, dict) and finished(step)}
    libraries = staged.get("libraries", [])
    files = patch.get("files", [])
    tools = object_or_empty(deployed.get("tools"))
    schemas = object_or_empty(extension.get("schemas"))
    return dict(
        build=dict(**status_fields(build), complete=finished(build)),
        patch=dict(
            source_commit=patch.get("source_commit"),
            files=files,
            complete=bool(patch.get("source_commit"))
            and bool(files)
            and isinstance(files, list)
            and all(
                isinstance(file, dict)
                and valid_hash(file.get("before_sha256"))
                and valid_hash(file.get("after_sha256"))
                for file in files
            ),
        ),
        deployed_tools=dict(tools=tools, complete=bool(tools) and all(valid_hash(value) for value in tools.values())),
        staged_opp=dict(
            vendor_leaf=staged.get("vendor_leaf"),
            libraries=libraries,
            complete=isinstance(libraries, list)
            and bool(libraries)
            and all(
                isinstance(library, dict)
                and valid_hash(library.get("sha256"))
                and nonnegative_integer(library.get("size"))
                and library["size"] > 0
                for library in libraries
            )
            and bool(staged.get("opapi_library"))
            and bool(staged.get("aicpu_library")),
        ),
        pipeline=dict(
            **status_fields(pipeline),
            steps=[pick(step, ("name", "state", "returncode", "error")) for step in steps if isinstance(step, dict)],
            missing_or_unqualified_steps=sorted(set(PIPELINE_STEPS) - successful_steps),
            complete=finished(pipeline) and set(PIPELINE_STEPS).issubset(successful_steps),
        ),
        extension=dict(
            **status_fields(extension),
            library=extension.get("extension_library"),
            library_sha256=extension.get("library_sha256"),
            complete=finished(extension)
            and valid_hash(extension.get("library_sha256"))
            and bool(extension.get("extension_library"))
            and all(
                object_or_empty(schemas.get(name)).get("PrivateUse1") is True
                and object_or_empty(schemas.get(name)).get("Meta") is True
                for name in (
                    "npu_fused_infer_attention_score_v2_sink",
                    "_npu_fused_infer_attention_score_v2_sink_metadata",
                )
            ),
        ),
    )


def valid_inventory(inventory):
    return (
        isinstance(inventory, dict)
        and bool(inventory)
        and all(
            isinstance(name, str)
            and name
            and "\\" not in name
            and not PurePosixPath(name).is_absolute()
            and ".." not in PurePosixPath(name).parts
            and valid_hash(value)
            for name, value in inventory.items()
        )
    )


def final_inventory_summary(payloads):
    final = object_or_empty(payloads["final-artifact-inventory.json"])
    patch = object_or_empty(payloads["d256-patch-manifest.json"])
    staged = object_or_empty(payloads["staged-opp-manifest.json"])
    extension = object_or_empty(payloads["extension-status.json"])
    eager_fp = object_or_empty(object_or_empty(payloads["eager"]).get("fingerprint"))
    inventory = object_or_empty(eager_fp.get("opp_artifacts"))
    files = final.get("files")
    files = files if isinstance(files, list) else []
    records, duplicate_paths = {}, []
    for item in files:
        item = object_or_empty(item)
        name = item.get("path")
        if isinstance(name, str):
            if name in records:
                duplicate_paths.append(name)
            records[name] = item
    sources = final.get("source_files")
    sources = sources if isinstance(sources, list) else []
    source_hashes = {}
    for item in sources:
        item = object_or_empty(item)
        name = item.get("path", item.get("file"))
        if isinstance(name, str):
            source_hashes[name.removeprefix("source/")] = item.get("sha256")
    patch_files = patch.get("files")
    patch_files = patch_files if isinstance(patch_files, list) else []
    checks = dict(
        source_matches_patch_manifest_reported=final.get("source_matches_patch_manifest") is True,
        source_after_hashes=len(patch_files) == 3
        and len(sources) == len(source_hashes) == 3
        and all(
            isinstance(item, dict)
            and isinstance(item.get("file"), str)
            and valid_hash(item.get("after_sha256"))
            and source_hashes.get(item["file"]) == item["after_sha256"]
            for item in patch_files
        ),
        unique_valid_final_records=bool(records)
        and len(records) == len(files)
        and not duplicate_paths
        and all(
            valid_hash(item.get("sha256")) and nonnegative_integer(item.get("bytes")) and item["bytes"] > 0
            for item in records.values()
        ),
    )
    linked = []
    try:
        remote_root = PurePosixPath(staged["prefix"]).parent
        vendor_relative = PurePosixPath(staged["vendor_leaf"]).relative_to(remote_root)
        extension_relative = PurePosixPath(extension["extension_library"]).relative_to(remote_root).as_posix()
        checks["extension_after_hash"] = (
            valid_hash(extension.get("library_sha256"))
            and records.get(extension_relative, {}).get("sha256") == extension["library_sha256"]
        )
        binary_paths = {name for name in inventory if name.endswith((".so", ".o"))}
        for name in sorted(binary_paths):
            path = (vendor_relative / name).as_posix()
            linked.append(
                dict(path=path, sha256=inventory[name], matches=records.get(path, {}).get("sha256") == inventory[name])
            )
        checks["execution_binary_after_hashes"] = bool(binary_paths) and all(item["matches"] for item in linked)
    except (KeyError, TypeError, ValueError) as error:
        checks["execution_binary_after_hashes"] = False
        linked.append(dict(error=str(error)))
    return dict(
        complete=all(checks.values()),
        checks=checks,
        linked_execution_binaries=linked,
        source_after_hashes=source_hashes,
        limitation="Recorded source and artifact hashes are bound; the native build is not independently reproduced.",
    )


def provenance_summary(payloads, artifacts):
    """Bind recorded execution to deployed scripts and one staged binary inventory."""
    deployed = object_or_empty(payloads["deployed-tools.json"])
    tools = object_or_empty(deployed.get("tools"))
    local_sources = {}
    for name in EXECUTION_TOOLS:
        path = Path(__file__).resolve().parent / name
        try:
            local_sources[name] = dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        except OSError as error:
            local_sources[name] = dict(path=str(path), error=str(error))
    expected_scripts = {name: item.get("sha256") for name, item in local_sources.items()}
    eager = object_or_empty(payloads["eager"])
    constant = object_or_empty(payloads["constant"])
    fingerprints = {
        name: object_or_empty(object_or_empty(payloads[name]).get("fingerprint")) for name in ("eager",) + GATES
    }
    eager_fp = fingerprints["eager"]
    staged = object_or_empty(payloads["staged-opp-manifest.json"])
    extension = object_or_empty(payloads["extension-status.json"])
    vendor = staged.get("vendor_leaf")
    inventory = eager_fp.get("opp_artifacts")
    extension_hash = extension.get("library_sha256")
    checks = dict(
        deployed_matches_local_execution_sources=all(
            valid_hash(value) and tools.get(name) == value for name, value in expected_scripts.items()
        ),
        eager_script_bindings=eager_fp.get("script_sha256") == expected_scripts[EXECUTION_TOOLS[1]]
        and eager_fp.get("helper_sha256") == expected_scripts[EXECUTION_TOOLS[0]],
        graph_script_bindings=all(fingerprints[gate].get("scripts") == expected_scripts for gate in GATES),
        constant_script_binding=constant.get("script_sha256") == expected_scripts[EXECUTION_TOOLS[0]],
        constant_extension_status_binding=constant.get("extension_status_sha256")
        == artifacts["extension-status.json"].get("sha256")
        and valid_hash(constant.get("extension_status_sha256")),
        extension_binary_bindings=valid_hash(extension_hash)
        and all(fp.get("extension_sha256") == extension_hash for fp in fingerprints.values())
        and constant.get("extension_library") == extension.get("extension_library")
        and bool(extension.get("extension_library")),
        opp_roots=bool(vendor)
        and all(fp.get("opp_root") == vendor for fp in fingerprints.values())
        and constant.get("opp_vendor_leaf") == vendor,
        complete_opp_inventories=valid_inventory(inventory)
        and all(
            valid_inventory(fp.get("opp_artifacts")) and fp["opp_artifacts"] == inventory
            for fp in fingerprints.values()
        ),
        core_limits=valid_core_limit(eager.get("core_limit"))
        and all(
            object_or_empty(payloads[name]).get("core_limit") == eager.get("core_limit")
            for name in ("constant",) + GATES
        ),
        seeds=nonnegative_integer(eager_fp.get("seed"))
        and all(fp.get("seed") == eager_fp.get("seed") for fp in fingerprints.values()),
    )
    libraries = staged.get("libraries")
    linked_libraries = []
    try:
        if not isinstance(vendor, str) or not isinstance(staged.get("prefix"), str):
            raise ValueError("staged vendor/prefix missing")
        if not isinstance(libraries, list) or not libraries or not valid_inventory(inventory):
            raise ValueError("staged libraries or execution OPP inventory missing")
        vendor_path = PurePosixPath(vendor)
        for library in libraries:
            if not isinstance(library, dict) or not isinstance(library.get("path"), str):
                raise ValueError("invalid staged library record")
            path = PurePosixPath(staged["prefix"]).parent / library["path"]
            relative = path.relative_to(vendor_path).as_posix()
            linked_libraries.append(
                dict(
                    path=str(path),
                    inventory_path=relative,
                    sha256=library.get("sha256"),
                    matches=valid_hash(library.get("sha256")) and inventory.get(relative) == library["sha256"],
                )
            )
        opapi_relative = PurePosixPath(staged["opapi_library"]).relative_to(vendor_path).as_posix()
        aicpu_relative = PurePosixPath(staged["aicpu_library"]).relative_to(vendor_path).as_posix()
        opapi_hash = inventory.get(opapi_relative)
        checks["staged_library_bindings"] = all(item["matches"] for item in linked_libraries)
        checks["opapi_binary_bindings"] = (
            valid_hash(opapi_hash)
            and eager_fp.get("opapi_sha256") == opapi_hash
            and constant.get("opapi_library_sha256") == opapi_hash
            and constant.get("opapi_library") == staged["opapi_library"]
        )
        checks["staged_required_libraries"] = all(
            any(item["inventory_path"] == relative and item["matches"] for item in linked_libraries)
            for relative in (opapi_relative, aicpu_relative)
        )
    except (KeyError, TypeError, ValueError) as error:
        checks["staged_library_bindings"] = False
        linked_libraries.append(dict(error=str(error)))
    return dict(
        complete=all(checks.values()),
        checks=checks,
        local_execution_sources=local_sources,
        linked_staged_libraries=linked_libraries,
        shared_distinct_contents_evidence="Source-backed assertion in the hash-bound executed graph runner; "
        "independent per-copy FP32 checks and storage addresses are recorded, raw tensor hashes are not persisted.",
        staging_script_evidence="stage-opp.py hash is declared in deployed-tools; its command is in pipeline-status. "
        "the staging process does not emit an independent self-hash. Binary linkage uses staged library hashes.",
    )


def summarize(evidence):
    """Summarize the fixed evidence contract while retaining raw-file hashes."""
    payloads, artifacts = {}, {}
    for name in ROOT_FILES:
        payloads[name], artifacts[name] = load_artifact(evidence, ["root/" + name, name])
    paths = dict(
        eager="eager-v1/stage2-status.json",
        constant="constant-v1/smoke-status.json",
        **{gate: f"graph-{gate}-v1/stage2-graph-status.json" for gate in GATES},
    )
    for name, path in paths.items():
        payloads[name], artifacts[name] = load_artifact(evidence, [path])
    inventories = {}
    foundations = foundation_summary(payloads)
    eager = eager_summary(payloads["eager"], inventories)
    constant = constant_summary(payloads["constant"])
    graphs = {gate: graph_summary(payloads[gate], gate, inventories) for gate in GATES}
    provenance = provenance_summary(payloads, artifacts)
    final_inventory = final_inventory_summary(payloads)
    components = list(foundations.values()) + [eager, constant, provenance, final_inventory] + list(graphs.values())
    complete = all(component["complete"] for component in components) and all(
        artifact["state"] == "present" for artifact in artifacts.values()
    )
    failed = any(component.get("state") == "failed" for component in components) or any(
        artifact["state"] == "invalid" for artifact in artifacts.values()
    )
    return dict(
        schema_version=2,
        state="completed" if complete else "failed" if failed else "partial",
        all_selected_stage2_gates_passed=complete,
        evidence_root=str(evidence),
        artifacts=artifacts,
        foundations=foundations,
        provenance=provenance,
        final_inventory=final_inventory,
        eager=eager,
        constant=constant,
        graphs=graphs,
        opp_inventories=inventories,
        performance_measured=False,
        full_model_qualified=False,
        incremental_planner_qualified=False,
        scope="Isolated operator/eager/graph correctness only; performance and model remain unqualified",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evidence = args.evidence.resolve(strict=True)
    if not evidence.is_dir():
        parser.error("--evidence must be a directory")
    output = args.output.resolve()
    protected = [evidence / prefix / name for name in ROOT_FILES for prefix in ("root", ".")]
    protected += [evidence / "eager-v1/stage2-status.json", evidence / "constant-v1/smoke-status.json"]
    protected += [evidence / f"graph-{gate}-v1/stage2-graph-status.json" for gate in GATES]
    if output in [path.resolve() for path in protected]:
        parser.error("--output cannot overwrite an original evidence file")
    report = summarize(evidence)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            dict(
                output=str(output),
                state=report["state"],
                all_selected_stage2_gates_passed=report["all_selected_stage2_gates_passed"],
            )
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
