# SPDX-License-Identifier: Apache-2.0
"""Qualify isolated FIA sink eager semantics; no graphs, model or performance.

Run ``--cpu-reference-check`` first. NPU execution requires the separately built
isolated OPP/extension and an explicit ``--phase eager --opp-root ...``. Default
``--max-cases 1`` runs only the first B1 noncausal gate. Resume skips passed cases
only when source, binary, selected cases and tolerances still match.

The wrapper allocates its own output: output allocation guard regions cannot be
provided through this ABI. This checks output shape/finiteness/FP32 reference,
and exact immutability of input storage including unused KV pages/tokens and
query guard rows. It does not establish graph safety or general model accuracy.
Every causal fixture has a forbidden-tail V witness; omitting its mask must
exceed the actual tolerance by more than 10 times. Q97 is an internal kernel
loop boundary gate, not a claim of support for a model or mixed requests.
"""

import argparse
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

import aclgraph_fia_sink_stage1_smoke as isolated

QUERY_HEADS = 12
KV_HEADS = 2
HEAD_DIM = 256
PAGE_SIZE = 128
MASK_SIZE = 2048
QUERY_GUARD_ROWS = 2
KV_UNUSED_EXTRA_PAGES = 4
KV_BOUNDARIES = (127, 128, 129, 511, 512, 513, 1025, 4097)
MAX_CUBE_CORES = 20
MAX_VECTOR_CORES = 40
MAX_FD_SLOTS = 24
METADATA_BASE_OFFSET = 576
METADATA_AIC_STRIDE = 10
METADATA_FD_S2_COUNT_INDEX = 6
METADATA_BASE_FIELDS = (
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
RTOL = 3e-2
ATOL = 3e-3
MIN_MASK_WITNESS_NORMALIZED_DIFFERENCE = 10.0
MASK_WITNESS_VALUE = 64.0
QUERY_CANARY = 3.125
KEY_CANARY = 13.25
VALUE_CANARY = -17.5


def cases_for(group):
    """Keep unsupported empty cases separate from positive-length qualification."""
    positive = []
    for causal in (False, True):
        for kvlen in KV_BOUNDARIES:
            positive.append(
                dict(
                    id=f"b1-q4-kv{kvlen}-{'causal' if causal else 'full'}",
                    q_lengths=[4],
                    kv_lengths=[kvlen],
                    causal=causal,
                    group="positive",
                )
            )
        positive.append(
            dict(
                id=f"ragged-{'causal' if causal else 'full'}",
                q_lengths=[4, 1, 3],
                kv_lengths=[129, 511, 1025],
                causal=causal,
                group="positive",
            )
        )
    positive.extend(
        [
            dict(
                id=f"b1-q4-kv{kvlen}-causal-short",
                q_lengths=[4],
                kv_lengths=[kvlen],
                causal=True,
                group="positive",
                purpose="short-context right-down causal semantic gate",
            )
            for kvlen in (4, 7)
        ]
    )
    positive.append(
        dict(
            id="b1-q97-kv513-full-internal-loop-boundary",
            q_lengths=[97],
            kv_lengths=[513],
            causal=False,
            group="positive",
            purpose="internal M=582 multi-loop boundary only; no model/mixed-request support claim",
        )
    )
    groups = {
        "positive": positive,
        "zero-kv": [
            dict(id="zero-kv-only", q_lengths=[4], kv_lengths=[0], causal=False, group="zero-kv"),
            dict(id="zero-kv-mixed", q_lengths=[4, 4, 4], kv_lengths=[0, 129, 0], causal=False, group="zero-kv"),
        ],
        "zero-q": [
            dict(id="zero-q-only", q_lengths=[0], kv_lengths=[129], causal=False, group="zero-q"),
            dict(id="zero-q-mixed", q_lengths=[0, 4, 0], kv_lengths=[129, 511, 0], causal=False, group="zero-q"),
        ],
    }
    return groups[group] if group != "all" else [case for cases in groups.values() for case in cases]


def cpu_reference(torch, query, key, value, pages, q_lengths, kv_lengths, causal, kv_heads=KV_HEADS):
    """FP32 reference from exact logical lengths, independent of native metadata.

    TND query lengths are interpreted here as per-request lengths; the caller
    separately builds cumulative native Q lengths. Paged KV lengths are local.
    Right-down causal visibility is key_index <= KV_length - Q_length + q_index.
    Empty KV and fully masked query rows produce zero; empty Q contributes no rows.
    """
    if query.device.type != "cpu" or key.device.type != "cpu" or value.device.type != "cpu":
        raise ValueError("reference inputs must be CPU tensors")
    dimension, query_heads = query.shape[-1], query.shape[-2]
    if query_heads % kv_heads or sum(q_lengths) != query.shape[0] or len(q_lengths) != len(kv_lengths):
        raise ValueError("invalid reference head/length contract")
    result = torch.zeros(query.shape, dtype=torch.float32)
    offset = 0
    for row, (qlen, kvlen) in enumerate(zip(q_lengths, kv_lengths)):
        if qlen < 0 or kvlen < 0:
            raise ValueError("lengths must be nonnegative")
        if qlen and kvlen:
            block_count = (kvlen + key.shape[1] - 1) // key.shape[1]
            physical_ids = pages[row, :block_count].long()
            logical_key = key[physical_ids].reshape(-1, kv_heads, dimension)[:kvlen].float()
            logical_value = value[physical_ids].reshape(-1, kv_heads, dimension)[:kvlen].float()
            key_heads = logical_key.repeat_interleave(query_heads // kv_heads, dim=1)
            value_heads = logical_value.repeat_interleave(query_heads // kv_heads, dim=1)
            scores = torch.einsum("qhd,khd->hqk", query[offset : offset + qlen].float(), key_heads) * dimension**-0.5
            visible = torch.ones((qlen, kvlen), dtype=torch.bool)
            if causal:
                visible = torch.arange(kvlen)[None, :] <= kvlen - qlen + torch.arange(qlen)[:, None]
                scores.masked_fill_(~visible.unsqueeze(0), -torch.inf)
            nonempty = visible.any(dim=1)
            probabilities = torch.zeros_like(scores)
            probabilities[:, nonempty] = torch.softmax(scores[:, nonempty], dim=-1)
            result[offset : offset + qlen] = torch.einsum("hqk,khd->qhd", probabilities, value_heads)
        offset += qlen
    return result


def make_inputs(torch, case, seed):
    """All unused physical pages, unused table slots and last-page tails are poisoned."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    q_lengths, kv_lengths = case["q_lengths"], case["kv_lengths"]
    blocks = [(length + PAGE_SIZE - 1) // PAGE_SIZE for length in kv_lengths]
    physical_count = sum(blocks) + KV_UNUSED_EXTRA_PAGES
    key = torch.full((physical_count, PAGE_SIZE, KV_HEADS * HEAD_DIM), KEY_CANARY, dtype=torch.bfloat16)
    value = torch.full_like(key, VALUE_CANARY)
    candidates = torch.randperm(physical_count - 2, generator=generator) + 1
    used_ids = candidates[: sum(blocks)].tolist()
    unused_id = int(candidates[sum(blocks)])
    table = torch.full((len(q_lengths), max(blocks, default=0) + 1), unused_id, dtype=torch.int32)
    used = []
    cursor = 0
    for row, (count, kvlen) in enumerate(zip(blocks, kv_lengths)):
        for block in range(count):
            page = used_ids[cursor]
            cursor += 1
            table[row, block] = page
            valid = min(PAGE_SIZE, kvlen - block * PAGE_SIZE)
            key[page, :valid] = (torch.randn((valid, KV_HEADS * HEAD_DIM), generator=generator) * 0.5).bfloat16()
            value[page, :valid] = (torch.randn((valid, KV_HEADS * HEAD_DIM), generator=generator) * 0.5).bfloat16()
            used.append(page)
        if case["causal"]:
            # First query cannot see this tail. Reading it despite the mask must
            # change output clearly even when Q4 is a tiny fraction of long KV.
            witness_tokens = min(max(q_lengths[row] - 1, 0), kvlen)
            for token in range(kvlen - witness_tokens, kvlen):
                value[int(table[row, token // PAGE_SIZE]), token % PAGE_SIZE] = MASK_WITNESS_VALUE
    total_q = sum(q_lengths)
    query_storage = torch.full(
        (total_q + 2 * QUERY_GUARD_ROWS, QUERY_HEADS, HEAD_DIM), QUERY_CANARY, dtype=torch.bfloat16
    )
    query = query_storage[QUERY_GUARD_ROWS : QUERY_GUARD_ROWS + total_q]
    query.copy_((torch.randn(query.shape, generator=generator) * 0.5).bfloat16())
    cumulative_q = torch.tensor(q_lengths, dtype=torch.int64).cumsum(0)
    inputs = dict(
        query_storage=query_storage,
        key=key,
        value=value,
        block_table=table,
        q_lengths=cumulative_q,
        kv_lengths=torch.tensor(kv_lengths, dtype=torch.int64),
    )
    if case["causal"]:
        # Pinned native consistency check requires compressed [2048,2048] mask.
        # True means disallowed, matching the native right-down causal mask ABI.
        inputs["atten_mask"] = torch.triu(torch.ones((MASK_SIZE, MASK_SIZE), dtype=torch.bool), diagonal=1)
    return inputs, dict(
        used_physical_pages=used,
        unused_physical_pages=sorted(set(range(physical_count)) - set(used)),
        query_guard_rows=QUERY_GUARD_ROWS,
        unused_table_page=unused_id,
        key_canary=KEY_CANARY,
        value_canary=VALUE_CANARY,
        forbidden_tail_value=MASK_WITNESS_VALUE if case["causal"] else None,
    )


def mask_witness(torch, expected, full):
    """Require an omitted causal mask to fail the actual comparison tolerance."""
    difference = (full - expected).abs()
    normalized = difference / (ATOL + RTOL * expected.abs())
    metrics = dict(
        max_abs_full_vs_causal=float(difference.max()),
        max_normalized_full_vs_causal=float(normalized.max()),
        required_normalized_difference=MIN_MASK_WITNESS_NORMALIZED_DIFFERENCE,
        forbidden_tail_value=MASK_WITNESS_VALUE,
    )
    if metrics["max_normalized_full_vs_causal"] <= MIN_MASK_WITNESS_NORMALIZED_DIFFERENCE:
        raise AssertionError(f"causal test would not clearly detect an ignored mask: {metrics}")
    return metrics


def metadata_summary(metadata_host, core_limit):
    """Decode the pinned uint32 metadata ABI and reject unsupported resource limits."""
    words = [int(value) & 0xFFFFFFFF for value in metadata_host.tolist()]
    base = dict(
        zip(METADATA_BASE_FIELDS, words[METADATA_BASE_OFFSET : METADATA_BASE_OFFSET + len(METADATA_BASE_FIELDS)])
    )
    if not (0 <= base["usedCoreNum"] <= min(core_limit["cube_core_num"], MAX_CUBE_CORES)):
        raise AssertionError(f"metadata compute core count exceeds qualified limit: {base}")
    if base["usedFdVecNum"] > min(core_limit["vector_core_num"], MAX_VECTOR_CORES) or base["numFd"] > min(
        MAX_FD_SLOTS, base["usedCoreNum"]
    ):
        raise AssertionError(f"metadata FD resource count exceeds qualified limit: {base}")
    counts = [words[core * METADATA_AIC_STRIDE + METADATA_FD_S2_COUNT_INDEX] for core in range(base["usedCoreNum"])]
    if max(counts, default=0) > MAX_FD_SLOTS:
        raise AssertionError(f"metadata FD partial count exceeds {MAX_FD_SLOTS}: {counts}")
    return dict(
        base=base,
        fd_s2_split_count_by_aic_metadata_slot=counts,
        fd_group_s2_split_count=counts[: base["numFd"]],
        fd_present=base["numFd"] > 0,
        fd_count_semantics="FD group arrays are serialized in AIC slots; a slot is not ownership of a core",
    )


def validate_core_limit(cores):
    cube, vector = int(cores["cube_core_num"]), int(cores["vector_core_num"])
    if not (0 < cube <= MAX_CUBE_CORES and 0 < vector <= MAX_VECTOR_CORES and vector == 2 * cube):
        raise RuntimeError(f"qualified limits require C<=20, V<=40 and C:V=1:2: {cores}")


def exact_zero_kv_outputs(torch, actual, case):
    """The native zero-KV path explicitly clears outputs; no numeric tolerance."""
    offset, checked = 0, []
    for row, (qlen, kvlen) in enumerate(zip(case["q_lengths"], case["kv_lengths"])):
        if qlen and kvlen == 0:
            output_slice = actual[offset : offset + qlen]
            count = int(torch.count_nonzero(output_slice))
            checked.append(dict(row=row, q_length=qlen, nonzero_output_elements=count))
            if not torch.equal(output_slice, torch.zeros_like(output_slice)):
                raise AssertionError(f"zero-KV output is not exactly zero: {checked[-1]}")
        offset += qlen
    return checked


def cpu_reference_check():
    """Small hand-computed tests; this exercises no NPU or custom operator."""
    import torch

    key = torch.zeros((3, 4, 1), dtype=torch.bfloat16)
    value = torch.full_like(key, 99)
    value[2, :, 0] = torch.tensor([1, 2, 3, 4], dtype=torch.bfloat16)
    value[0, 0, 0] = 5
    pages = torch.tensor([[2, 0], [1, 1], [1, 1]], dtype=torch.int32)
    query = torch.zeros((4, 2, 1), dtype=torch.bfloat16)
    full = cpu_reference(torch, query, key, value, pages, [2, 0, 2], [5, 4, 0], False, kv_heads=1)
    causal = cpu_reference(torch, query, key, value, pages, [2, 0, 2], [5, 4, 0], True, kv_heads=1)
    torch.testing.assert_close(full[:2], torch.full((2, 2, 1), 3.0))
    torch.testing.assert_close(causal[:2, 0, 0], torch.tensor([2.5, 3.0]))
    assert torch.equal(full[2:], torch.zeros_like(full[2:]))
    assert torch.equal(causal[2:], torch.zeros_like(causal[2:]))
    empty = cpu_reference(torch, query[:0], key, value, pages[:1], [0], [5], False, kv_heads=1)
    assert empty.shape == (0, 2, 1)
    weighted_query = torch.ones((1, 2, 1), dtype=torch.bfloat16)
    weighted_key = torch.tensor([[[0], [1]]], dtype=torch.bfloat16)
    weighted_value = torch.tensor([[[2], [6]]], dtype=torch.bfloat16)
    weighted = cpu_reference(
        torch,
        weighted_query,
        weighted_key,
        weighted_value,
        torch.tensor([[0]], dtype=torch.int32),
        [1],
        [2],
        False,
        kv_heads=1,
    )
    torch.testing.assert_close(weighted, torch.full((1, 2, 1), (2 + 6 * math.e) / (1 + math.e)))
    masked = cpu_reference(torch, query[:3], key, value, pages[:1], [3], [2], True, kv_heads=1)
    torch.testing.assert_close(masked[:, 0, 0], torch.tensor([0.0, 1.0, 1.5]))
    first = cases_for("positive")[0]
    inputs, canaries = make_inputs(torch, first, seed=123)
    q = inputs["query_storage"][QUERY_GUARD_ROWS:-QUERY_GUARD_ROWS]
    reference = cpu_reference(
        torch, q, inputs["key"], inputs["value"], inputs["block_table"], first["q_lengths"], first["kv_lengths"], False
    )
    # Replacing only unused physical pages/tails must not affect the reference.
    changed = inputs["value"].clone()
    changed[canaries["unused_physical_pages"]] = 71
    last_page = int(inputs["block_table"][0, 0])
    changed[last_page, 127:] = -71
    torch.testing.assert_close(
        reference,
        cpu_reference(
            torch, q, inputs["key"], changed, inputs["block_table"], first["q_lengths"], first["kv_lengths"], False
        ),
        rtol=0,
        atol=0,
    )
    causal_witnesses = {}
    for index, case in enumerate(cases_for("all")):
        if not case["causal"]:
            continue
        fixture, _ = make_inputs(torch, case, seed=1703 + index)
        query_fixture = fixture["query_storage"][QUERY_GUARD_ROWS:-QUERY_GUARD_ROWS]
        arguments = (
            torch,
            query_fixture,
            fixture["key"],
            fixture["value"],
            fixture["block_table"],
            case["q_lengths"],
            case["kv_lengths"],
        )
        causal_expected = cpu_reference(*arguments, True)
        full_expected = cpu_reference(*arguments, False)
        causal_witnesses[case["id"]] = mask_witness(torch, causal_expected, full_expected)
    validate_core_limit(dict(cube_core_num=20, vector_core_num=40))
    synthetic_metadata = torch.zeros(1024, dtype=torch.int32)
    synthetic_metadata[METADATA_BASE_OFFSET : METADATA_BASE_OFFSET + len(METADATA_BASE_FIELDS)] = torch.tensor(
        [512, 512, 32, 2, 4, 8, 4, 4097, 1, 1], dtype=torch.int32
    )
    synthetic_metadata[METADATA_FD_S2_COUNT_INDEX] = 3
    synthetic_metadata[METADATA_AIC_STRIDE + METADATA_FD_S2_COUNT_INDEX] = 4
    decoded = metadata_summary(synthetic_metadata, dict(cube_core_num=20, vector_core_num=40))
    assert decoded["base"]["numFd"] == 2 and decoded["fd_group_s2_split_count"] == [3, 4]
    assert (
        exact_zero_kv_outputs(torch, torch.zeros((4, QUERY_HEADS, HEAD_DIM)), dict(q_lengths=[4], kv_lengths=[0]))[0][
            "nonzero_output_elements"
        ]
        == 0
    )
    try:
        exact_zero_kv_outputs(
            torch, torch.full((4, QUERY_HEADS, HEAD_DIM), ATOL / 2), dict(q_lengths=[4], kv_lengths=[0])
        )
    except AssertionError:
        pass
    else:
        raise AssertionError("sub-atol zero-KV residue was accepted")
    return dict(
        cpu_only=True,
        npu_work_executed=False,
        hand_computed_full_and_right_down_causal=True,
        ragged_empty_q_and_empty_kv=True,
        unused_pages_and_tail_excluded=True,
        all_causal_cases_have_strong_ignored_mask_witness=causal_witnesses,
        metadata_fields_and_fd_counts_decoded=True,
        sub_atol_zero_kv_residue_rejected=True,
        performance_measured=False,
        native_operator_qualified=False,
    )


def eager_case(torch, case, core_limit, seed, progress):
    """Run one eager gate, persisting evidence before each native call/check."""
    inputs, canaries = make_inputs(torch, case, seed)
    host_query = inputs["query_storage"][QUERY_GUARD_ROWS:-QUERY_GUARD_ROWS]
    expected = cpu_reference(
        torch,
        host_query,
        inputs["key"],
        inputs["value"],
        inputs["block_table"],
        case["q_lengths"],
        case["kv_lengths"],
        case["causal"],
    )
    witness = None
    if case["causal"]:
        full_reference = cpu_reference(
            torch,
            host_query,
            inputs["key"],
            inputs["value"],
            inputs["block_table"],
            case["q_lengths"],
            case["kv_lengths"],
            False,
        )
        witness = mask_witness(torch, expected, full_reference)
    progress("reference_ready", mask_sensitivity=witness)
    device = torch.device("npu:0")
    tensors = {name: tensor.to(device) for name, tensor in inputs.items()}
    query = tensors["query_storage"][QUERY_GUARD_ROWS:-QUERY_GUARD_ROWS]
    progress(
        "inputs_ready",
        shapes={name: list(tensor.shape) for name, tensor in tensors.items()},
        canaries=canaries,
        length_semantics="Q cumulative; paged KV per request",
        output_allocation_guards="unavailable: native wrapper owns allocation",
    )
    operations = getattr(torch.ops, isolated.NAMESPACE)
    sparse_mode = 3 if case["causal"] else 0
    progress("metadata_started")
    metadata = getattr(operations, isolated.META_NAME)(
        QUERY_HEADS,
        KV_HEADS,
        HEAD_DIM,
        HEAD_DIM,
        actual_seq_lengths=tensors["q_lengths"],
        actual_seq_lengths_kv=tensors["kv_lengths"],
        batch_size=len(case["q_lengths"]),
        sparse_mode=sparse_mode,
        input_layout="TND",
        input_layout_kv="BnBsH",
        block_size=PAGE_SIZE,
        aic_core_num=core_limit["cube_core_num"],
        aiv_core_num=core_limit["vector_core_num"],
        batch_invariant=False,
    )
    torch.npu.synchronize()
    if metadata.dtype != torch.int32 or tuple(metadata.shape) != (1024,) or metadata.device != device:
        raise AssertionError("metadata must be device int32[1024]")
    before_metadata = metadata.cpu().clone()
    metadata_nonzero = int((before_metadata != 0).sum())
    if sum(case["q_lengths"]) and max(case["kv_lengths"]) and not metadata_nonzero:
        raise AssertionError("positive-length metadata is entirely zero")
    progress(
        "metadata_completed",
        metadata_nonzero=metadata_nonzero,
        metadata_sha256=hashlib.sha256(before_metadata.numpy().tobytes()).hexdigest(),
        metadata_plan=metadata_summary(before_metadata, core_limit),
    )
    progress("attention_started")
    output, lse = getattr(operations, isolated.ATTENTION_NAME)(
        query,
        tensors["key"],
        tensors["value"],
        actual_seq_qlen=tensors["q_lengths"],
        actual_seq_kvlen=tensors["kv_lengths"],
        block_table=tensors["block_table"],
        meta_data=metadata,
        atten_mask=tensors.get("atten_mask"),
        num_query_heads=QUERY_HEADS,
        num_key_value_heads=KV_HEADS,
        softmax_scale=HEAD_DIM**-0.5,
        input_layout="TND",
        sparse_mode=sparse_mode,
        block_size=PAGE_SIZE,
        return_softmax_lse=False,
    )
    torch.npu.synchronize()
    progress("attention_completed", output_shape=list(output.shape), output_dtype=str(output.dtype))
    if tuple(output.shape) != tuple(expected.shape) or output.dtype != torch.bfloat16 or output.device != device:
        raise AssertionError("output shape/dtype/device mismatch")
    actual = output.float().cpu()
    if not bool(torch.isfinite(actual).all()) or lse.numel() != 0:
        raise AssertionError("output nonfinite or unexpected LSE")
    zero_rows = exact_zero_kv_outputs(torch, actual, case)
    progress("zero_kv_exact_check", zero_kv_rows=zero_rows)
    absolute = (actual - expected).abs()
    metrics = dict(
        max_abs_error=float(absolute.max()) if absolute.numel() else 0.0,
        rms_error=float(absolute.square().mean().sqrt()) if absolute.numel() else 0.0,
        rtol=RTOL,
        atol=ATOL,
    )
    progress("reference_comparison", errors=metrics)
    changed_inputs = []
    for name, before in inputs.items():
        if not torch.equal(tensors[name].cpu(), before):
            changed_inputs.append(name)
    if not torch.equal(metadata.cpu(), before_metadata):
        changed_inputs.append("metadata")
    progress("read_only_canaries_checked", changed_read_only_inputs=changed_inputs, errors=metrics)
    if changed_inputs:
        raise AssertionError(f"read-only tensor storage changed, including guard/canary regions: {changed_inputs}")
    torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL)
    progress("reference_and_canaries_passed", errors=metrics)
    return metrics


def initialize_native(args, status, save):
    """Fresh-process isolated initialization; keep returned handle alive during gates."""
    status["python_startup_opp"] = os.environ.get("ASCEND_CUSTOM_OPP_PATH")
    if status["python_startup_opp"] != str(args.opp_root):
        if "torch" in sys.modules or "torch_npu" in sys.modules:
            raise RuntimeError("OPP startup path changed after Torch import")
        os.environ["ASCEND_CUSTOM_OPP_PATH"] = str(args.opp_root)
    if os.environ.get("ASCEND_RT_VISIBLE_DEVICES") != "0":
        raise RuntimeError("Only physical device 0 may be exposed")
    handle, symbols = isolated.symbol_provenance(args.opapi_lib)
    status.update(symbols=symbols, effective_opp_before_native_load=os.environ["ASCEND_CUSTOM_OPP_PATH"])
    save()
    import torch
    import torch_npu

    build = json.loads((args.root / "extension-status.json").read_text())
    if (
        build.get("state") != "completed"
        or build.get("returncode") != 0
        or build.get("namespace") != isolated.NAMESPACE
    ):
        raise RuntimeError("isolated extension build not qualified")
    if isolated.sha256(args.extension) != build["library_sha256"]:
        raise RuntimeError("extension changed since successful build")
    torch.ops.load_library(str(args.extension))
    status.update(
        schemas=isolated.checked_schemas(torch, build),
        opapi_mappings=isolated.opapi_mappings(args.opapi_lib),
        torch_version=torch.__version__,
        torch_npu_version=torch_npu.__version__,
    )
    torch.npu.set_device(0)
    cores = dict(torch.npu.get_stream_limit(torch.npu.current_stream()))
    validate_core_limit(cores)
    status.update(core_limit=cores, device_name=torch.npu.get_device_name(0))
    save()
    return torch, cores, handle


def child(args):
    status_path = args.output_dir / "stage2-status.json"
    status = json.loads(status_path.read_text())

    def save():
        isolated.persist(status_path, status)

    try:
        status.update(state="running", child_pid=os.getpid(), checkpoint="initialization")
        save()
        torch, core_limit, library_handle = initialize_native(args, status, save)
        passed = {attempt["id"] for attempt in status["case_attempts"] if attempt["state"] == "passed"}
        pending = [case for case in cases_for(args.case_group) if case["id"] not in passed]
        with torch.inference_mode():
            for case in pending[: args.max_cases]:
                attempt = dict(**case, state="running", checkpoint="starting", started_unix=time.time())
                status["case_attempts"].append(attempt)
                status["current_case"] = attempt
                status["npu_work_executed"] = True
                save()

                def progress(checkpoint, *, current_attempt=attempt, **evidence):
                    current_attempt.update(checkpoint=checkpoint, **evidence)
                    save()

                metrics = eager_case(torch, case, core_limit, args.seed + cases_for("all").index(case), progress)
                attempt.update(state="passed", errors=metrics, finished_unix=time.time())
                status["opapi_mappings"] = isolated.opapi_mappings(args.opapi_lib)
                save()
        if library_handle is None:
            raise AssertionError("native handle lifetime lost")
        passed = {attempt["id"] for attempt in status["case_attempts"] if attempt["state"] == "passed"}
        complete = all(case["id"] in passed for case in cases_for(args.case_group))
        status.update(
            state="completed" if complete else "partial",
            returncode=0,
            finished_unix=time.time(),
            selected_group_complete=complete,
            performance_measured=False,
            graph_qualified=False,
        )
        save()
        return 0
    except BaseException:
        if status.get("current_case", {}).get("state") == "running":
            status["current_case"].update(state="failed", error=traceback.format_exc(), finished_unix=time.time())
        status.update(state="failed", returncode=1, error=traceback.format_exc(), finished_unix=time.time())
        save()
        traceback.print_exc()
        return 1


def launcher(args):
    if os.name != "posix" or args.root.parent != Path("/workspace").resolve():
        raise RuntimeError("Native gates require an isolated direct /workspace child on Linux")
    args.opp_root = isolated.owned(args.opp_root, args.root)
    args.opapi_lib = isolated.owned(args.opp_root / "op_api/lib/libcust_opapi.so", args.root, file=True)
    build_status = json.loads((args.root / "extension-status.json").read_text())
    args.extension = isolated.owned(args.extension or build_status["extension_library"], args.root, file=True)
    args.output_dir = args.output_dir.resolve()
    if not args.output_dir.is_relative_to(args.root):
        raise ValueError("output directory must stay inside the isolated root")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    status_path = args.output_dir / "stage2-status.json"
    opp_artifacts = {
        str(path.relative_to(args.opp_root)): isolated.sha256(isolated.owned(path, args.root, file=True))
        for path in sorted(args.opp_root.rglob("*"))
        if path.is_file()
    }
    fingerprint = dict(
        phase=args.phase,
        seed=args.seed,
        case_group=args.case_group,
        cases=cases_for(args.case_group),
        rtol=RTOL,
        atol=ATOL,
        script_sha256=isolated.sha256(__file__),
        helper_sha256=isolated.sha256(isolated.__file__),
        opapi_sha256=isolated.sha256(args.opapi_lib),
        extension_sha256=isolated.sha256(args.extension),
        opp_root=str(args.opp_root),
        opp_artifacts=opp_artifacts,
    )
    # A held flock excludes concurrent launch/resume; no stale lock deletion is needed.
    import fcntl

    with (args.output_dir / "stage2.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if status_path.exists():
            if not args.resume:
                raise RuntimeError("status exists; use --resume or a fresh --output-dir")
            status = json.loads(status_path.read_text())
            if status["fingerprint"] != fingerprint:
                raise RuntimeError("resume contract differs: source/binary/cases/tolerances changed")
        else:
            if args.resume:
                raise RuntimeError("--resume requires an existing status")
            status = dict(
                fingerprint=fingerprint,
                case_attempts=[],
                runs=[],
                npu_work_executed=False,
                performance_measured=False,
                graph_qualified=False,
                full_model_qualified=False,
                scope="Eager operator correctness gates; test definitions are not executed evidence",
            )
        log_path = args.output_dir / f"stage2-run-{len(status['runs']):03d}.log"
        status["runs"].append(dict(log=str(log_path), started_unix=time.time(), max_cases=args.max_cases))
        status.update(state="queued", launcher_pid=os.getpid())
        isolated.persist(status_path, status)
        environment = dict(os.environ, ASCEND_CUSTOM_OPP_PATH=str(args.opp_root), ASCEND_RT_VISIBLE_DEVICES="0")
        environment["LD_LIBRARY_PATH"] = str(args.opapi_lib.parent) + ":" + environment.get("LD_LIBRARY_PATH", "")
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--child",
            "--phase",
            args.phase,
            "--root",
            str(args.root),
            "--opp-root",
            str(args.opp_root),
            "--extension",
            str(args.extension),
            "--output-dir",
            str(args.output_dir),
            "--case-group",
            args.case_group,
            "--max-cases",
            str(args.max_cases),
            "--seed",
            str(args.seed),
        ]
        with log_path.open("xb") as log:
            process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                cwd=args.root,
                env=environment,
                start_new_session=True,
            )
            try:
                returncode = process.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                returncode = 124
        final = json.loads(status_path.read_text())
        if final.get("returncode") != returncode or final.get("state") not in ("completed", "partial", "failed"):
            final.update(state="failed", returncode=returncode, error="child timeout/termination before final status")
            for attempt in final["case_attempts"]:
                if attempt["state"] == "running":
                    attempt.update(state="failed", error=final["error"], finished_unix=time.time())
            if final.get("current_case", {}).get("state") == "running":
                final["current_case"].update(state="failed", error=final["error"], finished_unix=time.time())
        final["runs"][-1].update(returncode=returncode, finished_unix=time.time())
        isolated.persist(status_path, final)
        print(json.dumps(dict(status=str(status_path), state=final["state"], returncode=returncode)))
        return returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("eager",), default="eager")
    parser.add_argument("--root", type=Path, default=isolated.ROOT)
    parser.add_argument("--opp-root", type=Path)
    parser.add_argument("--extension", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--case-group", choices=("positive", "zero-kv", "zero-q", "all"), default="positive")
    parser.add_argument("--max-cases", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1703)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument("--cpu-reference-check", action="store_true")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.cpu_reference_check:
        print(json.dumps(cpu_reference_check(), indent=2))
        return 0
    if args.list_cases:
        print(json.dumps(cases_for(args.case_group), indent=2))
        return 0
    if args.opp_root is None or args.max_cases <= 0 or args.timeout <= 0:
        parser.error("native execution needs --opp-root, positive --max-cases and --timeout")
    args.root = args.root.resolve(strict=True)
    args.opp_root = args.opp_root.resolve(strict=True)
    args.output_dir = (args.output_dir or args.root / "stage2-eager").resolve()
    if args.child:
        args.opapi_lib = args.opp_root / "op_api/lib/libcust_opapi.so"
        return child(args)
    return launcher(args)


if __name__ == "__main__":
    raise SystemExit(main())
