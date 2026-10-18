# SPDX-License-Identifier: Apache-2.0
"""Small isolated graph correctness gates; no model or performance measurement.

The captured graph contains the device metadata builder and attention. Exact
lengths, pages and Q/K/V contents change in place. B4 retains 16 physical query
rows: logical TND queries occupy their active prefix and the existing native
masked-output initialization must zero the remaining rows. No output zeroing
outside the operator may hide missing native writes. Output allocation guards
are unavailable because the native wrapper owns its output allocation.
"""

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

import aclgraph_fia_sink_stage2_validate as eager

isolated = eager.isolated
MAX_KV_LENGTH = 1025
MAX_QUERY_LENGTH = 4


def gate_cases(gate):
    if gate == "b1":
        return [
            dict(id=f"b1-{index}-kv{length}", q_lengths=[4], kv_lengths=[length], causal=True)
            for index, length in enumerate((4, 7, 127, 129, 513, 1025, 127, 4))
        ]
    return [
        dict(id=f"{gate}-capture", q_lengths=[4, 4, 4, 4], kv_lengths=[127] * 4, causal=True),
        dict(id=f"{gate}-holes", q_lengths=[4, 0, 4, 4], kv_lengths=[129, 0, 513, 1025], causal=True),
        dict(id=f"{gate}-resume", q_lengths=[0, 4, 0, 4], kv_lengths=[0, 1025, 0, 129], causal=True),
        dict(id=f"{gate}-shorten", q_lengths=[4, 4, 4, 0], kv_lengths=[513, 127, 129, 0], causal=True),
        dict(id=f"{gate}-zero-kv", q_lengths=[4, 4, 0, 4], kv_lengths=[0, 129, 0, 513], causal=True),
        dict(id=f"{gate}-empty", q_lengths=[0, 0, 0, 0], kv_lengths=[0, 0, 0, 0], causal=True),
        dict(id=f"{gate}-all-live", q_lengths=[4, 4, 4, 4], kv_lengths=[127, 129, 513, 1025], causal=True),
    ]


def fixed_host_inputs(torch, case, seed):
    """Reuse eager poison generation, then place it in fixed maximum storage."""
    capacity = len(case["q_lengths"])
    generated, canaries = eager.make_inputs(torch, case, seed)
    maximum_case = dict(q_lengths=[4] * capacity, kv_lengths=[MAX_KV_LENGTH] * capacity, causal=True)
    host, _ = eager.make_inputs(torch, maximum_case, seed)
    host["query_storage"].fill_(eager.QUERY_CANARY)
    live_tokens = sum(case["q_lengths"])
    host["query_storage"][eager.QUERY_GUARD_ROWS : eager.QUERY_GUARD_ROWS + live_tokens].copy_(
        generated["query_storage"][eager.QUERY_GUARD_ROWS : -eager.QUERY_GUARD_ROWS]
    )
    host["key"].fill_(eager.KEY_CANARY)
    host["value"].fill_(eager.VALUE_CANARY)
    for name in ("key", "value"):
        host[name][: generated[name].shape[0]].copy_(generated[name])
    host["block_table"].fill_(canaries["unused_table_page"])
    host["block_table"][:, : generated["block_table"].shape[1]].copy_(generated["block_table"])
    host["q_lengths"].copy_(generated["q_lengths"])
    host["kv_lengths"].copy_(generated["kv_lengths"])
    expected = torch.zeros((capacity * MAX_QUERY_LENGTH, eager.QUERY_HEADS, eager.HEAD_DIM), dtype=torch.float32)
    expected[:live_tokens] = eager.cpu_reference(
        torch,
        host["query_storage"][eager.QUERY_GUARD_ROWS : eager.QUERY_GUARD_ROWS + live_tokens],
        host["key"],
        host["value"],
        host["block_table"],
        case["q_lengths"],
        case["kv_lengths"],
        True,
    )
    return host, expected


def addresses(tensors):
    return {
        name: dict(pointer=tensor.data_ptr(), shape=list(tensor.shape), stride=list(tensor.stride()))
        for name, tensor in tensors.items()
    }


def metadata_call(torch, tensors, cores):
    return getattr(getattr(torch.ops, isolated.NAMESPACE), isolated.META_NAME)(
        eager.QUERY_HEADS,
        eager.KV_HEADS,
        eager.HEAD_DIM,
        eager.HEAD_DIM,
        actual_seq_lengths=tensors["q_lengths"],
        actual_seq_lengths_kv=tensors["kv_lengths"],
        batch_size=tensors["kv_lengths"].shape[0],
        sparse_mode=3,
        input_layout="TND",
        input_layout_kv="BnBsH",
        block_size=eager.PAGE_SIZE,
        aic_core_num=cores["cube_core_num"],
        aiv_core_num=cores["vector_core_num"],
        batch_invariant=False,
    )


def causal_separation(torch, host, case, expected):
    active = sum(case["q_lengths"])
    if not active:
        return None
    full = eager.cpu_reference(
        torch,
        host["query_storage"][eager.QUERY_GUARD_ROWS : eager.QUERY_GUARD_ROWS + active],
        host["key"],
        host["value"],
        host["block_table"],
        case["q_lengths"],
        case["kv_lengths"],
        False,
    )
    return eager.mask_witness(torch, expected[:active], full)


def attention_call(torch, tensors, metadata):
    return getattr(getattr(torch.ops, isolated.NAMESPACE), isolated.ATTENTION_NAME)(
        tensors["query_storage"][eager.QUERY_GUARD_ROWS : -eager.QUERY_GUARD_ROWS],
        tensors["key"],
        tensors["value"],
        actual_seq_qlen=tensors["q_lengths"],
        actual_seq_kvlen=tensors["kv_lengths"],
        block_table=tensors["block_table"],
        meta_data=metadata,
        atten_mask=tensors["atten_mask"],
        num_query_heads=eager.QUERY_HEADS,
        num_key_value_heads=eager.KV_HEADS,
        softmax_scale=eager.HEAD_DIM**-0.5,
        input_layout="TND",
        sparse_mode=3,
        block_size=eager.PAGE_SIZE,
        return_softmax_lse=False,
    )


def cpu_check():
    import torch

    eager.cpu_reference_check()
    for gate in ("b1", "b4", "shared"):
        descriptor = None
        for index, case in enumerate(gate_cases(gate)):
            host, expected = fixed_host_inputs(torch, case, 1703 + index)
            shape = {name: (tuple(value.shape), tuple(value.stride())) for name, value in host.items()}
            if descriptor is not None and descriptor != shape:
                raise AssertionError("fixture descriptor changed within fixed gate")
            descriptor = shape
            active = sum(case["q_lengths"])
            if not torch.equal(expected[active:], torch.zeros_like(expected[active:])):
                raise AssertionError("fixture output tail is not defined as zero")
            if not bool(torch.isfinite(expected).all()):
                raise AssertionError("reference nonfinite")
            causal_separation(torch, host, case, expected)
    return dict(
        cpu_only=True,
        fixed_input_descriptors=True,
        defined_zero_output_tails=True,
        graph_executed=False,
        npu_work_executed=False,
        performance_measured=False,
    )


def run_gate(torch, cores, gate, seed, progress):
    eager.validate_core_limit(cores)
    if (cores.get("cube_core_num"), cores.get("vector_core_num")) != (20, 40):
        raise RuntimeError(f"graph gate is restricted to the qualified 910B3 20/40 core allocation: {cores}")
    cases = gate_cases(gate)
    copies = 2 if gate == "shared" else 1
    host = [fixed_host_inputs(torch, cases[0], seed + copy * 10000)[0] for copy in range(copies)]
    tensors = [{name: value.to("npu:0") for name, value in item.items()} for item in host]
    for item in tensors[1:]:
        for name in ("q_lengths", "kv_lengths"):
            item[name] = tensors[0][name]
    input_addresses = [addresses(item) for item in tensors]
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    progress(
        "warmup_started",
        input_addresses=input_addresses,
        attention_occurrences=copies,
        metadata_occurrences=1,
        capture_includes_device_metadata=True,
    )
    with torch.npu.stream(stream):
        warm_metadata = metadata_call(torch, tensors[0], cores)
        warm_outputs = [attention_call(torch, item, warm_metadata) for item in tensors]
    torch.npu.synchronize()
    # Retain warmup owners until capture finishes; do not alias their outputs.
    graph = torch.npu.NPUGraph()
    progress("capture_started")
    with torch.npu.graph(graph, stream=stream, capture_error_mode="thread_local", auto_dispatch_capture=True):
        metadata = metadata_call(torch, tensors[0], cores)
        captured_outputs = [attention_call(torch, item, metadata) for item in tensors]
    torch.npu.synchronize()
    if warm_metadata is metadata or any(first[0] is second[0] for first, second in zip(warm_outputs, captured_outputs)):
        raise AssertionError("capture unexpectedly reused warmup Python tensor owners")
    stable_metadata = metadata.data_ptr()
    stable_outputs = [pair[0].data_ptr() for pair in captured_outputs]
    progress("capture_completed", metadata_pointer=stable_metadata, output_pointers=stable_outputs)
    results = []
    for index, case in enumerate(cases):
        generated = [fixed_host_inputs(torch, case, seed + index + copy * 10000) for copy in range(copies)]
        for item, (current_host, _) in zip(tensors, generated):
            for name, value in current_host.items():
                item[name].copy_(value)
        torch.npu.synchronize()
        progress("replay_started", case=case, occurrence=index)
        graph.replay()
        torch.npu.synchronize()
        if (
            metadata.data_ptr() != stable_metadata
            or [pair[0].data_ptr() for pair in captured_outputs] != stable_outputs
        ):
            raise AssertionError("metadata/output pointer changed")
        if [addresses(item) for item in tensors] != input_addresses:
            raise AssertionError("read-only input descriptor or pointer changed")
        errors = []
        for item, (current_host, expected), (output, lse) in zip(tensors, generated, captured_outputs):
            if output.shape != expected.shape or output.dtype != torch.bfloat16 or lse.numel():
                raise AssertionError("unexpected graph output shape/dtype/LSE")
            actual = output.float().cpu()
            if not bool(torch.isfinite(actual).all()):
                raise AssertionError("graph output nonfinite")
            changed = [name for name, before in current_host.items() if not torch.equal(item[name].cpu(), before)]
            if changed:
                raise AssertionError(f"read-only input/guard storage changed: {changed}")
            torch.testing.assert_close(actual, expected, rtol=eager.RTOL, atol=eager.ATOL)
            active = sum(case["q_lengths"])
            if not torch.equal(actual[active:], torch.zeros_like(actual[active:])):
                raise AssertionError("native masked-output initialization did not zero physical tail")
            absolute = (actual - expected).abs()
            errors.append(dict(max_abs_error=float(absolute.max()), rms_error=float(absolute.square().mean().sqrt())))
            errors[-1]["mask_sensitivity"] = causal_separation(torch, current_host, case, expected)
            errors[-1]["zero_kv_outputs_exact"] = eager.exact_zero_kv_outputs(torch, actual, case)
        if copies > 1 and sum(case["q_lengths"]) and torch.equal(generated[0][1], generated[1][1]):
            raise AssertionError("shared-plan test must use distinct Q/K/V contents")
        metadata_cpu = metadata.cpu()
        if metadata_cpu.dtype != torch.int32 or tuple(metadata_cpu.shape) != (1024,):
            raise AssertionError("metadata ABI changed")
        summary = eager.metadata_summary(metadata_cpu, cores)
        results.append(
            dict(
                case=case,
                errors=errors,
                passed=True,
                metadata_sha256=hashlib.sha256(metadata_cpu.numpy().tobytes()).hexdigest(),
                input_storage_exact=True,
                zero_tail_exact=True,
                stable_addresses=True,
                metadata_summary=summary,
            )
        )
        progress("replay_passed", results=results)
    if len({record["metadata_sha256"] for record in results}) <= 1:
        raise AssertionError("metadata contents never changed despite different device lengths")
    return dict(
        gate=gate,
        state="passed",
        results=results,
        captured_metadata_occurrences=1,
        captured_attention_occurrences=copies,
        graph_contains_planner=True,
        performance_measured=False,
    )


def child(args):
    target = args.output_dir / "stage2-graph-status.json"
    status = json.loads(target.read_text())

    def save():
        isolated.persist(target, status)

    try:
        if os.environ.get("TASK_QUEUE_ENABLE") != "0":
            raise RuntimeError("TASK_QUEUE_ENABLE=0 must be set before Torch import")
        status.update(state="running", child_pid=os.getpid(), checkpoint="initialization")
        save()
        torch, cores, handle = eager.initialize_native(args, status, save)

        def progress(checkpoint, **evidence):
            status.update(checkpoint=checkpoint, **evidence)
            save()

        with torch.inference_mode():
            status["npu_work_executed"] = True
            save()
            result = run_gate(torch, cores, args.gate, args.seed, progress)
        if handle is None:
            raise AssertionError("native library lifetime lost")
        status.update(
            state="completed",
            returncode=0,
            result=result,
            finished_unix=time.time(),
            selected_graph_gate_qualified=True,
            full_model_qualified=False,
            performance_measured=False,
        )
        save()
        return 0
    except BaseException:
        status.update(
            state="failed",
            returncode=1,
            error=traceback.format_exc(),
            finished_unix=time.time(),
            selected_graph_gate_qualified=False,
        )
        save()
        traceback.print_exc()
        return 1


def launcher(args):
    if os.name != "posix" or args.root.parent != Path("/workspace").resolve():
        raise RuntimeError("Native gate requires an isolated direct /workspace child on Linux")
    args.opp_root = isolated.owned(args.opp_root, args.root)
    args.opapi_lib = isolated.owned(args.opp_root / "op_api/lib/libcust_opapi.so", args.root, file=True)
    build = json.loads((args.root / "extension-status.json").read_text())
    args.extension = isolated.owned(args.extension or build["extension_library"], args.root, file=True)
    if not args.output_dir.is_relative_to(args.root):
        raise ValueError("output directory must remain inside isolated root")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target = args.output_dir / "stage2-graph-status.json"
    if target.exists():
        raise RuntimeError("status already exists; select a fresh output directory")
    fingerprint = dict(
        gate=args.gate,
        cases=gate_cases(args.gate),
        seed=args.seed,
        rtol=eager.RTOL,
        atol=eager.ATOL,
        scripts={str(Path(item).name): isolated.sha256(item) for item in (__file__, eager.__file__, isolated.__file__)},
        extension_sha256=isolated.sha256(args.extension),
        opp_root=str(args.opp_root),
        opp_artifacts={
            str(path.relative_to(args.opp_root)): isolated.sha256(isolated.owned(path, args.root, file=True))
            for path in sorted(args.opp_root.rglob("*"))
            if path.is_file()
        },
    )
    isolated.persist(
        target,
        dict(
            state="queued",
            fingerprint=fingerprint,
            npu_work_executed=False,
            selected_graph_gate_qualified=False,
            performance_measured=False,
            output_allocation_guards="unavailable: native wrapper owns allocation",
        ),
    )
    environment = dict(
        os.environ, ASCEND_CUSTOM_OPP_PATH=str(args.opp_root), ASCEND_RT_VISIBLE_DEVICES="0", TASK_QUEUE_ENABLE="0"
    )
    environment["LD_LIBRARY_PATH"] = str(args.opapi_lib.parent) + ":" + environment.get("LD_LIBRARY_PATH", "")
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child",
        "--root",
        str(args.root),
        "--opp-root",
        str(args.opp_root),
        "--extension",
        str(args.extension),
        "--output-dir",
        str(args.output_dir),
        "--gate",
        args.gate,
        "--seed",
        str(args.seed),
    ]
    with (args.output_dir / "stage2-graph.log").open("xb") as log:
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
            code = process.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            code = 124
    final = json.loads(target.read_text())
    if final.get("returncode") != code:
        final.update(
            state="failed",
            returncode=code,
            error="child terminated before final status",
            selected_graph_gate_qualified=False,
        )
        isolated.persist(target, final)
    print(json.dumps(dict(status=str(target), state=final["state"], returncode=code)))
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=isolated.ROOT)
    parser.add_argument("--opp-root", type=Path)
    parser.add_argument("--extension", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--gate", choices=("b1", "b4", "shared"), default="b1")
    parser.add_argument("--seed", type=int, default=1703)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--cpu-reference-check", action="store_true")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.cpu_reference_check:
        print(json.dumps(cpu_check(), indent=2))
        return 0
    if args.opp_root is None or args.timeout <= 0:
        parser.error("native execution requires --opp-root and positive timeout")
    args.root = args.root.resolve(strict=True)
    args.opp_root = args.opp_root.resolve(strict=True)
    args.output_dir = (args.output_dir or args.root / f"stage2-graph-{args.gate}").resolve()
    if args.child:
        args.opapi_lib = args.opp_root / "op_api/lib/libcust_opapi.so"
        return child(args)
    return launcher(args)


if __name__ == "__main__":
    raise SystemExit(main())
