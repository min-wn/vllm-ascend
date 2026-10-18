# SPDX-License-Identifier: Apache-2.0
"""Measure an isolated attention-chain budget on the unchanged dynamic trace.

No model weights are loaded. Each occurrence owns private Q/K/V, shared across
all controls. Synthetic resident Q/K/V are fixed; exact lengths and owner-page
routing follow the original trace. Timed wall includes host descriptor creation,
H2D, required planning/update, replay and the final join. It excludes upstream
QKV generation, model computation, sampler and serving. Attention-only is an
oracle diagnostic with a freshly prepared plan outside its timing interval.
"""

import argparse
import ctypes
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

import aclgraph_fia_sink_stage2_graph as graph_helpers
import aclgraph_fia_sink_stage2_validate as validation
from aclgraph_fia_sink_stage3_legacy import LegacyGraph
from aclgraph_fia_sink_stage3_trace import load_trace_schedule

isolated = validation.isolated
MODES = ("per_layer", "shared", "attention_only", "planner", "legacy")
MAX_ALLOCATED_BYTES = 48 * 1024**3


def installed_catalog(schedule, trace_path, model):
    """Resolve installed defaults without loading model weights or an executor."""
    from multi_graph_gdn_q4_full_loader import engine_options, resolve_default_catalog
    from multi_graph_gdn_q4_full_metadata import load_fixed_trace
    from vllm.engine.arg_utils import EngineArgs

    config = EngineArgs(**engine_options(model, 2, eager=False)).create_engine_config()
    selection = resolve_default_catalog(config, load_fixed_trace(trace_path))
    if selection["capture_sizes"] != list(schedule["default_capture_sizes"]) or selection["dispatched_tokens"] != [
        item["physical_tokens"] for item in schedule["records"]
    ]:
        raise RuntimeError("installed default catalog differs from the unchanged baseline; do not manufacture sizes")
    return dict(selection=selection, model_config_only=True, weights_loaded=False, installed_catalog_revalidated=True)


class SinkGraph:
    """Sequential graph controls share input storage, with bounded plan ownership."""

    def __init__(self, torch, mode, layers, tensors, cores, pool, stream):
        self.torch, self.mode, self.layers, self.tensors, self.cores = torch, mode, layers, tensors, cores
        self.stream, self.completion = stream, torch.npu.Event()
        self.graph, self.outputs, self.metadata = torch.npu.NPUGraph(), [], []
        self.external_metadata = (
            graph_helpers.metadata_call(torch, tensors, cores) if mode == "attention_only" else None
        )
        torch.npu.synchronize()
        stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(stream):
            warm = self.operations()
        torch.npu.synchronize()
        with torch.npu.graph(self.graph, pool=pool, stream=stream):
            self.metadata, self.outputs = self.operations()
        torch.npu.synchronize()
        del warm

    def operations(self):
        plans, outputs = [], []
        if self.mode == "planner":
            return [graph_helpers.metadata_call(self.torch, self.tensors, self.cores)], []
        shared = self.external_metadata
        if self.mode == "shared":
            shared = graph_helpers.metadata_call(self.torch, self.tensors, self.cores)
            plans.append(shared)
        for layer in self.layers:
            plan = shared
            if self.mode == "per_layer":
                plan = graph_helpers.metadata_call(self.torch, self.tensors, self.cores)
                plans.append(plan)
            arguments = {**self.tensors, **layer}
            # The helper's guarded-query ABI is unnecessary here: each private
            # query has the entire original bucket, including Q4/KV0 padding.
            outputs.append(sink_attention(self.torch, arguments, plan))
        return plans, outputs

    def prepare_oracle(self):
        if self.mode == "attention_only":
            current = graph_helpers.metadata_call(self.torch, self.tensors, self.cores)
            self.external_metadata.copy_(current)
            self.torch.npu.synchronize()

    def replay(self, query_lengths, kv_lengths):
        self.stream.wait_stream(self.torch.npu.current_stream())
        with self.torch.npu.stream(self.stream):
            self.graph.replay()
            self.completion.record(self.stream)

    def synchronize(self):
        self.completion.synchronize()


def sink_attention(torch, tensors, plan):
    return getattr(getattr(torch.ops, isolated.NAMESPACE), isolated.ATTENTION_NAME)(
        tensors["query"],
        tensors["key"].view(tensors["key"].shape[0], validation.PAGE_SIZE, -1),
        tensors["value"].view(tensors["value"].shape[0], validation.PAGE_SIZE, -1),
        actual_seq_qlen=tensors["q_lengths"],
        actual_seq_kvlen=tensors["kv_lengths"],
        block_table=tensors["block_table"],
        meta_data=plan,
        atten_mask=tensors["atten_mask"],
        num_query_heads=validation.QUERY_HEADS,
        num_key_value_heads=validation.KV_HEADS,
        softmax_scale=validation.HEAD_DIM**-0.5,
        input_layout="TND",
        sparse_mode=3,
        block_size=validation.PAGE_SIZE,
        return_softmax_lse=False,
    )[0]


def allocate_inputs(torch, schedule, occurrences, seed):
    maximum = max(schedule["required_capture_sizes"])
    pages, width = schedule["owner_pages"]["page_count"], schedule["owner_pages"]["width"]
    torch.manual_seed(seed)
    layers = []
    for occurrence in range(occurrences):
        shape = (pages, validation.PAGE_SIZE, validation.KV_HEADS, validation.HEAD_DIM)
        key = torch.randn(shape, dtype=torch.bfloat16, device="npu:0").mul_(0.5)
        value = torch.randn(shape, dtype=torch.bfloat16, device="npu:0").mul_(0.5)
        key[0].fill_(13.25)
        value[0].fill_(-17.5)
        query = torch.randn(
            (maximum, validation.QUERY_HEADS, validation.HEAD_DIM), dtype=torch.bfloat16, device="npu:0"
        ).mul_(0.5)
        layers.append(dict(query=query, key=key, value=value, occurrence=occurrence))
    mask = torch.triu(torch.ones((2048, 2048), dtype=torch.bool, device="npu:0"), diagonal=1)
    base = dict(
        block_table=torch.zeros((maximum // 4, width), dtype=torch.int32, device="npu:0"),
        q_lengths=torch.zeros(maximum // 4, dtype=torch.int64, device="npu:0"),
        kv_lengths=torch.zeros(maximum // 4, dtype=torch.int64, device="npu:0"),
        atten_mask=mask,
    )
    return layers, base


def frame_inputs(base, layers, capacity):
    # The native metadata op observes the length tensor's storage extent.
    # A sliced max-batch backing exposes 128 entries for a 100-row graph.
    tensors = {name: value[: capacity // 4].clone() if name != "atten_mask" else value for name, value in base.items()}
    views = [
        {"query": layer["query"][:capacity].clone(), "key": layer["key"], "value": layer["value"]} for layer in layers
    ]
    return views, tensors


def prepare(torch, tensors, record, *, upload_lengths=True):
    """Construct current host metadata inside the measured endpoint."""
    q = list(record["actual_seq_qlen"])
    kv = list(record["kv_lengths"])
    table = [list(row) for row in record["block_table"]]
    if upload_lengths:
        tensors["q_lengths"].copy_(torch.tensor(q, dtype=torch.int64))
        tensors["kv_lengths"].copy_(torch.tensor(kv, dtype=torch.int64))
    tensors["block_table"].copy_(torch.tensor(table, dtype=torch.int32))
    return q, kv


def memory(torch, label):
    result = dict(
        label=label,
        allocated_bytes=torch.npu.memory_allocated(),
        reserved_bytes=torch.npu.memory_reserved(),
        peak_allocated_bytes=torch.npu.max_memory_allocated(),
    )
    if result["allocated_bytes"] > MAX_ALLOCATED_BYTES:
        raise RuntimeError(f"isolated attention budget exceeds 48 GiB limit: {result}")
    return result


def profile_occurrences(torch, entries, frames, record, output_dir):
    """Separate diagnostic capture; never mix profiler overhead into trial times."""
    import torch_npu

    target = output_dir / "stage3-profile.json"
    capacity = record["physical_tokens"]
    with torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU]
    ) as profiler:
        for mode in MODES:
            entry = entries[mode, capacity]
            for repeat in range(3):
                q, kv = prepare(torch, frames[capacity][1], record, upload_lengths=mode != "legacy")
                if mode == "attention_only":
                    entry.prepare_oracle()
                torch.npu.synchronize()
                with torch.profiler.record_function(f"stage3/{mode}/{repeat}"):
                    entry.replay(q, kv)
                    entry.synchronize()
    profiler.export_chrome_trace(str(target))
    if not target.is_file():
        raise RuntimeError("profiler did not export the requested device trace")
    return dict(
        path=str(target),
        bytes=target.stat().st_size,
        sha256=isolated.sha256(target),
        outside_formal_timing=True,
        mode_replays=3,
        step=record["step"],
        capacity=capacity,
    )


def qualify(torch, entries, frames, schedule, layers, records, progress):
    """Qualify each used bucket, all private outputs, and independent CPU rows."""
    selected = {}
    for record in records:
        selected.setdefault(record["physical_tokens"], record)
    selected[-1] = records[-1]
    # Only the first occurrence's cache is needed for the independent oracle;
    # every occurrence is also compared across complete graph controls.
    key_cpu, value_cpu = layers[0]["key"].cpu(), layers[0]["value"].cpu()
    checks = []
    for record in selected.values():
        capacity = record["physical_tokens"]
        views, tensors = frames[capacity]
        q, kv = prepare(torch, tensors, record)
        torch.npu.synchronize()
        controls = {}
        for mode in ("shared", "per_layer", "attention_only", "legacy"):
            entry = entries[mode, capacity]
            if mode == "attention_only":
                entry.prepare_oracle()
            entry.replay(q, kv)
            entry.synchronize()
            actual = [output.cpu().float() for output in entry.outputs]
            for output in actual:
                if not bool(torch.isfinite(output).all()):
                    raise AssertionError(f"{mode} output nonfinite")
                if not torch.equal(output[record["live_tokens"] :], torch.zeros_like(output[record["live_tokens"] :])):
                    raise AssertionError(f"{mode} Q4/KV0 padding output not exactly zero")
            controls[mode] = actual
        for mode in ("per_layer", "attention_only", "legacy"):
            for actual, expected in zip(controls[mode], controls["shared"], strict=True):
                torch.testing.assert_close(actual, expected, rtol=validation.RTOL, atol=validation.ATOL)
        oracle_errors = []
        for row in sorted({0, record["batch"] // 2, record["batch"] - 1}):
            query = views[0]["query"][row * 4 : (row + 1) * 4].cpu()
            pages = tensors["block_table"][row : row + 1].cpu()
            expected = validation.cpu_reference(
                torch,
                query,
                key_cpu.view(key_cpu.shape[0], 128, -1),
                value_cpu.view(value_cpu.shape[0], 128, -1),
                pages,
                [4],
                [kv[row]],
                True,
            )
            actual = controls["shared"][0][row * 4 : (row + 1) * 4]
            torch.testing.assert_close(actual, expected, rtol=validation.RTOL, atol=validation.ATOL)
            oracle_errors.append(dict(row=row, max_abs_error=float((actual - expected).abs().max())))
        plans = entries["shared", capacity].metadata
        plan_summary = validation.metadata_summary(plans[0].cpu(), schedule["core_limit"])
        checks.append(
            dict(
                step=record["step"],
                capacity=capacity,
                all_private_occurrences=len(layers),
                zero_padding_exact=True,
                controls_close=True,
                fp32_sampled_rows=oracle_errors,
                metadata=plan_summary,
            )
        )
        progress("qualification", checks=checks)
    return checks


def experiment(torch, cores, args, status, save):
    """Capture matched sequential controls, then run paired dynamic trials."""
    schedule = load_trace_schedule(args.trace, args.selection)
    schedule["core_limit"] = cores
    records = schedule["records"][: args.screen_steps]
    status.update(
        schedule=schedule,
        trials=args.trials,
        screen_steps=args.screen_steps,
        modes=list(MODES),
        occurrences=args.occurrences,
        state="running",
        stage="catalog",
    )
    save()
    status["installed_catalog"] = installed_catalog(schedule, args.trace, args.model)
    status["schedule"]["installed_catalog_revalidated"] = True
    torch.set_num_threads(4)
    layers, base = allocate_inputs(torch, schedule, args.occurrences, args.seed)
    status["npu_work_executed"] = True
    required = sorted({record["physical_tokens"] for record in records}, reverse=True)
    frames = {capacity: frame_inputs(base, layers, capacity) for capacity in required}
    pool, stream = torch.npu.graph_pool_handle(), torch.npu.Stream()
    entries = {}
    status["memory"] = [memory(torch, "private_cache_allocated")]
    for capacity in required:
        record = next(item for item in records if item["physical_tokens"] == capacity)
        views, tensors = frames[capacity]
        q, kv = prepare(torch, tensors, record)
        torch.npu.synchronize()
        for mode in MODES:
            status.update(stage="capture", capture_mode=mode, capture_capacity=capacity)
            save()
            if mode == "legacy":
                entry = LegacyGraph(
                    torch, views, tensors["block_table"], tensors["atten_mask"], q, kv, pool=pool, stream=stream
                )
            else:
                entry = SinkGraph(torch, mode, views, tensors, cores, pool, stream)
            entries[mode, capacity] = entry
            status["memory"].append(memory(torch, f"captured_{mode}_{capacity}"))
            print(f"CAPTURED {mode} capacity={capacity}", flush=True)
        save()

    def progress(stage, **data):
        status.update(stage=stage, **data)
        save()

    status["checks"] = qualify(torch, entries, frames, schedule, layers, records, progress)
    status["qualification_scope"] = (
        "first occurrence of each required bucket plus final step; all private outputs cross-controls, "
        "three rows of occurrence0 FP32; not all-step/full-model correctness"
    )
    status["samples"] = []
    for (mode, capacity), entry in entries.items():
        record = next(item for item in records if item["physical_tokens"] == capacity)
        q, kv = prepare(torch, frames[capacity][1], record, upload_lengths=mode != "legacy")
        if mode == "attention_only":
            entry.prepare_oracle()
        entry.replay(q, kv)
        entry.synchronize()
    start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
    for trial in range(args.trials):
        order = MODES if trial % 2 == 0 else tuple(reversed(MODES))
        for mode in order:
            for record in records:
                capacity = record["physical_tokens"]
                entry, tensors = entries[mode, capacity], frames[capacity][1]
                if mode == "attention_only":
                    prepare(torch, tensors, record)
                    torch.npu.synchronize()
                    entry.prepare_oracle()
                torch.npu.synchronize()
                wall_start = time.perf_counter_ns()
                q, kv = prepare(torch, tensors, record, upload_lengths=mode != "legacy")
                prepared_at = time.perf_counter_ns()
                # Events mark a queue span including possible host submission
                # gaps. They are not pure kernel-duration measurements.
                start.record(entry.stream)
                entry.replay(q, kv)
                end.record(entry.stream)
                submitted_at = time.perf_counter_ns()
                entry.synchronize()
                end.synchronize()
                finished = time.perf_counter_ns()
                status["samples"].append(
                    dict(
                        trial=trial,
                        mode=mode,
                        step=record["step"],
                        batch=record["batch"],
                        capacity=capacity,
                        wall_us=(finished - wall_start) / 1000,
                        queue_span_us=start.elapsed_time(end) * 1000,
                        preparation_enqueue_us=(prepared_at - wall_start) / 1000,
                        submit_join_cpu_us=(submitted_at - prepared_at) / 1000,
                    )
                )
            progress("timing", completed_trial=trial, completed_mode=mode)

    def post_progress(stage, **data):
        progress("post_timing_qualification", post_timing_checks=data["checks"])

    status["post_timing_checks"] = qualify(torch, entries, frames, schedule, layers, records[-1:], post_progress)
    status["legacy_update_tasks_per_graph"] = args.occurrences
    status["legacy_workspace_bytes_by_capacity"] = {
        str(capacity): entries["legacy", capacity].workspace_bytes() for capacity in required
    }
    progress("profiling")
    status["profile"] = profile_occurrences(torch, entries, frames, records[0], args.output_dir)
    status.update(
        state="completed",
        returncode=0,
        stage="completed",
        attention_trace_measured=True,
        performance_measured=True,
        full_model_qualified=False,
        end_to_end_gain_established=False,
        incremental_native_planner_implemented=False,
        finished_unix=time.time(),
    )
    status["memory"].append(memory(torch, "timing_complete"))
    save()


def child(args):
    target = args.output_dir / "stage3-budget.json"
    status = json.loads(target.read_text())

    def save():
        isolated.persist(target, status)

    try:
        if os.environ.get("TASK_QUEUE_ENABLE") != "0":
            raise RuntimeError("TASK_QUEUE_ENABLE=0 required before Torch import")
        status.update(state="initializing", child_pid=os.getpid())
        save()
        torch, cores, handle = validation.initialize_native(args, status, save)
        validation.validate_core_limit(cores)
        with torch.inference_mode():
            experiment(torch, cores, args, status, save)
        if handle is None:
            raise AssertionError("native library owner lost")
        status["final_native_provenance"] = final_native_provenance(args.opapi_lib)
        save()
        return 0
    except BaseException:
        status.update(state="failed", returncode=1, error=traceback.format_exc(), finished_unix=time.time())
        save()
        traceback.print_exc()
        return 1


def final_native_provenance(expected):
    """Permit the known vLLM registration library, but no alternate sink owner."""
    allowed = Path("/usr/local/Ascend/cann-9.1.0/opp/vendors/batch_invariant/op_api/lib/libcust_opapi.so")
    allowed_hash = "d1eecbbe84ac8b32af671b09a988f43d953198f671e34c379bb38c2b356670b8"
    mappings = set()
    for line in Path("/proc/self/maps").read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and Path(fields[5]).name == "libcust_opapi.so":
            mappings.add(Path(fields[5]).resolve(strict=True))
    if expected not in mappings or mappings - {expected, allowed}:
        raise RuntimeError(f"unexpected native operator library: {mappings}")
    records = []
    for path in sorted(mappings):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if path == allowed:
            if digest != allowed_hash:
                raise RuntimeError("known registration library hash changed")
            handle = ctypes.CDLL(str(path), mode=os.RTLD_NOW | os.RTLD_LOCAL)
            for name in isolated.REQUIRED_SYMBOLS:
                if hasattr(handle, name):
                    raise RuntimeError(f"additional library exports experiment symbol: {name}")
        records.append(dict(path=str(path), sha256=digest))
    handle, symbols = isolated.symbol_provenance(expected)
    return dict(
        libraries=records,
        sink_symbols=symbols,
        known_extra_library="vLLM imports batch-invariant registration even when that mode is disabled",
        alternate_sink_symbols_absent=True,
        selected_library_handle_retained=handle is not None,
    )


def launcher(args):
    if os.name != "posix" or args.root.parent != Path("/workspace").resolve():
        raise RuntimeError("only the isolated remote /workspace tree is supported")
    args.opp_root = isolated.owned(args.opp_root, args.root)
    args.opapi_lib = isolated.owned(args.opp_root / "op_api/lib/libcust_opapi.so", args.root, file=True)
    build = json.loads((args.root / "extension-status.json").read_text())
    args.extension = isolated.owned(args.extension or build["extension_library"], args.root, file=True)
    if not args.output_dir.is_relative_to(args.root):
        raise ValueError("output directory must be inside isolated root")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target = args.output_dir / "stage3-budget.json"
    if target.exists():
        raise RuntimeError("preserve previous run; choose a fresh output directory")
    scripts = (
        Path(__file__),
        Path(validation.__file__),
        Path(graph_helpers.__file__),
        Path(isolated.__file__),
        Path(__file__).with_name("aclgraph_fia_sink_stage3_trace.py"),
        Path(__file__).with_name("aclgraph_fia_sink_stage3_legacy.py"),
    )
    isolated.persist(
        target,
        dict(
            state="queued",
            started_unix=time.time(),
            scope="isolated sequential attention chain, not full model or online serving",
            synthetic_inputs=(
                "fixed resident private Q/K/V per occurrence; current owner pages and exact lengths "
                "follow canonical trace"
            ),
            endpoint="host descriptor construction + H2D + required planner/FIA updates + graph replay + final join",
            oracle_diagnostic=(
                "attention_only excludes freshly generated exact metadata from timing; not a valid dynamic algorithm"
            ),
            scripts={path.name: isolated.sha256(path) for path in scripts},
            extension_sha256=isolated.sha256(args.extension),
            opp_artifacts={
                str(path.relative_to(args.opp_root)): isolated.sha256(path)
                for path in sorted(args.opp_root.rglob("*"))
                if path.is_file()
            },
            performance_measured=False,
            full_model_qualified=False,
            end_to_end_gain_established=False,
        ),
    )
    env = dict(
        os.environ,
        ASCEND_CUSTOM_OPP_PATH=str(args.opp_root),
        ASCEND_RT_VISIBLE_DEVICES="0",
        TASK_QUEUE_ENABLE="0",
        VLLM_BATCH_INVARIANT="0",
    )
    env["LD_LIBRARY_PATH"] = str(args.opapi_lib.parent) + ":" + env.get("LD_LIBRARY_PATH", "")
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
        "--trace",
        str(args.trace),
        "--selection",
        str(args.selection),
        "--model",
        args.model,
        "--occurrences",
        str(args.occurrences),
        "--screen-steps",
        str(args.screen_steps),
        "--trials",
        str(args.trials),
        "--seed",
        str(args.seed),
    ]
    with (args.output_dir / "stage3-budget.log").open("xb") as log:
        process = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            cwd=args.root,
            env=env,
            start_new_session=True,
        )
        try:
            rc = process.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            rc = 124
    status = json.loads(target.read_text())
    if status.get("returncode") != rc:
        status.update(state="failed", returncode=rc, error="child terminated before final status")
        isolated.persist(target, status)
    print(json.dumps(dict(status=str(target), state=status["state"], returncode=rc)))
    return rc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--opp-root", type=Path, required=True)
    parser.add_argument("--extension", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--model", default="/models/Qwen3.5-27B-original")
    parser.add_argument("--occurrences", type=int, choices=(1, 16), default=16)
    parser.add_argument("--screen-steps", type=int, default=128)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1703)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.root, args.opp_root = args.root.resolve(strict=True), args.opp_root.resolve(strict=True)
    args.output_dir = args.output_dir.resolve()
    args.trace, args.selection = args.trace.resolve(strict=True), args.selection.resolve(strict=True)
    if not 1 <= args.screen_steps <= 128 or args.trials < 1 or args.timeout < 1:
        parser.error("positive trials/timeout and screen steps in 1..128 required")
    if args.child:
        args.opapi_lib = args.opp_root / "op_api/lib/libcust_opapi.so"
        return child(args)
    return launcher(args)


if __name__ == "__main__":
    raise SystemExit(main())
