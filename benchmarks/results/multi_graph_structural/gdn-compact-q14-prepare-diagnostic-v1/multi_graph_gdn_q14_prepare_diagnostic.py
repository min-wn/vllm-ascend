# SPDX-License-Identifier: Apache-2.0
"""One bounded, instrumented replay of the frozen Q14 preparation regression.

This is diagnostic evidence, never a replacement for the frozen performance run.
The original prepare statements are preserved and checked structurally; only
host-side trace ranges are inserted. No per-step device synchronization is added.
"""

import argparse
import ast
import contextlib
import copy
import hashlib
import inspect
import json
import textwrap
import time
from pathlib import Path

import multi_graph_gdn_q14_probe as frozen

FROZEN_PROBE_SHA256 = "bde7a1c1e4d9e401f9ce37528e2442f018cd9c067ecec3877494c557d7872b5a"
ORDER = ("forced_q4", "q14_ready", "q14_ready", "forced_q4")


def instrument_prepare(torch, records):
    """Wrap unchanged statements, so the two modes keep the same adapter work."""
    original = ast.parse(textwrap.dedent(inspect.getsource(frozen.Trial.prepare)))
    function = original.body[0]
    statements = function.body
    assert len(statements) == 19, "frozen prepare structure changed"
    assert isinstance(statements[12], ast.For)

    @contextlib.contextmanager
    def stage(trial, name):
        label = f"q14diag/stage/{trial.diagnostic_prefix}/{name}"
        with torch.profiler.record_function(label):
            start = time.perf_counter_ns()
            try:
                yield
            finally:
                records.append(dict(label=label, start_ns=start, duration_ns=time.perf_counter_ns() - start))

    def wrap(name, body):
        expression = ast.Constant(name) if isinstance(name, str) else name
        return ast.With(
            items=[
                ast.withitem(
                    context_expr=ast.Call(
                        func=ast.Name(id="diagnostic_stage", ctx=ast.Load()),
                        args=[ast.Name(id="self", ctx=ast.Load()), expression],
                        keywords=[],
                    )
                )
            ],
            body=body,
        )

    loop = statements[12]
    loop.body = [wrap(ast.BinOp(ast.Constant("input/"), ast.Add(), ast.Name("name", ast.Load())), loop.body)]
    function.body = [
        wrap("cpu_metadata", statements[:9]),
        wrap("metadata_tensor_and_h2d", statements[9:10]),
        wrap("index_conversion", statements[10:12]),
        loop,
        wrap("cpu_bookkeeping", statements[13:]),
    ]

    class RemoveRanges(ast.NodeTransformer):
        def visit_With(self, node):
            return [self.visit(statement) for statement in node.body]

    flattened = RemoveRanges().visit(copy.deepcopy(original))
    reference = ast.parse(textwrap.dedent(inspect.getsource(frozen.Trial.prepare)))
    assert ast.dump(flattened) == ast.dump(reference), "instrumentation changed executable statements"
    namespace = dict(frozen.__dict__, diagnostic_stage=stage)
    exec(compile(ast.fix_missing_locations(original), "<q14_prepare_diagnostic>", "exec"), namespace)
    return namespace["prepare"]


def collect(args):
    import torch
    import torch_npu

    import vllm_ascend.ops.gdn as gdn

    if args.output.exists() or args.trace.exists():
        raise FileExistsError("preserve existing diagnostic collection; never overwrite or repeat automatically")
    frozen_path = Path(inspect.getfile(frozen))
    assert hashlib.sha256(frozen_path.read_bytes().replace(b"\r\n", b"\n")).hexdigest() == FROZEN_PROBE_SHA256
    manifest = frozen.load_manifest(args.manifest)
    rows = manifest["payload"]["steps"]
    assert len(rows) == 16 and (rows[0]["step"], rows[-1]["step"]) == (54, 69)
    torch.npu.set_device(0)
    torch.npu.config.allow_internal_format = False
    torch.npu.matmul.allow_hf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.set_num_threads(1)
    extensions = {
        "new": frozen.load_extension("gdn_q14", args.new_extension),
        "old": frozen.load_extension("gdn_q4_tile_pipeline", args.old_extension),
        "deferred": frozen.load_extension("gdn_q4_log_deferred", args.deferred_extension),
    }
    host = torch.load(args.site, map_location="cpu", weights_only=True)
    fixture = frozen.Fixture(torch, host, len(manifest["payload"]["stable_owners"]), extensions["deferred"])
    capacity = max(row["capacity_tokens"] for row in rows) // 4
    trials = {
        mode: frozen.Trial(
            torch, fixture, mode, capacity, extensions["new"], extensions["old"], extensions["deferred"], gdn, None
        )
        for mode in dict.fromkeys(ORDER)
    }
    for trial in trials.values():
        trial.capture(rows)
        for row in rows:
            trial.prepare(row)
            trial.graphs[trial.key].replay()
        torch.npu.synchronize()
        trial.terminal()
        torch.npu.synchronize()
    host_records = []
    prepare = instrument_prepare(torch, host_records)
    event_sets = [[tuple(torch.npu.Event(enable_timing=True) for _ in range(3)) for _ in rows] for _ in ORDER]
    result = dict(
        status="running",
        diagnostic_only=True,
        formal_performance_unchanged=True,
        frozen_probe_sha256=FROZEN_PROBE_SHA256,
        diagnostic_source_sha256=frozen.sha(__file__),
        source_hashes={
            str(path): frozen.sha(path)
            for path in (args.site, args.manifest, *(args.new_extension, args.old_extension, args.deferred_extension))
        },
        manifest_payload_sha256=manifest["payload_sha256"],
        order=ORDER,
        precision="BF16 operands/output and FP32 compact state; HF32 off; highest",
        instrumentation="unchanged prepare AST with host record_function ranges; original three NPU events per step",
        synchronization="only chain boundaries, same blocking copy_ semantics as frozen source",
        capture_and_warmup_outside_trace=True,
        source_owners=fixture.source_owners,
        operand_shapes={
            name: dict(shape=list(t.shape), dtype=str(t.dtype), element_size=t.element_size())
            for name, t in fixture.operands.items()
        },
        chains=[],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.trace.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    terminal_states = {}
    with (
        torch.inference_mode(),
        torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
            record_shapes=True,
            profile_memory=False,
            with_stack=False,
            experimental_config=torch_npu.profiler._ExperimentalConfig(
                profiler_level=torch_npu.profiler.ProfilerLevel.Level0
            ),
        ) as profiler,
    ):
        for chain, mode in enumerate(ORDER):
            trial = trials[mode]
            with torch.profiler.record_function(f"q14diag/restore/{chain}/{mode}"):
                trial.restore()
                torch.npu.synchronize()
            record = dict(chain=chain, mode=mode, steps=[])
            with torch.profiler.record_function(f"q14diag/chain/{chain}/{mode}"):
                for row, (begin, compute_begin, end) in zip(rows, event_sets[chain]):
                    trial.diagnostic_prefix = f"{chain}/{mode}/{row['step']}"
                    with torch.profiler.record_function(f"q14diag/step/{trial.diagnostic_prefix}"):
                        begin.record()
                        with torch.profiler.record_function(f"q14diag/prepare/{trial.diagnostic_prefix}"):
                            prepare(trial, row)
                        compute_begin.record()
                        with torch.profiler.record_function(f"q14diag/graph/{trial.diagnostic_prefix}"):
                            trial.graphs[trial.key].replay()
                        end.record()
                with torch.profiler.record_function(f"q14diag/final_wait/{chain}/{mode}"):
                    torch.npu.synchronize()
            for row, (begin, compute_begin, end) in zip(rows, event_sets[chain]):
                record["steps"].append(
                    dict(
                        step=row["step"],
                        actual_widths=row["query_lengths"],
                        prepared_rows=224 if mode == "forced_q4" else sum(row["query_lengths"]),
                        device_prepare_ms=begin.elapsed_time(compute_begin),
                        device_graph_ms=compute_begin.elapsed_time(end),
                        device_step_ms=begin.elapsed_time(end),
                    )
                )
            with torch.profiler.record_function(f"q14diag/terminal/{chain}/{mode}"):
                terminal_states[chain] = trial.terminal().clone()
                torch.npu.synchronize()
            result["chains"].append(record)
    profiler.export_chrome_trace(str(args.trace))
    # Tensor downloads/comparison are outside the diagnostic trace.
    frozen.require_equal(torch, terminal_states[0], terminal_states[1], "diagnostic terminal Q4/Q14")
    frozen.require_equal(torch, terminal_states[1], terminal_states[2], "diagnostic Q14 repeats")
    frozen.require_equal(torch, terminal_states[0], terminal_states[3], "diagnostic Q4 repeats")
    result.update(
        status="complete_diagnostic_only",
        host_stage_records=host_records,
        terminal_bitwise=True,
        trace_sha256=frozen.sha(args.trace),
        trace_bytes=args.trace.stat().st_size,
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(status=result["status"], output=str(args.output), trace=str(args.trace))), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("new-extension", "old-extension", "deferred-extension", "site", "manifest", "output", "trace"):
        parser.add_argument("--" + name, type=Path, required=True)
    collect(parser.parse_args())


if __name__ == "__main__":
    main()
