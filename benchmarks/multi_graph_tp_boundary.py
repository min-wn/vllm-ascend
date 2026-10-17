"""TP communication/QKV dependency gate, not a decoder speed benchmark.

Use the loaded model's real TP group and local QKV shard on both sides. No
attention is present, so this diagnostic cannot establish an FIA-update or
complete-replay speedup. All graph capacities come from the default dispatcher.
"""

import argparse
import hashlib
import json
import os
import random
import statistics
import time
from pathlib import Path

from multi_graph_bucket_block import bucket_topology
from multi_graph_replay import run

from vllm_ascend.worker.multi_graph.replay_benchmark import ReplayWorkload


def select_workload(args):
    from vllm.engine.arg_utils import EngineArgs
    from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher

    batch = random.Random(20261016).randint(64, 256)
    options = (
        {"speculative_config": {"method": "ngram", "num_speculative_tokens": args.query - 1}} if args.query > 1 else {}
    )
    config = EngineArgs(
        model=args.model,
        dtype="bfloat16",
        tensor_parallel_size=args.tensor_parallel_size,
        max_num_seqs=256,
        max_model_len=2048,
        **options,
    ).create_engine_config()
    dispatcher = CudagraphDispatcher(config)
    dispatcher.initialize_cudagraph_keys(config.compilation_config.cudagraph_mode, args.query)
    mode, descriptor = dispatcher.dispatch(batch * args.query, uniform_decode=True)
    if str(mode) != "FULL":
        raise ValueError("sample does not fit default FULL graphs; refuse a larger custom capture size")
    sizes = [n for n in config.compilation_config.cudagraph_capture_sizes if n % args.query == 0]
    capacity, pair = bucket_topology(batch * args.query, sizes, requested=args.pair)
    if capacity != descriptor.num_tokens or capacity // args.query != descriptor.num_reqs:
        raise ValueError("default dispatcher and graph capacity disagree")
    return dict(
        seed=20261016,
        batch=batch,
        query=args.query,
        capture_sizes=config.compilation_config.cudagraph_capture_sizes,
        capacity_tokens=capacity,
        capacity_requests=descriptor.num_reqs,
        pair_tokens=pair,
        pair_requests=[n // args.query for n in pair],
        default_config_uses_ngram_for_capacity_only=args.query > 1,
    )


def experiment(args, workload, runner, model, inputs, context, reset_kv, *, module=None):
    # Capture and qualify all schedules in one process group before timing.
    import torch
    import torch.distributed as dist
    import torch.nn.functional as functional
    import torch_npu
    from vllm.distributed.parallel_state import get_tp_group, get_world_group

    group = get_tp_group()
    cpu_group = get_world_group().cpu_group
    rank, world = dist.get_rank(), dist.get_world_size()
    if world != args.tensor_parallel_size or group.world_size != world or world < 2:
        raise ValueError("requires the requested multi-rank tensor-parallel group")
    if module is None:
        module = model.model.layers[0].self_attn.qkv_proj
    if module.bias is not None or module.weight.dtype != torch.bfloat16:
        raise ValueError("this diagnostic requires unquantized, bias-free BF16 QKV")
    torch.npu.config.allow_internal_format = True
    weight_nd = torch_npu.npu_format_cast(module.weight, 2)
    weight_nz = torch_npu.npu_format_cast(weight_nd, 29)
    if torch_npu.get_npu_format(weight_nz) != 29:
        raise ValueError("NZ weight conversion did not take effect")
    rows = workload.capacity * workload.query
    boundary = args.pair[0]
    data = torch.empty((rows, weight_nd.shape[1]), device="npu", dtype=torch.bfloat16)
    data_context = inputs(0, workload.capacity)
    main = torch.npu.current_stream()
    checks = []
    graphs = {}
    numerical_diagnostics = []
    args.numerical_diagnostics = numerical_diagnostics

    def prepare(batch, variant):
        torch.manual_seed(42 + variant)
        value = torch.randn(data.shape, dtype=data.dtype, device=data.device) * 0.1
        value[batch * workload.query :].zero_()
        data.copy_(value * (rank + 1))
        torch.npu.synchronize()

    def references():
        independent_sum = data.float().cpu()
        dist.all_reduce(independent_sum, group=cpu_group)
        reduced = group.all_reduce(data.clone())
        torch.testing.assert_close(reduced.float().cpu(), independent_sum, rtol=0.02, atol=0.02)
        with context(data_context):
            official, _ = module(reduced)
        reference = functional.linear(reduced, weight_nd)
        torch.testing.assert_close(official, reference, rtol=0.02, atol=0.02)
        split_reduced = torch.cat(
            (group.all_reduce(data[:boundary].clone()), group.all_reduce(data[boundary:].clone()))
        )

        def compare(label, actual, wanted):
            actual, wanted = actual.float(), wanted.float()
            delta = (actual - wanted).abs()
            return dict(
                label=label,
                max_abs=delta.max().item(),
                mismatches=(delta > 0.02 + 0.02 * wanted.abs()).sum().item(),
                elements=actual.numel(),
            )

        def split_linear(value, weight):
            return torch.cat((functional.linear(value[:boundary], weight), functional.linear(value[boundary:], weight)))

        diagnostic = dict(
            rank=rank,
            checks=[
                compare("split_collective_vs_full", split_reduced, reduced),
                compare("split_collective_vs_cpu", split_reduced.cpu(), independent_sum),
                compare("nd_split_same_sum_vs_nd_full", split_linear(reduced, weight_nd), reference),
                compare("nz_split_same_sum_vs_nd_full", split_linear(reduced, weight_nz), reference),
                compare("nd_full_split_sum_vs_nd_full", functional.linear(split_reduced, weight_nd), reference),
                compare("nz_split_split_sum_vs_nd_full", split_linear(split_reduced, weight_nz), reference),
            ],
        )
        numerical_diagnostics.append(diagnostic)
        print("numerical_diagnostics=" + json.dumps(diagnostic), flush=True)
        return {
            "ar": reduced.clone(),
            "qkv": functional.linear(data, weight_nd),
            "combined": reference,
        }

    prepare(workload.batch, 0)
    reference = references()

    class CapturedBoundary:
        def __init__(self, mode):
            self.mode = mode
            self.root, self.comm = torch.npu.Stream(), torch.npu.Stream()
            self.events = [torch.npu.Event() for _ in range(3)]
            self.graph = torch.npu.NPUGraph()
            self.pool = torch.npu.graph_pool_handle()
            self.working = torch.empty_like(data)
            self.root.wait_stream(main)
            with torch.npu.stream(self.root):
                for _ in range(3):
                    self.operation()
            torch.npu.synchronize()
            dist.barrier(group=cpu_group)
            with torch.npu.graph(self.graph, pool=self.pool, stream=self.root):
                self.output = self.operation()
            torch.npu.synchronize()

        def operation(self):
            mode = self.mode
            # The installed TP communicator reduces in place. Restore one
            # identical full-size working buffer in every diagnostic graph.
            self.working.copy_(data)
            value = self.working
            left, right = value[:boundary], value[boundary:]
            if mode.startswith("ar_"):
                values = {"ar_full": (value,), "ar_left": (left,), "ar_right": (right,), "ar_split": (left, right)}
                return tuple(group.all_reduce(value) for value in values[mode])
            if mode == "qkv_full":
                return (functional.linear(value, weight_nz),)
            if mode == "qkv_split":
                return tuple(functional.linear(value, weight_nz) for value in (left, right))
            if mode == "joined":
                return (functional.linear(group.all_reduce(value), weight_nz),)
            if mode == "split_serial":
                first, second = group.all_reduce(left), group.all_reduce(right)
                return tuple(functional.linear(value, weight_nz) for value in (first, second))
            if mode != "split_overlap":
                raise ValueError(mode)
            fork, first_ready, second_ready = self.events
            fork.record(self.root)
            self.comm.wait_event(fork)
            with torch.npu.stream(self.comm):
                first = group.all_reduce(left)
                first_ready.record(self.comm)
                second = group.all_reduce(right)
                second_ready.record(self.comm)
            self.root.wait_event(first_ready)
            first_output = functional.linear(first, weight_nz)
            self.root.wait_event(second_ready)
            second_output = functional.linear(second, weight_nz)
            return first_output, second_output

        def replay(self):
            with torch.npu.stream(self.root):
                self.graph.replay()

        def check(self, expected, label):
            source = expected[
                "ar" if self.mode.startswith("ar_") else "qkv" if self.mode.startswith("qkv_") else "combined"
            ]
            if self.mode == "ar_left":
                references = (source[:boundary],)
            elif self.mode == "ar_right":
                references = (source[boundary:],)
            elif len(self.output) == 2:
                references = (source[:boundary], source[boundary:])
            else:
                references = (source,)
            for actual, wanted in zip(self.output, references, strict=True):
                torch.testing.assert_close(
                    actual, wanted, rtol=0.02, atol=0.02, msg=lambda message: f"{self.mode}/{label}: {message}"
                )
            checks.append(f"{self.mode}/{label}")

    modes = (
        "ar_full",
        "ar_left",
        "ar_right",
        "ar_split",
        "qkv_full",
        "qkv_split",
        "joined",
        "split_serial",
        "split_overlap",
    )
    for name in modes:
        print(f"rank={rank} capture {name}", flush=True)
        entry = CapturedBoundary(name)
        entry.replay()
        torch.npu.synchronize()
        entry.check(reference, "initial")
        graphs[name] = entry
    identities = {name: (id(entry.graph), tuple(t.data_ptr() for t in entry.output)) for name, entry in graphs.items()}
    for batch, variant in ((workload.batch - 1, 1), (workload.capacity - 1, 2), (workload.batch, 3)):
        prepare(batch, variant)
        reference = references()
        for name, entry in graphs.items():
            entry.root.wait_stream(main)
            entry.replay()
            torch.npu.synchronize()
            entry.check(reference, f"changed-input-b{batch}-v{variant}")
            if identities[name] != (id(entry.graph), tuple(t.data_ptr() for t in entry.output)):
                raise ValueError("graph or output address changed")

    samples = []
    start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
    for round_id in range(args.rounds):
        for name in modes if round_id % 2 == 0 else reversed(modes):
            entry = graphs[name]
            for _ in range(3):
                entry.replay()
            torch.npu.synchronize()
            wall, device = [], []
            for _ in range(args.repeats):
                dist.barrier(group=cpu_group)
                start.record(entry.root)
                started = time.perf_counter_ns()
                for _ in range(args.train):
                    entry.replay()
                end.record(entry.root)
                end.synchronize()
                wall.append((time.perf_counter_ns() - started) / (1000 * args.train))
                device.append(start.elapsed_time(end) * 1000 / args.train)
            entry.check(reference, f"round{round_id}")
            samples.append(dict(mode=name, round=round_id, wall_us=wall, device_us=device))
            print(f"rank={rank} {name} round={round_id}: device={statistics.median(device):.3f} us", flush=True)
    if args.trace:
        with torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU]
        ) as profiler:
            for name in ("joined", "split_serial", "split_overlap"):
                for _ in range(2):
                    dist.barrier(group=cpu_group)
                    with torch.profiler.record_function(f"tp_boundary.{name}"):
                        graphs[name].replay()
                        torch.npu.synchronize()
        profiler.export_chrome_trace(str(args.output.with_suffix(f".rank{rank}.trace.json")))
    local = dict(
        rank=rank,
        checks=checks,
        samples=samples,
        local_qkv_shape=list(weight_nd.shape),
        numerical_diagnostics=numerical_diagnostics,
    )
    results = [None] * world
    dist.all_gather_object(results, local, group=cpu_group)
    if rank == 0:
        summaries = {}
        for name in modes:
            summary = {}
            for metric in ("wall_us", "device_us"):
                rounds = []
                for round_id in range(args.rounds):
                    rank_rows = [
                        next(row for row in result["samples"] if row["mode"] == name and row["round"] == round_id)
                        for result in results
                    ]
                    rounds.append(
                        statistics.median(max(row[metric][i] for row in rank_rows) for i in range(args.repeats))
                    )
                summary[metric] = statistics.median(rounds)
                summary[f"rounds_{metric}"] = rounds
            summaries[name] = summary
        times = {name: value["device_us"] for name, value in summaries.items()}
        report = dict(
            status="passed",
            scope="communication_and_qkv_only_no_attention_no_decoder_speedup",
            model=args.model,
            module_path=getattr(args, "boundary_module_path", "model.model.layers[0].self_attn.qkv_proj"),
            loader=getattr(args, "loader_metadata", {"method": "multi_graph_replay.run"}),
            tensor_parallel_size=world,
            selection=args.selection,
            capacity=rows,
            pair=args.pair,
            payload_bytes=data.numel() * data.element_size(),
            qkv_weight_format=torch_npu.get_npu_format(weight_nz),
            communicator=type(group.device_communicator).__qualname__,
            communication_path="get_tp_group().all_reduce on both baseline and split",
            replay_train_length=args.train,
            identical_full_input_restore_copy_per_graph=True,
            summaries=summaries,
            diagnostics_us=dict(
                early_release_estimate=times["ar_full"] - times["ar_left"],
                added_collective_cost=times["ar_split"] - times["ar_full"],
                qkv_split_cost=times["qkv_split"] - times["qkv_full"],
                overlap_saving_vs_same_split=times["split_serial"] - times["split_overlap"],
                boundary_saving_vs_joined=times["joined"] - times["split_overlap"],
            ),
            ranks=results,
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            limitations=(
                "Hot selected-layer QKV weights; synthetic rank-local Down outputs. "
                "No Down, Norm, FIA, FIA updates, input preparation or end-to-end decoder. "
                "Component differences are diagnostics, not additive wall-time predictions. "
                "Graph replay trains amortize host calls; event spans can include submission gaps."
            ),
        )
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        print(json.dumps(dict(summary=summaries, diagnostics=report["diagnostics_us"]), indent=2), flush=True)


def write_failure(args, error):
    rank = int(os.environ.get("RANK", "0"))
    report = dict(
        status="failed",
        rank=rank,
        scope="communication_and_qkv_only_no_attention_no_decoder_speedup",
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        selection=args.selection,
        loader=getattr(args, "loader_metadata", {"method": args.loader}),
        numerical_diagnostics=getattr(args, "numerical_diagnostics", []),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        error_type=type(error).__name__,
        error=str(error),
    )
    content = json.dumps(report, indent=2, allow_nan=False)
    # A nonzero rank can fail first while rank zero is inside a collective.
    args.output.with_suffix(f".rank{rank}.failure.json").write_text(content, encoding="utf-8")
    if rank == 0:
        args.output.write_text(content, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/models/Qwen3-4B")
    parser.add_argument("--loader", choices=("replay", "qwen35-weights"), default="replay")
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--query", type=int, choices=(1, 4), default=1)
    parser.add_argument("--pair", type=int, nargs=2, help="two existing default token capacities summing to padding")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--train", type=int, default=8)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or min(args.rounds, args.repeats, args.train) < 1:
        parser.error("positive counts and a new output path required")
    if int(os.environ.get("WORLD_SIZE", "1")) != args.tensor_parallel_size:
        parser.error("launch with torchrun and matching --tensor-parallel-size")
    args.dtype, args.batch_invariant = "bfloat16", False
    args.kv_length = 512
    args.selection = select_workload(args)
    capacity = args.selection["capacity_requests"]
    args.pair = args.selection["pair_tokens"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    batch = args.selection["batch"]
    left = min(batch, args.selection["pair_requests"][0])
    workload = ReplayWorkload(batch, capacity, args.query, args.kv_length, (left, batch - left))
    try:
        if args.loader == "qwen35-weights":
            from multi_graph_tp_boundary_loader import run_qwen35

            run_qwen35(args, workload, experiment=experiment)
        else:
            run(args, workload, experiment=experiment, snapshot_first_block=True)
    except Exception as error:
        write_failure(args, error)
        raise


if __name__ == "__main__":
    main()
