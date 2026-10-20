# SPDX-License-Identifier: Apache-2.0
"""Qualify one replicated Down endpoint with optional AIV GEMM/RS fusion.

Synthetic inputs and optional checkpoint weights at Qwen3.5-27B TP2 dimensions. Reconstruction, input
refresh and stream joins are included. No attention, state or serving claim.
"""

import argparse
import hashlib
import json
import os
import statistics
import time
from datetime import timedelta
from pathlib import Path

MODES = ("mm_ar", "mm_rs_ag", "fused_rs_ag")


def validate_capacity(capacity, catalog):
    if type(capacity) is not int or capacity not in catalog or capacity % 8:
        raise ValueError("capacity must be an existing default bucket with whole Q4 requests on both TP ranks")


def down_weight_spec(index, rank):
    if type(rank) is not int or rank not in (0, 1):
        raise ValueError("Down weight ownership requires rank 0 or 1")
    key = "model.language_model.layers.3.mlp.down_proj.weight"
    mapping = json.loads(index.read_bytes())["weight_map"]
    return index.parent / mapping[key], key, (rank * 8704, (rank + 1) * 8704)


def pair_samples(ranks):
    if len(ranks) != 2 or not ranks[0] or len(ranks[0]) != len(ranks[1]):
        raise ValueError("require two nonempty equally sized rank sample lists")
    result, seen = [], set()
    for a, b in zip(*ranks, strict=True):
        identity = (a["trial"], a["mode"], a["sample"])
        if identity != (b["trial"], b["mode"], b["sample"]) or identity in seen:
            raise ValueError("mismatched or repeated paired sample")
        seen.add(identity)
        row = dict(zip(("trial", "mode", "sample"), identity, strict=True))
        for metric in ("device_us", "wall_us"):
            if any(not 0 < item[metric] < float("inf") for item in (a, b)):
                raise ValueError("nonfinite or nonpositive latency")
            row[metric] = max(a[metric], b[metric])
        result.append(row)
    return result


def experiment(args, report, save):
    import torch
    import torch.distributed as dist
    import torch_npu
    from torch.distributed.distributed_c10d import _get_default_group

    rank = int(os.environ["RANK"])
    if int(os.environ["WORLD_SIZE"]) != 2:
        raise ValueError("this probe requires TP2")
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    torch.npu.config.allow_internal_format = True
    dist.init_process_group("hccl", timeout=timedelta(seconds=120))
    cpu_group = dist.new_group(backend="gloo", timeout=timedelta(seconds=120))
    group = _get_default_group()
    hcom = group._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
    report["runtime"] = dict(torch=torch.__version__, torch_npu=torch_npu.__version__)
    report["checks"], report["samples"] = [], []

    def stage(name):
        report["status"] = name
        save()
        print(f"rank={rank} stage={name}", flush=True)

    if args.weight_index is None:
        generator = torch.Generator(device="cpu").manual_seed(20261020 + rank)
        cpu_weight = torch.randn((5120, 8704), generator=generator).to(torch.bfloat16) * 0.01
    else:
        from safetensors import safe_open

        path, key, (start, stop) = down_weight_spec(args.weight_index, rank)
        with safe_open(path, framework="pt", device="cpu") as checkpoint:
            tensor = checkpoint.get_slice(key)
            if tensor.get_shape() != [5120, 17408]:
                raise ValueError("require the actual Qwen3.5-27B Down shape")
            cpu_weight = tensor[:, start:stop].contiguous()
        if cpu_weight.dtype != torch.bfloat16:
            raise ValueError("require the original BF16 checkpoint weight")
        report["checkpoint_weight"] = dict(
            file=str(path),
            key=key,
            columns=[start, stop],
            index_sha256=hashlib.sha256(args.weight_index.read_bytes()).hexdigest(),
            shard_sha256=hashlib.sha256(cpu_weight.view(torch.uint8).numpy().tobytes()).hexdigest(),
        )
    weight = torch_npu.npu_format_cast(cpu_weight.to("npu"), 29)
    source = torch.zeros((args.capacity, 8704), dtype=torch.bfloat16, device="npu")
    data = torch.empty_like(source)
    report["weight"] = dict(shape=list(weight.shape), format=torch_npu.get_npu_format(weight))
    root = torch.npu.Stream()
    graphs, outputs = {}, {}

    def prepare(epoch):
        gen = torch.Generator(device="cpu").manual_seed(1700 + 17 * rank + epoch)
        value = torch.randn((args.capacity, 8704), generator=gen).to(torch.bfloat16)
        # The last two/three Q4 requests stay inactive; changed input tests stale replay storage.
        value[args.capacity - (8 + 4 * (epoch % 2)) :].zero_()
        source.copy_(value)
        torch.npu.synchronize()

    def operation(mode):
        data.copy_(source)
        if mode == "mm_ar":
            result = torch.nn.functional.linear(data, weight)
            dist.all_reduce(result)
            return result
        if mode == "mm_rs_ag":
            partial = torch.nn.functional.linear(data, weight)
            local = torch.empty((args.capacity // 2, 5120), dtype=data.dtype, device=data.device)
            dist.reduce_scatter_tensor(local, partial)
        else:
            local = torch_npu.npu_mm_reduce_scatter_base(data, weight.t(), hcom, 2, comm_mode="aiv")
        result = torch.empty((args.capacity, 5120), dtype=data.dtype, device=data.device)
        dist.all_gather_into_tensor(result, local)
        return result

    def check(actual, reference, label, *, exact=False):
        if actual.shape != (args.capacity, 5120) or not torch.isfinite(actual).all().item():
            raise AssertionError(f"{label}: invalid replicated endpoint")
        if torch.count_nonzero(actual[-8:]).item():
            raise AssertionError(f"{label}: inactive rows changed")
        delta = actual.float() - reference.float()
        report["checks"].append(
            dict(
                label=label,
                max_abs=delta.abs().max().item(),
                nrmse=(delta.square().mean() / reference.float().square().mean().clamp_min(1e-30)).sqrt().item(),
                exact=exact,
            )
        )
        torch.testing.assert_close(actual, reference, rtol=0 if exact else 0.02, atol=0 if exact else 0.02)

    for epoch in range(2):
        prepare(epoch)
        references = {}
        for mode in MODES:
            stage(f"eager/{epoch}/{mode}")
            dist.barrier(group=cpu_group)
            value = operation(mode)
            torch.npu.synchronize()
            references[mode] = value.clone()
            check(value, references["mm_ar"], f"eager/{epoch}/{mode}")
        if args.phase == "eager":
            continue
        if epoch == 0:
            for mode in MODES:
                stage(f"capture/{mode}")
                for _ in range(2):
                    operation(mode)
                torch.npu.synchronize()
                root.wait_stream(torch.npu.current_stream())
                dist.barrier(group=cpu_group)
                graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph, stream=root):
                    output = operation(mode)
                graphs[mode], outputs[mode] = graph, output
        for mode in MODES:
            stage(f"replay_check/{epoch}/{mode}")
            root.wait_stream(torch.npu.current_stream())
            with torch.npu.stream(root):
                graphs[mode].replay()
            torch.npu.synchronize()
            check(outputs[mode], references[mode], f"replay/{epoch}/{mode}", exact=True)
    if args.phase == "eager":
        stage("eager_qualified_no_graph_or_timing")
        dist.destroy_process_group()
        return

    samples = []
    for trial in range(5):
        prepare(10 + trial)
        for mode in MODES if trial % 2 == 0 else reversed(MODES):
            for sample in range(-2, 6):
                root.wait_stream(torch.npu.current_stream())
                dist.barrier(group=cpu_group)
                begin, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
                start = time.perf_counter_ns()
                with torch.npu.stream(root):
                    begin.record(root)
                    graphs[mode].replay()
                    end.record(root)
                end.synchronize()
                wall_us = (time.perf_counter_ns() - start) / 1000
                if sample >= 0:
                    samples.append(
                        dict(
                            trial=trial,
                            mode=mode,
                            sample=sample,
                            wall_us=wall_us,
                            device_us=begin.elapsed_time(end) * 1000,
                        )
                    )
    reports = [None, None]
    report["rank_samples"] = samples
    dist.all_gather_object(reports, samples, group=cpu_group)
    report["samples"] = pair_samples(reports)
    report["medians"] = {
        mode: {
            metric: statistics.median(s[metric] for s in report["samples"] if s["mode"] == mode)
            for metric in ("device_us", "wall_us")
        }
        for mode in MODES
    }
    stage("timing_complete_profiles_pending")
    report["profiles"] = []
    for mode in MODES:
        with torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU]
        ) as prof:
            dist.barrier(group=cpu_group)
            with torch.npu.stream(root):
                graphs[mode].replay()
            torch.npu.synchronize()
        path = args.output.with_name(f"{args.output.stem}.{mode}.rank{rank}.trace.json")
        prof.export_chrome_trace(str(path))
        report["profiles"].append(dict(mode=mode, file=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    stage("qualified_synthetic_down_endpoint_not_full_model")
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog-report", type=Path, required=True)
    parser.add_argument("--capacity", type=int, default=400)
    parser.add_argument("--phase", choices=("eager", "replay"), default="eager")
    parser.add_argument("--weight-index", type=Path, help="optional real Qwen3.5-27B checkpoint index")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rank = int(os.environ.get("RANK", "0"))
    if args.output.exists() or args.output.with_suffix(f".rank{rank}.json").exists():
        parser.error("require a new output path")
    selection = json.loads(args.catalog_report.read_bytes())["selection"]
    validate_capacity(args.capacity, selection["capture_sizes"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(
        status="starting",
        rank=rank,
        phase=args.phase,
        capacity=args.capacity,
        scope="synthetic TP2 Down to replicated full output; input copy and reconstruction included; no model/state",
        synthetic_weights=args.weight_index is None,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        catalog_sha256=hashlib.sha256(args.catalog_report.read_bytes()).hexdigest(),
        selection=selection,
    )

    def save():
        args.output.with_suffix(f".rank{rank}.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    save()
    try:
        experiment(args, report, save)
    except Exception as error:
        report.update(status="failed", failed_stage=report["status"], error_type=type(error).__name__, error=str(error))
        save()
        raise


if __name__ == "__main__":
    main()
