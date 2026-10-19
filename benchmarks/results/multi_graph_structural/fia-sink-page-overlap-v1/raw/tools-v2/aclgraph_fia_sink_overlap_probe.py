# SPDX-License-Identifier: Apache-2.0
"""Optimistic sink attention/TP-local FFN overlap, with real descriptor H2D.

FFN inputs are independently ready; no TP collective or model dependency is
claimed. Failure to beat unrestricted serial execution rejects this shape's
resource-complementarity hypothesis before building a model scheduler.
"""

import argparse
import hashlib
import json
import statistics
import time
import traceback
from pathlib import Path

import aclgraph_fia_sink_stage3_budget as base
from aclgraph_fia_sink_stage3_trace import load_trace_schedule
from aclgraph_fia_sink_stage4_prepare import PreparedDescriptors
from aclgraph_fia_sink_stage4_whole_plan import validate_whole_plan


def measurement_order(round_index):
    modes = ("full_serial", "half_serial", "full_parallel", "half_parallel")
    return modes if round_index % 2 == 0 else modes[::-1]


def load_ffn(torch, torch_npu, model):
    from safetensors import safe_open

    index = json.loads((model / "model.safetensors.index.json").read_text())["weight_map"]
    config = json.loads((model / "config.json").read_text())["text_config"]
    if (config["hidden_size"], config["intermediate_size"]) != (5120, 17408):
        raise ValueError("requires the established Qwen3.5-27B FFN geometry")
    weights, hashes = {}, {}
    for name in ("gate", "up", "down"):
        key = f"model.language_model.layers.3.mlp.{name}_proj.weight"
        with safe_open(model / index[key], framework="pt", device="cpu") as handle:
            tensor = handle.get_tensor(key)
        tensor = (tensor[:, :8704] if name == "down" else tensor[:8704]).contiguous()
        hashes[name] = hashlib.sha256(tensor.view(torch.uint8).numpy().tobytes()).hexdigest()
        weights[name] = tensor
    nd = {"gateup": torch.cat((weights["gate"], weights["up"])).npu(), "down": weights["down"].npu()}
    nz = {name: torch_npu.npu_format_cast(value, 29) for name, value in nd.items()}
    if any(torch_npu.get_npu_format(value) != 29 for value in nz.values()):
        raise RuntimeError("NZ weight qualification failed")
    return nd, nz, hashes


class Pair:
    def __init__(self, torch, torch_npu, views, tensors, cores, x, weights, *, half, parallel):
        self.torch = torch
        self.root, self.attn, self.dense = [torch.npu.Stream() for _ in range(3)]
        self.cores = dict(cores)
        if half:
            for stream in (self.attn, self.dense):
                torch.npu.set_stream_limit(stream, cube_num=10, vector_num=20)
            self.cores.update(cube_core_num=10, vector_core_num=20)
        self.graph = torch.npu.NPUGraph()
        self.events = [torch.npu.Event() for _ in range(3)]
        self.completion = torch.npu.Event()

        def attention():
            plan = base.graph_helpers.metadata_call(torch, tensors, self.cores)
            return plan, base.sink_attention(torch, {**tensors, **views[0]}, plan)

        def dense():
            hidden = torch.nn.functional.linear(x, weights["gateup"])
            return torch.nn.functional.linear(torch_npu.npu_swiglu(hidden, dim=-1), weights["down"])

        torch.npu.synchronize()
        with torch.npu.stream(self.attn):
            self.plan, self.output = attention()
        with torch.npu.stream(self.dense):
            self.dense_output = dense()
        torch.npu.synchronize()
        with torch.npu.graph(self.graph, stream=self.root):
            fork, attention_done, dense_done = self.events
            fork.record(self.root)
            self.attn.wait_event(fork)
            with torch.npu.stream(self.attn):
                self.plan, self.output = attention()
                attention_done.record(self.attn)
            self.dense.wait_event(fork if parallel else attention_done)
            with torch.npu.stream(self.dense):
                self.dense_output = dense()
                dense_done.record(self.dense)
            self.root.wait_event(attention_done)
            self.root.wait_event(dense_done)
        torch.npu.synchronize()

    def replay(self):
        self.root.wait_stream(self.torch.npu.current_stream())
        with self.torch.npu.stream(self.root):
            self.graph.replay()
            self.completion.record(self.root)
        self.completion.synchronize()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--opp-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("/models/Qwen3.5-27B-original"))
    parser.add_argument("--order-offset", type=int, choices=(0, 1), default=0)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    target = args.output / "overlap.json"
    status = dict(
        state="running",
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope=__doc__,
        samples=[],
        checks=[],
        full_model=False,
        split_vs_padding=False,
        weights="real layer3 rank0 TP2 FFN shards; one shared NZ copy across modes",
        endpoint="pinned descriptor fill + H2D + device plan + attention/FFN fork/join + final sync",
        omitted="real FFN input dependencies, QKV generation, TP collectives, KV writes, model/serving",
        hypothesis="optimistic feasibility only; negative result rejects co-scheduling at this shape",
    )

    def save():
        target.write_text(json.dumps(status, indent=2) + "\n")

    save()
    try:
        args.opapi_lib = args.opp_root / "op_api/lib/libcust_opapi.so"
        args.extension = Path(json.loads((args.root / "extension-status.json").read_text())["extension_library"])
        torch, cores, handle = base.validation.initialize_native(args, status, save)
        import torch_npu

        torch.set_num_threads(4)
        torch.npu.config.allow_internal_format = True
        tools = Path(__file__).resolve().parent
        schedule = load_trace_schedule(
            tools / "structural-dynamic-trace-v1.json", tools / "q4-tile-full-replay-v8-queue0.rank0.json"
        )
        records = [r for r in schedule["records"] if r["physical_tokens"] == 400][:2]
        if len(records) != 2:
            raise ValueError("need two original same-capacity records")
        status["records"] = records
        with torch.inference_mode():
            layers, storage = base.allocate_inputs(torch, schedule, 1, 1703)
            views, tensors = base.frame_inputs(storage, layers, 400)
            base.prepare(torch, tensors, records[0])
            source = PreparedDescriptors(torch, tensors, records[0])
            torch.npu.synchronize()
            source.complete()
            nd, nz, status["weight_sha256"] = load_ffn(torch, torch_npu, args.model)
            x = torch.randn((400, 5120), dtype=torch.bfloat16, device="npu").mul_(0.1)
            dense_reference = torch.nn.functional.linear(
                torch_npu.npu_swiglu(torch.nn.functional.linear(x, nd["gateup"]), dim=-1), nd["down"]
            )
            torch.npu.synchronize()
            pairs = {
                name: Pair(
                    torch,
                    torch_npu,
                    views,
                    tensors,
                    cores,
                    x,
                    nz,
                    half=name.startswith("half"),
                    parallel=name.endswith("parallel"),
                )
                for name in measurement_order(args.order_offset)
            }
            for record in records:
                source.prepare(record)
                torch.npu.synchronize()
                metadata = base.graph_helpers.metadata_call(torch, tensors, cores)
                reference = base.sink_attention(torch, {**tensors, **views[0]}, metadata)
                torch.npu.synchronize()
                for name, pair in pairs.items():
                    pair.replay()
                    torch.testing.assert_close(pair.output, reference, rtol=0.02, atol=0.002)
                    torch.testing.assert_close(pair.dense_output, dense_reference, rtol=0.02, atol=0.02)
                    validate_whole_plan(record, pair.plan.cpu().tolist(), cores=pair.cores["cube_core_num"])
                    status["checks"].append(dict(mode=name, step=record["step"], passed=True))
                source.complete()
            for iteration in range(4):
                for name in measurement_order(iteration + args.order_offset):
                    for rep in range(13):
                        record = records[rep % 2]
                        start = time.perf_counter_ns()
                        source.prepare(record)
                        pairs[name].replay()
                        source.complete()
                        elapsed = (time.perf_counter_ns() - start) / 1000
                        if rep >= 3:
                            status["samples"].append(dict(round=iteration, mode=name, step=record["step"], us=elapsed))
                    save()
            status["median_us"] = {
                name: statistics.median(s["us"] for s in status["samples"] if s["mode"] == name) for name in pairs
            }
            # Trace separately to establish overlap/interference, not latency statistics.
            for name, pair in pairs.items():
                source.prepare(records[0])
                torch.npu.synchronize()
                with torch_npu.profiler.profile(
                    activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU]
                ) as prof:
                    pair.replay()
                source.complete()
                prof.export_chrome_trace(str(args.output / (name + ".json")))
            source.prepare(records[0])
            torch.npu.synchronize()
            metadata = base.graph_helpers.metadata_call(torch, tensors, cores)
            reference = base.sink_attention(torch, {**tensors, **views[0]}, metadata)
            torch.npu.synchronize()
            for name, pair in pairs.items():
                pair.replay()
                torch.testing.assert_close(pair.output, reference, rtol=0.02, atol=0.002)
                torch.testing.assert_close(pair.dense_output, dense_reference, rtol=0.02, atol=0.02)
                validate_whole_plan(records[0], pair.plan.cpu().tolist(), cores=pair.cores["cube_core_num"])
                status["checks"].append(dict(mode=name, step=records[0]["step"], phase="post", passed=True))
            source.complete()
            status["descriptors"] = source.summary()
            if handle is None:
                raise AssertionError("native library handle lost")
        status.update(state="completed", returncode=0)
    except BaseException:
        status.update(state="failed", returncode=1, error=traceback.format_exc())
        traceback.print_exc()
    save()
    return status["returncode"]


if __name__ == "__main__":
    raise SystemExit(main())
