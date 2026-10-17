"""Optimistic FIA/Dense co-scheduling gate; not a split-vs-padding benchmark.

Inputs to Dense are already ready. Measure whether any resource complementarity
exists before implementing tile-level dependency scheduling. Uses real first-layer
Qwen3-4B BF16 weights, NZ on every Dense path, and default lane capacities 56/48.
"""

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import torch_npu
from safetensors import safe_open


def load_weights(model):
    index = json.loads((model / "model.safetensors.index.json").read_text())["weight_map"]
    result = {}
    for name in ("self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"):
        key = f"model.layers.0.{name}.weight"
        with safe_open(model / index[key], framework="pt", device="cpu") as handle:
            result[name] = handle.get_tensor(key)
    result["gate_up"] = torch.cat([result.pop("mlp.gate_proj"), result.pop("mlp.up_proj")])
    return {name: torch_npu.npu_format_cast(weight.npu(), 29) for name, weight in result.items()}


class Probe:
    def __init__(self, weights, kind, fia_cores, dense_cores):
        self.kind = kind
        self.weights = weights
        self.q = torch.randn((48, 32, 128), dtype=torch.bfloat16, device="npu")
        self.k = torch.randn((192, 128, 8, 128), dtype=self.q.dtype, device="npu")
        self.v = torch.randn_like(self.k)
        self.table = torch.arange(192, dtype=torch.int32, device="npu").view(48, 4)
        self.x = torch.randn((56, 4096 if kind == "o" else 2560), dtype=self.q.dtype, device="npu") * 0.1
        self.fia_output = torch.empty_like(self.q)
        self.kwargs = dict(
            query=self.q,
            key=self.k.view(192, 128, -1),
            value=self.v.view(192, 128, -1),
            block_table=self.table,
            input_layout="TND",
            block_size=128,
            actual_seq_lengths=list(range(1, 49)),
            actual_seq_lengths_kv=[512] * 42 + [0] * 6,
            num_key_value_heads=8,
            num_heads=32,
            scale=128**-0.5,
            sparse_mode=0,
        )
        workspace = torch_npu._npu_fused_infer_attention_score_get_max_workspace(**self.kwargs)
        self.kwargs.update(workspace=workspace, out=[self.fia_output, torch.empty(1, device="npu", dtype=self.q.dtype)])
        self.root, self.attn, self.dense, self.update = [torch.npu.Stream() for _ in range(4)]
        for stream, cores in ((self.attn, fia_cores), (self.update, fia_cores), (self.dense, dense_cores)):
            if cores:
                torch.npu.set_stream_limit(stream, cube_num=cores, vector_num=cores * 2)
        self.graphs = {}
        self.keepalive = []
        self.checks = []
        # Baseline eager accuracy uses original ND views and unconstrained stream.
        self.dense_reference = self.dense_op(nd=True).clone()
        torch.npu.synchronize()
        for _ in range(3):
            with torch.npu.stream(self.attn):
                self.fia()
            with torch.npu.stream(self.dense):
                self.dense_op()
        torch.npu.synchronize()
        for updated in (False, True):
            for parallel in (False, True):
                self.capture(updated, parallel)

    def fia(self):
        torch_npu.npu_fused_infer_attention_score.out(**self.kwargs)

    def dense_op(self, nd=False):
        def weight(name):
            value = self.weights[name]
            return torch_npu.npu_format_cast(value, 2) if nd else value

        if self.kind == "o":
            return F.linear(self.x, weight("self_attn.o_proj"))
        gate_up = F.linear(self.x, weight("gate_up"))
        return F.linear(torch_npu.npu_swiglu(gate_up, dim=-1), weight("mlp.down_proj"))

    def capture(self, updated, parallel):
        graph = torch.npu.NPUGraph()
        fork, attn_done, dense_done = [torch.npu.Event() for _ in range(3)]
        external = torch.npu.ExternalEvent() if updated else None
        self.root.wait_stream(torch.npu.current_stream())
        with torch.npu.graph(graph, stream=self.root):
            fork.record(self.root)
            self.attn.wait_event(fork)
            with torch.npu.stream(self.attn):
                if updated:
                    external.wait(self.attn)
                    external.reset(self.attn)
                    torch.npu.graph_task_group_begin(self.attn)
                self.fia()
                handle = torch.npu.graph_task_group_end(self.attn) if updated else None
                attn_done.record(self.attn)
            self.dense.wait_event(fork if parallel else attn_done)
            with torch.npu.stream(self.dense):
                output = self.dense_op()
                dense_done.record(self.dense)
            self.root.wait_event(attn_done)
            self.root.wait_event(dense_done)
        torch.npu.synchronize()
        name = ("updated" if updated else "fixed") + ("_parallel" if parallel else "_serial")
        self.graphs[name] = (graph, external, handle, output)
        self.keepalive.extend([fork, attn_done, dense_done])

    def replay(self, name):
        graph, external, handle, _ = self.graphs[name]
        with torch.npu.stream(self.root):
            graph.replay()
        if external is not None:
            with torch.npu.stream(self.update):
                torch.npu.graph_task_update_begin(self.update, handle)
                self.fia()
                torch.npu.graph_task_update_end(self.update)
                external.record(self.update)

    def qualify(self):
        for length in (512, 129, 511, 512):
            self.kwargs["actual_seq_lengths_kv"] = [length] * 42 + [0] * 6
            self.fia()
            torch.npu.synchronize()
            expected = self.fia_output.clone()
            torch.npu.synchronize()
            names = list(self.graphs) if length == 512 else [n for n in self.graphs if n.startswith("updated")]
            for name in names:
                self.replay(name)
                torch.npu.synchronize()
                torch.testing.assert_close(self.fia_output[:42], expected[:42], rtol=0.02, atol=0.02)
                torch.testing.assert_close(self.graphs[name][3], self.dense_reference, rtol=0.02, atol=0.02)
                self.checks.append(dict(mode=name, length=length, passed=True))

    def measure(self, rounds, repeats):
        samples = {name: [] for name in self.graphs}
        for iteration in range(rounds):
            names = list(self.graphs)
            if iteration % 2:
                names.reverse()
            for name in names:
                for _ in range(3):
                    self.replay(name)
                torch.npu.synchronize()
                start = time.perf_counter_ns()
                for _ in range(repeats):
                    self.replay(name)
                torch.npu.synchronize()
                samples[name].append((time.perf_counter_ns() - start) / repeats / 1000)
        return dict(samples_us=samples, median_us={name: statistics.median(v) for name, v in samples.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("/models/Qwen3-4B"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--focus-ffn", action="store_true", help="repeat only unrestricted and half-core FFN")
    args = parser.parse_args()
    if args.output.exists() or min(args.rounds, args.repeats) < 1:
        parser.error("new output path and positive measurement counts required")
    torch.npu.set_device(0)
    torch.npu.config.allow_internal_format = True
    torch.manual_seed(42)
    weights = load_weights(args.model)
    assert all(torch_npu.get_npu_format(w) == 29 for w in weights.values())
    assert weights["self_attn.o_proj"].shape == (2560, 4096)
    assert weights["gate_up"].shape == (19456, 2560)
    assert weights["mlp.down_proj"].shape == (2560, 9728)
    limits = torch.npu.get_device_limit(0)
    cores = limits["cube_core_num"]
    allocations = [(0, 0), (cores // 2, cores - cores // 2)]
    if args.focus_ffn:
        allocations.reverse()
    result = dict(
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        limits=limits,
        scope=(
            "Optimistic independent ready inputs; hot first-layer weights; "
            "capacity56 Dense and capacity48 FIA with42 live. Not model replay or split/padding speedup."
        ),
        rows=[],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    retained_probes = []
    for kind in ("ffn",) if args.focus_ffn else ("o", "ffn"):
        for fia_cores, dense_cores in allocations:
            torch.manual_seed(42)
            print(f"START {kind} FIA={fia_cores} Dense={dense_cores}", flush=True)
            probe = Probe(weights, kind, fia_cores, dense_cores)
            probe.qualify()
            row = dict(
                kind=kind,
                fia_cores=fia_cores,
                dense_cores=dense_cores,
                checks=probe.checks,
                **probe.measure(args.rounds, args.repeats),
            )
            result["rows"].append(row)
            args.output.write_text(json.dumps(result, indent=2))
            print(json.dumps({k: v for k, v in row.items() if k not in ("checks", "samples_us")}), flush=True)
            if args.trace:
                trace_dir = args.output.parent / f"trace-{kind}-{fia_cores}-{dense_cores}"
                with torch_npu.profiler.profile(
                    activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
                    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(trace_dir)),
                    record_shapes=True,
                ):
                    for name in probe.graphs:
                        with torch.profiler.record_function(name):
                            probe.replay(name)
                            torch.npu.synchronize()
            retained_probes.append(probe)
            torch.npu.synchronize()


if __name__ == "__main__":
    with torch.inference_mode():
        main()
