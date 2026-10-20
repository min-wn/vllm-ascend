"""Trace-derived release-window gate with measured FFN stages, not model timing."""

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import torch_npu
from multi_graph_resource_probe import load_weights


def release_windows(document):
    rows = []
    for row in document["trace_analysis"]:
        if row["marker"] != "layer.split_parallel":
            continue
        lanes = sorted(row["attention_intervals"], key=lambda x: x["end_us"])
        if len(lanes) != 2:
            raise ValueError("two attention lanes required")
        rows.append(dict(first_stream=lanes[0]["stream"], gap_us=lanes[1]["end_us"] - lanes[0]["end_us"]))
    if not rows:
        raise ValueError("no split trace observations")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refuse output overwrite")
    windows = release_windows(json.loads(args.trace_report.read_text()))
    torch.npu.set_device(0)
    torch.npu.config.allow_internal_format = True
    torch.manual_seed(42)
    weights = load_weights(Path("/models/Qwen3-4B"))
    x = torch.randn((104, 2560), device="npu", dtype=torch.bfloat16) * 0.1
    weights_nd = {name: torch_npu.npu_format_cast(w, 2) for name, w in weights.items()}

    def gate(value, w=weights):
        return torch_npu.npu_swiglu(F.linear(value, w["gate_up"]), dim=-1)

    def down(value, w=weights):
        return F.linear(value, w["mlp.down_proj"])

    graphs = {}
    checks = []
    buffers = []
    for rows in (104, 56, 48):
        value = x[:rows]
        intermediate = gate(value)
        for stage, operation, reference in (
            ("gate", lambda value=value: gate(value), gate(value, weights_nd)),
            ("down", lambda intermediate=intermediate: down(intermediate), down(intermediate, weights_nd)),
            ("ffn", lambda value=value: down(gate(value)), down(gate(value, weights_nd), weights_nd)),
        ):
            torch.npu.synchronize()
            for _ in range(3):
                operation()
            torch.npu.synchronize()
            stream = torch.npu.Stream()
            stream.wait_stream(torch.npu.current_stream())
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph, stream=stream):
                output = operation()
            torch.npu.synchronize()
            graph.replay()
            torch.npu.synchronize()
            torch.testing.assert_close(output, reference, rtol=0.02, atol=0.02)
            name = f"{stage}{rows}"
            checks.append(name)
            graphs[name] = (graph, stream)
            buffers.extend([value, intermediate, output, reference])
    samples = {name: [] for name in graphs}
    for iteration in range(3):
        names = list(graphs)
        if iteration % 2:
            names.reverse()
        for name in names:
            graph, stream = graphs[name]
            for _ in range(3):
                graph.replay()
            torch.npu.synchronize()
            start = time.perf_counter_ns()
            for _ in range(32):
                graph.replay()
            torch.npu.synchronize()
            samples[name].append((time.perf_counter_ns() - start) / 32000)
    times = {name: statistics.median(v) for name, v in samples.items()}
    predictions = []
    for window in windows:
        delta = window["gap_us"]
        for early, late in ((56, 48), (48, 56)):
            # Optimistic release model, not a rigorous lower bound: no live
            # attention interference, O/norm, assembly copies or kernel fusion.
            joined = delta + times["ffn104"]
            split = max(times[f"ffn{early}"], delta) + times[f"ffn{late}"]
            shared_down = max(times[f"gate{early}"], delta) + times[f"gate{late}"] + times["down104"]
            predictions.append(
                dict(
                    early_rows=early,
                    late_rows=late,
                    gap_us=delta,
                    join_us=joined,
                    split_ffn_us=split,
                    split_gate_shared_down_us=shared_down,
                    predicted_split_ffn_saving_us=joined - split,
                    predicted_shared_down_saving_us=joined - shared_down,
                )
            )
    result = dict(
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        trace_sha256=hashlib.sha256(args.trace_report.read_bytes()).hexdigest(),
        windows=windows,
        checks=checks,
        samples_us=samples,
        median_us=times,
        predictions=predictions,
        limitations=(
            "Analytical optimistic release model, no synthetic device sleep. "
            "Both early-lane orientations are evaluated; stream IDs do not identify request capacity. "
            "Hot weights and stage boundaries differ from a real fused graph. Predictions are not speedups."
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    with torch.inference_mode():
        main()
