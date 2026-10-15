from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any


def inspect_device_trace(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    events = payload if isinstance(payload, list) else payload["traceEvents"]
    device_pids = {
        event["pid"]
        for event in events
        if event.get("name") == "process_name" and event.get("args", {}).get("name") == "Ascend Hardware"
    }
    durations = [
        event
        for event in events
        if event.get("pid") in device_pids and event.get("ph") == "X" and float(event.get("dur", 0)) > 0
    ]
    if not durations:
        raise ValueError("profiler trace contains no Ascend device execution events")
    return {"device_events": len(durations), "device_streams": sorted({event["tid"] for event in durations})}


def summarize(values: list[float]) -> dict[str, float | int]:
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("measurements must be nonempty, positive and finite")
    ordered = sorted(values)

    def percentile(fraction):
        position = (len(ordered) - 1) * fraction
        lower = math.floor(position)
        upper = math.ceil(position)
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)

    mean = statistics.mean(values)
    deviation = statistics.stdev(values) if len(values) > 1 else 0.0
    return {
        "count": len(values),
        "mean": mean,
        "min": ordered[0],
        "max": ordered[-1],
        "p50": percentile(0.5),
        "p90": percentile(0.9),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "stdev": deviation,
        "coefficient_of_variation": deviation / mean,
    }


def resolve_workload(args) -> None:
    if args.batch_size < 2:
        raise ValueError("batch-size must be at least two")
    if args.single_capacity is None:
        args.single_capacity = 2 ** (args.batch_size - 1).bit_length()
    args.lane_sizes = args.lane_sizes or [(args.batch_size + 1) // 2, args.batch_size // 2]
    if args.single_capacity < args.batch_size:
        raise ValueError("single-capacity must cover the actual batch")
    if len(args.lane_sizes) != 2 or min(args.lane_sizes) <= 0 or sum(args.lane_sizes) != args.batch_size:
        raise ValueError("two positive lane sizes must sum to batch-size; split padding is not supported")
    if args.warmups < 1 or args.repeats < 1 or args.output_tokens < 2 or args.prompt_tokens < 1:
        raise ValueError("warmups/repeats/prompt-tokens must be positive; output-tokens must be at least two")
    if args.prompt_spread < 0 or args.graph_budget <= 0 or not 0 < args.gpu_memory_utilization < 1:
        raise ValueError("invalid prompt spread, graph budget or memory utilization")
    args.prompt_lengths = [
        args.prompt_tokens
        + (
            0
            if args.prompt_pattern == "uniform"
            else args.prompt_spread * (index % 2)
            if args.prompt_pattern == "bimodal"
            else args.prompt_spread * index // max(args.batch_size - 1, 1)
        )
        for index in range(args.batch_size)
    ]
    if max(args.prompt_lengths) + args.output_tokens > args.max_model_len:
        raise ValueError("max-model-len must cover every prompt plus output")
    if args.max_batched_tokens is None:
        args.max_batched_tokens = max(args.max_model_len, args.single_capacity, sum(args.prompt_lengths))
    if args.max_batched_tokens < max(args.single_capacity, sum(args.prompt_lengths)):
        raise ValueError("max-batched-tokens must cover the full prefill batch and graph capacity")
    if args.kv_blocks is None:
        args.kv_blocks = max(256, args.single_capacity * math.ceil(args.max_model_len / 128))
    required = sum(math.ceil((length + args.output_tokens) / 128) for length in args.prompt_lengths)
    if args.kv_blocks < required:
        raise ValueError("kv-blocks cannot hold the workload without preemption")


def verify_outputs(results: dict[str, dict[str, Any]], dtype: str) -> dict[str, Any]:
    reference = results["single"]["generation_outputs"]
    eager = results["eager"]["generation_outputs"]
    if not reference or any(not batch for batch in reference):
        raise ValueError("empty correctness baseline")
    eps = 2 ** (-7 if dtype == "bfloat16" else -10)
    errors = {}
    for mode, result in results.items():
        generations = result["generation_outputs"]
        if len(generations) != len(reference) or len(eager) != len(reference):
            raise ValueError("generation repeat counts differ")
        maximum = 0.0
        for baseline, actual, eager_batch in zip(reference, generations, eager):
            if len(baseline) != len(actual) or len(baseline) != len(eager_batch):
                raise ValueError("request counts differ")
            for expected, observed, eager_request in zip(baseline, actual, eager_batch):
                if observed["tokens"] != expected["tokens"]:
                    raise ValueError(f"{mode}: generated tokens differ from single graph")
                lists = [request["selected_logprobs"] for request in (expected, observed, eager_request)]
                if not lists[0] or any(len(values) != len(expected["tokens"]) for values in lists):
                    raise ValueError("missing or mismatched token log probabilities")
                if any(not math.isfinite(value) for values in lists for value in values):
                    raise ValueError("nonfinite token log probability")
                baseline_error = max(abs(a - b) for a, b in zip(lists[0], lists[2]))
                error = max(abs(a - b) for a, b in zip(lists[0], lists[1]))
                if error > baseline_error + eps:
                    raise ValueError(f"{mode}: log probabilities exceed eager/single baseline tolerance")
                maximum = max(maximum, error)
        errors[mode] = maximum
    return {"passed": True, "all_repeats_checked": True, "maximum_logprob_error_vs_single": errors}


def verify_decode_steps(
    steps: list[dict], mode: str, batch: int, capacity: int, lanes: list[int], expected_steps: int | None = None
) -> None:
    if not steps:
        raise ValueError("no decode steps observed")
    if expected_steps is not None and len(steps) != expected_steps:
        raise ValueError("decode step coverage is incomplete")
    for step in steps:
        if step["actual_requests"] != batch:
            raise ValueError("decode batch changed; the requested comparison is invalid")
        if mode in ("serial", "parallel"):
            if step["lane_sizes"] != lanes or step["single_replays"] or step["fallback"]:
                raise ValueError("split decode missed its intended lane graphs")
        elif mode == "single":
            if step["single_replays"] != [capacity] or step["lane_sizes"] is not None:
                raise ValueError("single decode did not replay the intended padding graph")
        elif step["single_replays"] or step["lane_sizes"] is not None:
            raise ValueError("eager control unexpectedly replayed a graph")
