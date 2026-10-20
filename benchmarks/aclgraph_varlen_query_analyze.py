# SPDX-License-Identifier: Apache-2.0
"""Audit frozen inputs, measured sources, and a nonoverlapping task inventory."""

import argparse
import collections
import hashlib
import json
import statistics
from pathlib import Path

from aclgraph_varlen_query_trace import validate_trace


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inventory(events):
    if isinstance(events, dict):
        events = events["traceEvents"]
    kernels, dma, waits = [], [], []
    for event in events:
        if event.get("ph") != "X":
            continue
        task = event.get("args", {}).get("Task Type", "")
        if task.startswith("KERNEL_"):
            kernels.append(event)
        elif task == "PCIE_DMA_SQE":
            dma.append(event)
        elif task == "NOTIFY_WAIT":
            # This overlaps work on the graph stream. It is a join span,
            # not additional device work or an independently removable cost.
            waits.append(event)
    groups = collections.defaultdict(list)
    for event in kernels:
        groups[event["name"]].append(event["dur"])
    return dict(
        kernels={
            name: dict(count=len(values), duration_us=values, median_us=statistics.median(values))
            for name, values in groups.items()
        },
        dma=dict(count=len(dma), total_us=sum(e["dur"] for e in dma)),
        waits=dict(count=len(waits), duration_us=[e["dur"] for e in waits], counted_as_extra_work=False),
    )


def analyze(folder, root):
    frozen = validate_trace(json.loads((folder / "input-trace.json").read_bytes()))
    report = json.loads((folder / "probe.json").read_bytes())
    if report["state"] != "completed" or report["trace_sha256"] != frozen["trace_sha256"]:
        raise ValueError("incomplete probe or wrong frozen trace")
    if sha(folder / "input-trace.json") != report["trace_file_sha256"]:
        raise ValueError("input file identity mismatch")
    for name, expected in report["scripts"].items():
        if sha(root / "benchmarks" / name) != expected:
            raise ValueError(f"measured script changed: {name}")
    native = json.loads((folder / "native-audit.json").read_bytes())
    for name, expected in native["probe_helper_hashes_collected_after_run"].items():
        if sha(root / "benchmarks" / name) != expected:
            raise ValueError(f"probe helper changed: {name}")
    native_artifacts = {item["path"]: item["sha256"] for item in native["artifacts"]}
    if native_artifacts[report["opp_root"] + "/op_api/lib/libcust_opapi.so"] != report["opapi_sha256"]:
        raise ValueError("native OpAPI identity mismatch")
    expected_steps = frozen["trace"]["probe_steps"]
    if [c["step"] for c in report["checks"]] != expected_steps or len(report["q4_regression"]) != 128:
        raise ValueError("qualification scope mismatch")
    for check in report["checks"] + report["q4_regression"] + report["post_timing"]:
        if not all(check[k] for k in ("finite", "exact_zero_tail", "stable_addresses")):
            raise ValueError("qualification failed")
    profiles = []
    for evidence in report["profiles"]:
        path = folder / Path(evidence["path"]).name
        if sha(path) != evidence["sha256"]:
            raise ValueError("profile identity mismatch")
        item = inventory(json.loads(path.read_bytes()))
        counts = item["kernels"]
        expected_names = (
            "aclnnInplaceZero_ZerosLikeAiCore_ZerosLike",
            "FusedInferAttentionScoreV2SinkMetadata",
            "aclnnFusedInferAttentionScoreV2SinkV3_FusedInferAttentionScoreV2Sink_FusedInferAttentionScoreV2Sink",
        )
        if set(counts) != set(expected_names) or any(v["count"] != 3 for v in counts.values()):
            raise ValueError("unexpected kernel inventory; reassess instead of silently assigning a budget")
        if item["dma"]["count"] != 9:
            raise ValueError("three required descriptor/table uploads per replay expected")
        profiles.append(dict(step=evidence["step"], sha256=sha(path), **item))
    rounds = []
    for trial in range(4):
        for mode in ("closed", "producer", "consumer", "upload"):
            samples = [s for s in report["samples"] if s["trial"] == trial and s["mode"] == mode]
            if sorted(s["step"] for s in samples) != expected_steps:
                raise ValueError("timing scope mismatch")
            rounds.append(
                dict(
                    trial=trial,
                    mode=mode,
                    **{k: statistics.median(s[k] for s in samples) for k in ("wall_us", "queue_us")},
                )
            )
    trace = frozen["trace"]
    histogram = collections.Counter(q for step in trace["steps"] for q in step["q_lengths"] if q)
    return dict(
        status="screening_closed_no_new_attributed_budget",
        trace_sha256=frozen["trace_sha256"],
        source_hashes_verified=True,
        q_histogram=dict(sorted(histogram.items())),
        q_total=sum(q * count for q, count in histogram.items()),
        all_trajectory_q_range=[
            min(s["live_tokens"] for s in trace["steps"]),
            max(s["live_tokens"] for s in trace["steps"]),
        ],
        checked_variable_steps=expected_steps,
        checked_q4_steps=len(report["q4_regression"]),
        profiles=profiles,
        endpoint_medians=report["summary"],
        rounds=rounds,
        gate=trace["predeclared_gate"],
        decision=dict(
            new_repeated_independent_preparation_kernels_observed=0,
            fused_output_initialization_cost_us=None,
            reason=(
                "output initialization and its barrier are inside the fused kernel; no causal removable budget measured"
            ),
            shared_metadata_producer_excluded="already once per step and previous candidate closed",
            note=(
                "absence of an attributed budget does not prove fused initialization costs zero or cannot be optimized"
            ),
            candidate_implemented=False,
            full_model_gain_established=False,
        ),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    result = analyze(args.folder, root)
    target = args.folder / "analysis.json"
    target.write_bytes((json.dumps(result, indent=2) + "\n").encode())
    print(json.dumps(dict(status=result["status"], endpoints=result["endpoint_medians"]), indent=2))


if __name__ == "__main__":
    main()
