# SPDX-License-Identifier: Apache-2.0
"""Require complete scheduling equivalence before reporting replay savings."""

import argparse
import json
import statistics
from pathlib import Path

from multi_graph_gdn_ready_chain_protocol import LOADS, MODES, PAIRS, critical_cost, run_plan

COMPLETE = "closed_complete_model_replay_pending_cross_process_pair_audit"


def analyze(directory):
    protocol = json.loads((directory / "protocol.json").read_text(encoding="utf-8"))
    if protocol["plan"] != run_plan():
        raise ValueError("independent-process order differs from frozen protocol")
    reports, pairs = {}, []
    for item in run_plan():
        label = f"{item['load']}.pair{item['pair']}.{item['mode']}"
        ranks = [
            json.loads((directory / label / f"result.rank{rank}.json").read_text(encoding="utf-8")) for rank in (0, 1)
        ]
        for rank, row in enumerate(ranks):
            if (
                row["rank"] != rank
                or row["mode"] != item["mode"]
                or row["load"] != item["load"]
                or row["status"] != COMPLETE
                or not row["timed_final_state_exact"]
            ):
                raise ValueError("incomplete or mismatched process-group result")
            if (
                row["head"]["weight_dtype"] != "torch.bfloat16"
                or row["head"]["output_dtype"] != "torch.bfloat16"
                or not row["head"]["actual_projection_audited"]
                or row["head_precision_changes"]
            ):
                raise ValueError("original BF16 head was not qualified")
            if row["trace_sha256"] != protocol["workloads"][item["load"]]["sha256"]:
                raise ValueError("trace changed after protocol freeze")
            if len(row["correctness"]["output_hashes"]) != protocol["workloads"][item["load"]]["steps"]:
                raise ValueError("per-step same-algorithm checks incomplete")
            if row["critical_cost"] != critical_cost(row["cost_ranks"]):
                raise ValueError("critical TP rank or additive cost columns changed")
        if ranks[0]["cost_ranks"] != ranks[1]["cost_ranks"]:
            raise ValueError("both TP ranks must agree on collected cost endpoints")
        reports[item["load"], item["pair"], item["mode"]] = ranks
    for load in LOADS:
        for pair in range(PAIRS):
            delayed, ready = (reports[load, pair, mode] for mode in MODES)
            for rank in (0, 1):
                for key in (
                    "trace_sha256",
                    "prefix",
                    "selection",
                    "load_contract",
                    "terminal_contract",
                    "initial_state_digest",
                    "correctness",
                    "dense_weights",
                    "loader",
                ):
                    # Prefill wall time is evidence, not an input or correctness field.
                    left, right = delayed[rank][key], ready[rank][key]
                    if key == "prefix":
                        left = {k: v for k, v in left.items() if k != "elapsed_seconds"}
                        right = {k: v for k, v in right.items() if k != "elapsed_seconds"}
                    if left != right:
                        raise ValueError(f"{load}/pair{pair}/rank{rank}: {key} differs across schedules")
                if (
                    delayed[rank]["capture"]["required_default_buckets"]
                    != ready[rank]["capture"]["required_default_buckets"]
                ):
                    raise ValueError("capture capacity differs")
            costs = {mode: reports[load, pair, mode][0]["critical_cost"] for mode in MODES}
            pairs.append(
                dict(
                    load=load,
                    pair=pair,
                    costs=costs,
                    execution_percent=100 * (1 - costs["ready"]["execution_us"] / costs["delayed"]["execution_us"]),
                    closed_percent=100 * (1 - costs["ready"]["total_us"] / costs["delayed"]["total_us"]),
                    exact_initial_outputs_and_closed_states=True,
                )
            )
    loads = []
    for load in LOADS:
        subset = [p for p in pairs if p["load"] == load]
        trace_report = reports[load, 0, "delayed"][0]
        # Sum denominators before the percentage; do not average ratios.
        means = {
            mode: {
                key: statistics.mean(p["costs"][mode][key] for p in subset)
                for key in ("execution_us", "initialization_us", "terminal_us", "total_us")
            }
            for mode in MODES
        }
        loads.append(
            dict(
                load=load,
                steps=len(trace_report["selection"]["dispatched_tokens"]),
                capacities=trace_report["selection"]["required_tokens"],
                mean_costs=means,
                closed_percent=100 * (1 - means["ready"]["total_us"] / means["delayed"]["total_us"]),
                pair_percent=[p["closed_percent"] for p in subset],
                both_pairs_benefit=all(p["closed_percent"] > 0 for p in subset),
            )
        )
    return dict(
        scope="fixed-trajectory complete-model execution replay; excludes sampling/draft/service",
        exact_same_algorithm_pair_audit=True,
        independent_pairs_per_load=PAIRS,
        total_unprofiled_process_groups=len(run_plan()),
        pairs=pairs,
        loads=loads,
        quality_claim=False,
        old_numeric_and_quality_failures_unchanged=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.directory)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for row in result["loads"]:
        print(json.dumps(row))


if __name__ == "__main__":
    main()
