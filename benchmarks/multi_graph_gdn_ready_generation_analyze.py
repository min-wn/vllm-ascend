# SPDX-License-Identifier: Apache-2.0
"""Audit actual emitted work before aggregating three independent process pairs."""

import argparse
import json
import statistics
from pathlib import Path

from multi_graph_gdn_ready_generation import COST_KEYS, MODES, PAIRS, check_pair, run_plan


def analyze(root):
    protocol = json.loads((root / "protocol.json").read_bytes())
    if protocol["plan"] != run_plan():
        raise ValueError("execution order differs from the frozen three pairs")
    reports, pairs = {}, []
    for item in run_plan():
        label = f"pair{item['pair']}.{item['mode']}"
        reports[item["pair"], item["mode"]] = [
            json.loads((root / label / f"result.rank{rank}.json").read_bytes()) for rank in (0, 1)
        ]
    for pair in range(PAIRS):
        delayed, ready = (reports[pair, mode] for mode in MODES)
        check_pair(delayed, ready)
        costs = {mode: reports[pair, mode][0]["critical_cost"] for mode in MODES}
        generation = delayed[0]["generation"]
        pairs.append(
            dict(
                pair=pair,
                costs=costs,
                generation_gain_percent=100 * (1 - costs["ready"]["generation_us"] / costs["delayed"]["generation_us"]),
                batch_gain_percent=100 * (1 - costs["ready"]["batch_us"] / costs["delayed"]["batch_us"]),
                output_tokens=generation["output_tokens"],
                loop_output_tokens=generation["loop_output_tokens"],
                steps=len(generation["steps"]),
                loop_output_tokens_per_second={
                    mode: generation["loop_output_tokens"] * 1e6 / costs[mode]["generation_us"] for mode in MODES
                },
                batch_output_tokens_per_second={
                    mode: generation["output_tokens"] * 1e6 / costs[mode]["batch_us"] for mode in MODES
                },
            )
        )
    reference = reports[0, "delayed"][0]["generation"]
    if any(reports[pair, mode][0]["generation"] != reference for pair in range(PAIRS) for mode in MODES):
        raise ValueError("actual work changed between process repetitions")
    means = {mode: {key: statistics.mean(p["costs"][mode][key] for p in pairs) for key in COST_KEYS} for mode in MODES}
    repeated = {}
    for mode in MODES:
        values = [p["costs"][mode]["generation_us"] for p in pairs]
        repeated[mode] = dict(
            min_us=min(values),
            max_us=max(values),
            relative_range_percent=100 * (max(values) - min(values)) / statistics.median(values),
        )
    gain = 100 * (1 - means["ready"]["generation_us"] / means["delayed"]["generation_us"])
    # A positive pooled mean alone does not justify runner work.
    stable = all(p["generation_gain_percent"] > 0 for p in pairs) and gain > max(
        1.0, *(2 * r["relative_range_percent"] for r in repeated.values())
    )
    acceptance = {str(n): 0 for n in range(1, 5)}
    proposed, matched = 0, 0
    for step in reference["steps"]:
        proposed += sum(map(len, step["proposals"]))
        for decision in step["decisions"]:
            acceptance[str(decision["accepted"])] += 1
            matched += decision["matched_drafts"]
    return dict(
        status="complete_same_algorithm_offline_actual_generation",
        pairs=pairs,
        mean_costs=means,
        generation_gain_percent=gain,
        batch_gain_percent=100 * (1 - means["ready"]["batch_us"] / means["delayed"]["batch_us"]),
        repeat_spread=repeated,
        stable_generation_benefit=stable,
        output_tokens=reference["output_tokens"],
        loop_output_tokens=reference["loop_output_tokens"],
        proposed_draft_tokens=proposed,
        matched_draft_tokens=matched,
        acceptance_histogram=acceptance,
        finish_reasons=[r["finish_reason"] for r in reference["requests"]],
        generated_lengths=[len(r["tokens"]) for r in reference["requests"]],
        boundaries=("batch sums four measured phases; capture/validation gap excluded; not service throughput"),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    result = analyze(args.root)
    path = args.root / "analysis.json"
    if path.exists():
        raise FileExistsError("preserve prior analysis")
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
