# SPDX-License-Identifier: Apache-2.0
"""Auditable workload counts for runner compatibility, with no latency inference."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

QUERY = 4
GDN_LAYERS = 48
VALUE_HEADS = 24
HEAD_WIDTH = 128
FP32_BYTES = 4


def histogram(values):
    return dict(sorted(Counter(values).items()))


def summarize_ledger(report):
    steps = report["generation"]["steps"]
    drafts, accepted, native_rows, capacities, modes = (Counter() for _ in range(5))
    previous, owners = {}, defaultdict(lambda: dict(drafts=Counter(), transitions=Counter()))
    transitions = Counter()
    for step in steps:
        ids, proposals = step["request_ids"], step["proposals"]
        counts = list(map(len, proposals))
        if len(ids) != step["batch"] or len(ids) != len(counts) or len(set(ids)) != len(ids):
            raise ValueError("one unique owner and draft list per live row required")
        if any(not 0 <= count < QUERY for count in counts):
            raise ValueError("the existing verifier permits at most three proposed tokens")
        capacity = step["capacity_tokens"]
        if capacity % QUERY or capacity < QUERY * len(ids):
            raise ValueError("default Q4 capacity must cover the logical request rows")
        if len(step["accepted_from_previous"]) != len(ids) or any(
            not 1 <= value <= QUERY for value in step["accepted_from_previous"]
        ):
            raise ValueError("accepted prefixes and query widths are separate valid fields")
        mode = "all_q1" if not sum(counts) else "all_q4" if all(c == 3 for c in counts) else "mixed_or_short_spec"
        modes[mode] += 1
        native_rows[len(ids) + sum(counts)] += 1
        capacities[capacity] += 1
        accepted.update(step["accepted_from_previous"])
        for owner, count in zip(ids, counts, strict=True):
            drafts[count] += 1
            owners[owner]["drafts"][count] += 1
            request_mode = "spec" if count else "q1"
            if owner in previous and previous[owner] != request_mode:
                transition = previous[owner] + "_to_" + request_mode
                transitions[transition] += 1
                owners[owner]["transitions"][transition] += 1
            previous[owner] = request_mode
    logical = QUERY * sum(drafts.values())
    executed = sum(k * v for k, v in capacities.items())
    proposed = sum(k * v for k, v in drafts.items())
    return dict(
        provenance="counts of an existing real compact generation ledger; native widths are conditional reconstruction",
        native_generation_observation=False,
        steps=len(steps),
        request_steps=sum(drafts.values()),
        draft_counts=dict(sorted(drafts.items())),
        accepted_previous_counts=dict(sorted(accepted.items())),
        real_proposed_candidates=proposed,
        q4_filler=sum((QUERY - 1 - k) * v for k, v in drafts.items()),
        logical_q4_rows=logical,
        native_unpadded_rows=sum(drafts.values()) + proposed,
        executed_q4_graph_rows=executed,
        graph_padding_rows=executed - logical,
        native_query_lengths={k + 1: v for k, v in sorted(drafts.items())},
        native_row_histogram=dict(sorted(native_rows.items())),
        q4_capacity_histogram=dict(sorted(capacities.items())),
        batch_modes=dict(modes),
        request_mode_transitions=dict(transitions),
        per_request=[
            dict(owner=owner, draft_counts=dict(sorted(row["drafts"].items())), transitions=dict(row["transitions"]))
            for owner, row in sorted(owners.items())
        ],
    )


def summarize_metadata(metadata, ledger):
    steps = metadata["steps"]
    if (
        metadata["model_executed"]
        or metadata["native_generation_observation"]
        or metadata["physical_block_ids_observed"]
    ):
        raise ValueError("metadata replay must not be relabeled as native generation")
    if metadata["workload_sha256"] != ledger["workload_sha256"]:
        raise ValueError("metadata and generation must use the same frozen workload")
    if len(steps) != len(ledger["generation"]["steps"]):
        raise ValueError("complete existing ledger metadata required")
    work = Counter()
    for observed, source in zip(steps, ledger["generation"]["steps"], strict=True):
        counts = list(map(len, source["proposals"]))
        if observed["step"] != source["step"]:
            raise ValueError("metadata step identity changed")
        if observed["owners"] != source["request_ids"] or observed["draft_counts"] != counts:
            raise ValueError("metadata replay changed the existing owners or actual drafts")
        widths = [1 + d for d in counts]
        if observed["query_lengths"] != widths or observed["head_rows"] != sum(widths):
            raise ValueError("query/head rows must follow real proposals, never accepted length")
        if observed["capacity"] < sum(widths) or observed["capacity"] not in metadata["effective"]["capture_sizes"]:
            raise ValueError("graph capacity must cover queries using the installed default catalog")
        gdn = observed["gdn"]
        if sum(gdn[k] for k in ("num_spec_tokens", "num_decode_tokens", "num_prefill_tokens")) != sum(widths):
            raise ValueError("GDN processing did not cover each actual query exactly once")
        for key in (
            "num_spec",
            "num_spec_tokens",
            "num_decodes",
            "num_decode_tokens",
            "num_prefills",
            "num_prefill_tokens",
        ):
            work[key] += gdn[key]
    return dict(
        provenance=(
            "installed dispatcher and installed GDN metadata builder on CPU; no model, kernel or physical allocation"
        ),
        effective=metadata["effective"],
        graph_modes=histogram(s["graph_mode"] for s in steps),
        capacity_histogram=histogram(s["capacity"] for s in steps),
        head_row_histogram=histogram(s["head_rows"] for s in steps),
        total_capacity_rows=sum(s["capacity"] for s in steps),
        total_head_rows=sum(s["head_rows"] for s in steps),
        graph_padding_rows=sum(s["capacity"] - sum(s["query_lengths"]) for s in steps),
        gdn_work=dict(work),
        allocation_budget=metadata["native_allocation_budget"],
    )


def conversion_scenario(counts, saved_loop_seconds):
    if saved_loop_seconds <= 0:
        raise ValueError("positive previously observed loop saving required")
    per_owner = GDN_LAYERS * VALUE_HEADS * HEAD_WIDTH**2 * FP32_BYTES
    switches = sum(counts["request_mode_transitions"].values())
    payload = switches * per_owner
    return dict(
        scenario="separate dense/compact storage, one accepted FP32 state transferred per owner mode switch",
        implemented=False,
        applies_to="existing compact ledger reconstruction only, not observed native requests",
        fp32_state_bytes_per_owner_per_rank=per_owner,
        owner_switches=switches,
        state_payload_bytes_per_rank=payload,
        copy_read_write_bytes_per_rank=2 * payload,
        excludes=(
            "flush arithmetic, logs, conv shift, owner metadata, launches, synchronization, initialization and terminal"
        ),
        previous_loop_saving_seconds=saved_loop_seconds,
        break_even_seconds_per_owner_switch=saved_loop_seconds / switches if switches else None,
        required_extra_copy_bandwidth_bytes_per_second=2 * payload / saved_loop_seconds,
        caveat=(
            "sharing storage could change this cost, but requires an explicit consumer/ownership adapter; "
            "no latency measured"
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    ledger = json.loads(args.ledger.read_bytes())
    metadata = json.loads(args.metadata.read_bytes())
    ledger_sha256 = hashlib.sha256(args.ledger.read_bytes()).hexdigest()
    if ledger_sha256 != metadata["source_sha256"]:
        raise ValueError("metadata replay source does not match the exact generation ledger bytes")
    counts = summarize_ledger(ledger)
    result = dict(
        schema="gdn-ready-runner-compatibility-v1",
        workload_sha256=ledger["workload_sha256"],
        ledger_sha256=ledger_sha256,
        metadata_sha256=hashlib.sha256(args.metadata.read_bytes()).hexdigest(),
        ledger_inventory=counts,
        installed_metadata_replay=summarize_metadata(metadata, ledger),
        conversion_scenario=conversion_scenario(counts, 21.325973 - 20.756844),
        decision="defer_native_runner_integration_frozen_q4_contract_incompatible_and_native_capture_failed",
        native_generation_measured=False,
        quality_failures_preserved=True,
    )
    args.root.mkdir(parents=True, exist_ok=True)
    (args.root / "analysis.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "ledger_inventory"}, indent=2))


if __name__ == "__main__":
    main()
