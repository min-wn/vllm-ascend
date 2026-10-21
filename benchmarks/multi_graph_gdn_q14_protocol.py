# SPDX-License-Identifier: Apache-2.0
"""Frozen single-layer Q1/Q4 state contract; no model or device imports.

The scalar oracle checks ownership and deferred-log semantics in Python double
arithmetic. It is not an oracle for device rounding or model numerical quality.
"""

import argparse
import copy
import hashlib
import json
import math
import statistics
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

SCHEMA = "gdn-compact-q14-single-layer-v1"
QUERY_LENGTHS = (1, 4)
SEGMENT_STEPS = 16
TRANSITIONS = {(1, 1), (1, 4), (4, 1), (4, 4)}
FROZEN_LEDGER_SHA256 = "4db07f4e6ebcd704ad462231105052e91e126a43602c7143d4cbf5edd8162588"
FROZEN_WORKLOAD_SHA256 = "520c70cd7415449c2e8b53b78b8492a1e130b1c544d5a31a6bb3f3c35cb73181"


def digest(value):
    # JSON object keys become strings on disk; normalize before sorting owners.
    value = json.loads(json.dumps(value, ensure_ascii=False))
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


def settings():
    return dict(
        schema=SCHEMA,
        segment_steps=SEGMENT_STEPS,
        selection="first chronological 16-step window with all four owner mode transitions and a mixed step",
        selection_uses_performance=False,
        query_lengths=list(QUERY_LENGTHS),
        max_buffer_tokens_per_request=4,
        previous_acceptance="independent of current query; must not exceed the previous published log length",
        state="one shared compact FP32 base/log per stable owner; no dense transition bridge",
        resources=dict(aic=20, aiv=40, producer_columns_per_key_head=8),
        input_mode="real_site_reuse",
        initial_state_matches_original_ledger_step=False,
        initial_state="same saved real-layer accepted state materialized once; no pending log at segment entry",
        activation_scope="reuse saved real single-layer sites; not original consecutive trajectory activations",
        baseline="frozen original forced-Q4 compact ready",
        candidate="independent Q1/Q4 compact prototype; delayed and ready are matched schedules",
        native="same actual query widths, installed recurrence/chunk branch; independent continuous dense state",
        precision="BF16 recurrence operands/output; FP32 compact base/log/products; no new precision scheme",
        scope="single-layer prototype, not runner integration or service throughput",
        quality_failures_preserved=True,
        performance_screen=dict(
            rounds=3,
            alternating_order=True,
            all_rounds_positive=True,
            mean_improvement_percent_gt=1.0,
            improvement_gt_twice_larger_relative_repeat_span=True,
        ),
    )


def query_lengths(step):
    owners, proposals = step["request_ids"], step["proposals"]
    if len(owners) != len(proposals) or len(owners) != step["batch"] or len(set(owners)) != len(owners):
        raise ValueError("one unique owner and proposal vector per request required")
    lengths = [1 + len(tokens) for tokens in proposals]
    if any(q not in QUERY_LENGTHS for q in lengths):
        raise ValueError("this prototype supports actual Q1/Q4 only, never Q2/Q3")
    return lengths


def validate_contract(owners, lengths, accepted, previous_widths=None):
    if len(owners) != len(lengths) or len(owners) != len(accepted) or len(set(owners)) != len(owners):
        raise ValueError("one unique owner, current width and previous acceptance per live row required")
    if any(type(owner) is not int or owner < 0 for owner in owners):
        raise ValueError("fixture owners must be nonnegative integer identities")
    if any(type(q) is not int or q not in QUERY_LENGTHS for q in lengths):
        raise ValueError("actual current query must be Q1 or Q4")
    if any(type(a) is not int or not 1 <= a <= 4 for a in accepted):
        raise ValueError("previous accepted prefix must be 1..4")
    if previous_widths is not None:
        for owner, count in zip(owners, accepted, strict=True):
            if owner in previous_widths and count > previous_widths[owner]:
                raise ValueError("acceptance exceeds this owner's previous published log length")


def metadata(owners, lengths, accepted, capacity_requests=None):
    """Packed current inputs plus fixed maximum backing storage; no Q1 filler.

    The owner comes from ids[current_begin], never ids[begin+accepted-1]: an
    accepted prefix of four may precede a current query of one.
    """
    validate_contract(owners, lengths, accepted)
    capacity_requests = len(owners) if capacity_requests is None else capacity_requests
    if type(capacity_requests) is not int or not len(owners) <= capacity_requests <= 128:
        raise ValueError("maximum request capacity must cover live rows and remain <=128")
    tokens = sum(lengths)
    padding_requests = capacity_requests - len(owners)
    packed_ids = [owner + 1 for owner, q in zip(owners, lengths, strict=True) for _ in range(q)]
    return dict(
        sequence=[0, *lengths, *([0] * padding_requests)],
        ids=[*packed_ids, *([0] * (4 * capacity_requests - tokens))],
        accepted=[*accepted, *([1] * padding_requests)],
        live_requests=len(owners),
        actual_query_tokens=tokens,
        buffer_rows=4 * capacity_requests,
        inactive_buffer_rows=4 * capacity_requests - tokens,
        forced_q4_filler_tokens=4 * len(owners) - tokens,
        candidate_filler_tokens=0,
        producer_columns_per_key_head=8,
        useful_producer_columns_per_request=[2 * q for q in lengths],
    )


def summarize_steps(steps):
    previous, transitions, widths, accepted_counts, modes = {}, Counter(), Counter(), Counter(), Counter()
    total_capacity, total_tokens, total_request_steps = 0, 0, 0
    for step in steps:
        owners = step["request_ids"]
        current = query_lengths(step)
        accepted = step["accepted_from_previous"]
        validate_contract(owners, current, accepted, previous)
        for owner, q in zip(owners, current, strict=True):
            if owner in previous:
                transitions[(previous[owner], q)] += 1
            previous[owner] = q
        widths.update(current)
        accepted_counts.update(accepted)
        modes["q1" if all(q == 1 for q in current) else "q4" if all(q == 4 for q in current) else "mixed"] += 1
        total_tokens += sum(current)
        total_request_steps += len(owners)
        total_capacity += step["capacity_tokens"]
    return dict(
        steps=len(steps),
        request_steps=total_request_steps,
        query_length_counts=dict(sorted(widths.items())),
        previous_acceptance_counts=dict(sorted(accepted_counts.items())),
        transitions={f"q{a}_to_q{b}": transitions[a, b] for a, b in sorted(TRANSITIONS)},
        step_modes=dict(modes),
        actual_query_tokens=total_tokens,
        forced_q4_logical_tokens=4 * total_request_steps,
        forced_q4_filler_tokens=4 * total_request_steps - total_tokens,
        original_graph_capacity_rows=total_capacity,
        original_graph_padding_rows=total_capacity - 4 * total_request_steps,
        no_latency_inferred_from_token_counts=True,
    )


def freeze_ledger(ledger, ledger_sha256):
    if ledger_sha256 != FROZEN_LEDGER_SHA256 or ledger["workload_sha256"] != FROZEN_WORKLOAD_SHA256:
        raise ValueError("use the unchanged saved short-text ledger")
    source = ledger["generation"]["steps"]
    selected, summary = None, None
    for start in range(len(source) - SEGMENT_STEPS + 1):
        window = source[start : start + SEGMENT_STEPS]
        if [s["step"] for s in window] != list(range(window[0]["step"], window[0]["step"] + SEGMENT_STEPS)):
            raise ValueError("existing ledger must have consecutive chronological step identities")
        observed = summarize_steps(window)
        if all(observed["transitions"].values()) and observed["step_modes"].get("mixed", 0):
            selected, summary = copy.deepcopy(window), observed
            break
    if selected is None:
        raise ValueError("saved ledger has no eligible continuous mixed segment")
    capacity = max(s["batch"] for s in selected)
    for row in selected:
        row["query_lengths"] = query_lengths(row)
        row["actual_query_tokens"] = sum(row["query_lengths"])
    terminal = {}
    for row in selected:
        for owner, q, decision in zip(row["request_ids"], row["query_lengths"], row["decisions"], strict=True):
            count = decision["accepted"]
            if not 1 <= count <= q:
                raise ValueError("terminal acceptance includes unproposed filler")
            terminal[owner] = count
    payload = dict(
        settings=settings(),
        source_ledger_sha256=ledger_sha256,
        source_workload_sha256=ledger["workload_sha256"],
        first_step=selected[0]["step"],
        last_step=selected[-1]["step"],
        capacity_requests=capacity,
        stable_owners=sorted({owner for row in selected for owner in row["request_ids"]}),
        initial_previous_log_lengths={owner: 0 for owner in selected[0]["request_ids"]},
        initial_acceptance_semantics="no pending log: segment-entry accepted metadata does not reapply any old update",
        steps=selected,
        terminal_accepted={str(owner): count for owner, count in sorted(terminal.items())},
        summary=summary,
    )
    return dict(payload=payload, payload_sha256=digest(payload))


def load_manifest(path):
    document = json.loads(Path(path).read_bytes())
    if document["payload_sha256"] != digest(document["payload"]):
        raise ValueError("frozen mixed segment changed")
    payload = document["payload"]
    if payload["settings"] != settings() or payload["source_ledger_sha256"] != FROZEN_LEDGER_SHA256:
        raise ValueError("prototype contract differs from frozen protocol")
    if payload["summary"] != json.loads(json.dumps(summarize_steps(payload["steps"]))):
        raise ValueError("frozen metadata counts changed")
    return document


@dataclass
class ScalarCompact:
    """One scalar base plus unpublished candidate updates; logical CPU oracle."""

    base: float
    logs: list = field(default_factory=list)

    def accepted_state(self, accepted):
        if type(accepted) is not int or not 1 <= accepted <= 4:
            raise ValueError("previous accepted prefix must be 1..4")
        if self.logs and accepted > len(self.logs):
            raise ValueError("acceptance exceeds previous published log length")
        state = self.base
        for gamma, key, delta in self.logs[:accepted]:
            state = gamma * state + key * delta
        return state

    def step(self, query, key, value, beta, gate, accepted, *, schedule="ready"):
        current = len(query)
        if current not in QUERY_LENGTHS or any(len(x) != current for x in (key, value, beta, gate)):
            raise ValueError("one set of inputs for actual Q1/Q4 only")
        if schedule not in ("delayed", "ready"):
            raise ValueError("only matched delayed/ready schedules")
        old_accepted = self.accepted_state(accepted)
        state, logs, outputs = old_accepted, [], []
        for q, k, v, b, g in zip(query, key, value, beta, gate, strict=True):
            gamma = math.exp(g)
            delta = b * (v - gamma * state * k)
            state = gamma * state + k * delta
            logs.append((gamma, k, delta))
            outputs.append(q * state)
        # Current candidate logs are relative to exactly the accepted old base.
        self.base, self.logs = old_accepted, logs
        return outputs

    def terminal(self, accepted):
        self.base = self.accepted_state(accepted)
        self.logs.clear()
        return self.base


def stability_screen(candidate, baseline):
    """Frozen minimal three-round screening rule, never a service claim."""
    if len(candidate) != 3 or len(baseline) != 3:
        raise ValueError("exactly three paired round endpoints required")
    if any(not math.isfinite(x) or x <= 0 for x in [*candidate, *baseline]):
        raise ValueError("finite positive complete lifecycle endpoints required")
    gains = [100 * (b - c) / b for c, b in zip(candidate, baseline, strict=True)]
    improvement = 100 * (sum(baseline) - sum(candidate)) / sum(baseline)
    spans = {
        "candidate": 100 * (max(candidate) - min(candidate)) / statistics.median(candidate),
        "baseline": 100 * (max(baseline) - min(baseline)) / statistics.median(baseline),
    }
    return dict(
        paired_improvement_percent=gains,
        pooled_improvement_percent=improvement,
        relative_repeat_span_percent=spans,
        stable=all(gain > 0 for gain in gains) and improvement > 1 and improvement > 2 * max(spans.values()),
        scope="single-layer screening only; not online service or whole-model qualification",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("preserve the frozen selection; do not silently replace it")
    raw = args.ledger.read_bytes()
    document = freeze_ledger(json.loads(raw), hashlib.sha256(raw).hexdigest())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: document["payload"][key] for key in ("first_step", "last_step", "summary")}, indent=2))


if __name__ == "__main__":
    main()
