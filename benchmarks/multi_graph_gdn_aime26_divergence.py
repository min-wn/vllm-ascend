# SPDX-License-Identifier: Apache-2.0
"""Align the frozen question-28 outputs and extract only its first-divergence replay."""

import argparse
import hashlib
import json
from pathlib import Path

from multi_graph_gdn_aime26 import QUERY, GenerationRequest, load_manifest

QUESTION = 27


def common_prefix(left, right):
    return next((i for i, (a, b) in enumerate(zip(left, right)) if a != b), min(len(left), len(right)))


def reconstruct(rows, generations, journal, capacity, target_offset):
    requests = {r["index"]: GenerationRequest(r["index"], tuple(r["prompt_tokens"])) for r in rows}
    for index, request in requests.items():
        request.begin(generations[index]["generated_tokens"][0])
    replay, emission, previous = [], None, None
    for step in journal:
        ids = [0] * capacity
        target = requests[QUESTION]
        before = len(target.tokens)
        for row, index in enumerate(step["question_indices"]):
            request = requests[index]
            if request.processed + QUERY != step["seq_lengths"][row]:
                raise AssertionError("logged position is inconsistent with the actual accepted history")
            if request.accepted != step["accepted_from_previous"][row]:
                raise AssertionError("logged acceptance is inconsistent with actual verification")
            ids[row * QUERY : (row + 1) * QUERY] = request.input_tokens(step["proposals"][row])
        replay.append(
            {
                **{key: step[key] for key in ("step", "batch", "request_ids", "seq_lengths", "accepted_from_previous")},
                "input_ids": ids,
                "expected_predictions": step["predictions"],
            }
        )
        for row, index in enumerate(step["question_indices"]):
            decision = requests[index].advance(
                step["proposals"][row], step["predictions"][row * QUERY : (row + 1) * QUERY]
            )
            if decision != step["decisions"][row]:
                raise AssertionError("verification does not reproduce the immutable journal")
        if before <= target_offset < len(target.tokens):
            row = step["question_indices"].index(QUESTION)
            offset = target_offset - before
            owner = step["request_ids"][row]
            emission = dict(
                step=step["step"],
                batch=step["batch"],
                row=row,
                owner=owner,
                generated_before=before,
                verification_offset=offset,
                input_positions=list(range(step["seq_lengths"][row] - QUERY, step["seq_lengths"][row])),
                predictor_position=step["seq_lengths"][row] - QUERY + offset,
                generated_token_position=step["seq_lengths"][row] - QUERY + offset + 1,
                accepted_from_previous=step["accepted_from_previous"][row],
                accepted_this_step=step["decisions"][row]["accepted"],
                candidate_ids=[1 + QUERY * owner + t for t in range(QUERY)],
                compact_owner_row=owner + 1,
                proposals=step["proposals"][row],
                input_tokens=ids[row * QUERY : (row + 1) * QUERY],
                effective_tokens=step["batch"] * QUERY,
                capacity_tokens=capacity,
                active_question_indices=step["question_indices"],
                seq_lengths=step["seq_lengths"],
                batch_changed=previous is not None and previous["request_ids"] != step["request_ids"],
                exiting_this_step=[
                    i for i, d in zip(step["question_indices"], step["decisions"], strict=True) if d["finish_reason"]
                ],
            )
            break
        previous = step
    if emission is None:
        raise ValueError("first divergence is not covered by a decode verification in this journal")
    if len(replay) > 1024:
        raise ValueError("this bounded diagnosis does not authorize a long generation rerun")
    return replay, emission


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "alignment.json").exists():
        parser.error("preserve the first extracted diagnostic trace")
    manifest = load_manifest(args.evidence / "inputs/manifest.json")
    formal = args.evidence / "formal"
    records = {v: json.loads((formal / f"{v}.question27.json").read_bytes()) for v in ("official", "compact")}
    tokens = {v: r["generated_tokens"] for v, r in records.items()}
    lcp = common_prefix(tokens["official"], tokens["compact"])
    if lcp == min(map(len, tokens.values())):
        raise ValueError("no different token to diagnose")
    traces, sites = {}, {}
    for variant in records:
        group = json.loads((formal / f"{variant}.shard1.group3.result.json").read_bytes())
        rows = [manifest["payload"]["inputs"][i] for i in group["indices"]]
        outputs = {i: json.loads((formal / f"{variant}.question{i:02}.json").read_bytes()) for i in group["indices"]}
        with (formal / f"{variant}.shard1.group3.steps.jsonl").open(encoding="utf-8") as stream:
            trace, site = reconstruct(
                rows, outputs, (json.loads(line) for line in stream), group["capacity_tokens"], lcp
            )
        traces[variant], sites[variant] = trace, site
        replay = dict(
            schedule_source=variant,
            manifest_sha256=manifest["payload_sha256"],
            purpose="bounded forced-prefix arithmetic/scheduling diagnosis; not task evaluation",
            rows=rows,
            steps=trace,
            site=site,
            original_first_tokens={i: outputs[i]["generated_tokens"][0] for i in outputs},
        )
        (args.output / f"{variant}-trace.json").write_text(
            json.dumps(replay, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
        )
    own_history_equal = len(traces["official"]) == len(traces["compact"])
    if own_history_equal:
        for a, b in zip(traces["official"], traces["compact"], strict=True):
            row = sites["official"]["row"]
            own_history_equal &= all(a[k][row] == b[k][row] for k in ("seq_lengths", "accepted_from_previous"))
            own_history_equal &= (
                a["input_ids"][row * QUERY : (row + 1) * QUERY] == b["input_ids"][row * QUERY : (row + 1) * QUERY]
            )
    result = dict(
        question_id=28,
        manifest_sha256=manifest["payload_sha256"],
        common_output_tokens=lcp,
        first_different_output_0based=lcp,
        first_different_output_1based=lcp + 1,
        prompt_tokens=records["official"]["prompt_tokens"],
        first_tokens={v: t[lcp] for v, t in tokens.items()},
        token_context={v: t[max(0, lcp - 20) : lcp + 21] for v, t in tokens.items()},
        sites=sites,
        own_q4_inputs_positions_and_acceptance_equal_before_divergence=bool(own_history_equal),
        raw_output_sha256={
            v: hashlib.sha256((formal / f"{v}.question27.json").read_bytes()).hexdigest() for v in records
        },
    )
    (args.output / "alignment.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
