# SPDX-License-Identifier: Apache-2.0
"""Reconstruct actual greedy decisions and each independent continuous history."""

import argparse
import json
from pathlib import Path

from multi_graph_gdn_aime26 import MAX_MODEL_LEN, GenerationRequest, load_manifest


class HistoryNgrams:
    """Earliest-match oracle independent of the installed reversed-KMP proposer."""

    def __init__(self, tokens):
        self.tokens = []
        self.first = {n: {} for n in range(2, 5)}
        self.append(tokens)

    def append(self, tokens):
        previous = len(self.tokens)
        self.tokens.extend(tokens)
        for width, lookup in self.first.items():
            for index in range(max(0, previous - width + 1), len(self.tokens) - width + 1):
                lookup.setdefault(tuple(self.tokens[index : index + width]), index)

    def propose(self):
        for width in range(4, 1, -1):
            if len(self.tokens) < width:
                continue
            start = self.first[width].get(tuple(self.tokens[-width:]), len(self.tokens))
            if start < len(self.tokens) - width:
                return self.tokens[start + width : start + width + 3]
        return []


def audit_group(manifest, result_path):
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result["status"] != "complete" or result["fallback_calls"] or len(result["q4_layer_calls"]) != 48:
        raise ValueError("a completed 48-layer Q4 group without fallback is required")
    directory = result_path.parent
    variant = result["variant"]
    records = {
        i: json.loads((directory / f"{variant}.question{i:02}.json").read_text(encoding="utf-8"))
        for i in result["indices"]
    }
    requests, ngrams = [], []
    for index in result["indices"]:
        record = records[index]
        if record["manifest_sha256"] != manifest["payload_sha256"] or record["finish_reason"] == "runtime_error":
            raise ValueError("missing or failed independent generation")
        request = GenerationRequest(index, tuple(manifest["payload"]["inputs"][index]["prompt_tokens"]))
        request.begin(record["generated_tokens"][0])
        requests.append(request)
        ngrams.append(HistoryNgrams([*request.prompt, *request.tokens]))
    step_count, accepted = 0, {n: 0 for n in range(1, 5)}
    journal = result_path.with_name(result_path.name.replace(".result.json", ".steps.jsonl"))
    with journal.open(encoding="utf-8") as stream:
        for line in stream:
            step = json.loads(line)
            active = [i for i, r in enumerate(requests) if not r.finish_reason]
            if step["step"] != step_count or step["request_ids"] != active or step["batch"] != len(active):
                raise AssertionError("actual active owner order changed")
            if step["question_indices"] != [requests[o].index for o in active]:
                raise AssertionError("question identity was confused with local owner")
            if step["accepted_from_previous"] != [requests[o].accepted for o in active]:
                raise AssertionError("next-state acceptance did not come from the preceding actual validation")
            if step["seq_lengths"] != [requests[o].processed + 4 for o in active]:
                raise AssertionError("FIA/KV positions did not follow actual accepted lengths")
            if max(step["seq_lengths"]) > MAX_MODEL_LEN or step["q4_calls"] != 48 or step["fallback_calls"]:
                raise AssertionError("Q4 invocation or context boundary contract failed")
            for row, owner in enumerate(active):
                proposal = ngrams[owner].propose()
                if proposal != step["proposals"][row]:
                    raise AssertionError("proposal is not the earliest longest ngram of this request's own history")
                decision = requests[owner].advance(proposal, step["predictions"][row * 4 : (row + 1) * 4])
                if decision != step["decisions"][row]:
                    raise AssertionError("accepted outputs differ from the greedy target validation")
                ngrams[owner].append(decision["tokens"])
                accepted[decision["accepted"]] += 1
            step_count += 1
    if step_count != result["steps"] or result["q4_layer_calls"] != [step_count] * 48:
        raise AssertionError("recurrence call counts or complete journal missing")
    for request in requests:
        record = records[request.index]
        if (
            request.tokens != record["generated_tokens"]
            or request.finish_reason != record["finish_reason"]
            or request.processed != record["processed"]
            or request.accepted != record["final_accepted"]
            or request.q4_steps != record["q4_steps"]
            or not request.finish_reason
        ):
            raise AssertionError("final output/state lifecycle cannot be reconstructed from actual decisions")
    if result["terminal"]["selectors"] != [r.accepted for r in requests]:
        raise AssertionError("terminal drain did not select each final actual accepted prefix")
    return dict(
        variant=variant,
        group=result["group"],
        questions=len(requests),
        steps=step_count,
        q4_calls=step_count * 48,
        accepted_counts=accepted,
        passed=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    paths = sorted(args.generations.glob("*.group*.result.json"))
    if len(paths) != 8:
        parser.error("all four groups from both controls must complete")
    groups = [audit_group(manifest, p) for p in paths]
    if any(sum(r["questions"] for r in groups if r["variant"] == variant) != 30 for variant in ("official", "compact")):
        raise ValueError("all 30 actual generations required per control")
    args.output.write_text(json.dumps(dict(groups=groups, passed=True), indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(passed=True, groups=len(groups), q4_calls=sum(r["q4_calls"] for r in groups))))


if __name__ == "__main__":
    main()
