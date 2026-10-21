# SPDX-License-Identifier: Apache-2.0
"""Score the three frozen B1/Q28 paths without running another model."""

import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path

from multi_graph_gdn_aime26 import load_evalscope_rules, load_manifest, score_output
from multi_graph_gdn_aime26_audit import audit_group
from multi_graph_gdn_aime26_divergence import common_prefix
from multi_graph_gdn_aime26_fp32_head import real_projection
from multi_graph_gdn_aime26_fp32_head_analyze import emission_site


def audit_head(head, dtype, steps):
    if (
        head["calls"] != steps + 1
        or head["fallback_calls"]
        or head["input_dtype"] != "torch.bfloat16"
        or head["weight_dtype"] != "torch.bfloat16"
        or head["configured_head_dtype"] != dtype
        or head["output_dtype"] != dtype
        or head["allow_hf32"]
        or head["float32_matmul_precision"] != "highest"
        or [item["rows"] for item in head["audited_calls"]] != [1, 4]
        or not all(real_projection(item["operations"], dtype) for item in head["audited_calls"])
    ):
        raise AssertionError("complete matching-precision projection and call-count evidence required")


def answer_observation(fp32_correct, bf16_correct):
    if fp32_correct and not bf16_correct:
        return "controlled_answer_improvement_on_this_one_question"
    if not fp32_correct and bf16_correct:
        return "controlled_answer_regression_on_this_one_question"
    if fp32_correct:
        return "both_correct_no_observed_B1_answer_accuracy_gain"
    return "both_incorrect_no_observed_B1_answer_recovery"


def analyze(args):
    original = args.original
    manifest = load_manifest(original / "inputs/manifest.json")
    target = json.loads((original / "inputs/targets.json").read_bytes())[27]
    rules = load_evalscope_rules(original / "evalscope-source")
    directories = {
        "official_fp32": (args.fp32_evidence / "generation/official", "official", "torch.float32"),
        "compact_fp32": (args.fp32_evidence / "generation/compact", "compact", "torch.float32"),
        "compact_bf16": (args.bf16_evidence / "generation/compact", "compact", "torch.bfloat16"),
    }
    records, scores, lifecycle, heads, hashes = {}, {}, {}, {}, {}
    for label, (directory, variant, dtype) in directories.items():
        path = directory / f"{variant}.question27.json"
        record = json.loads(path.read_bytes())
        if (
            record["index"] != 27
            or record["manifest_sha256"] != manifest["payload_sha256"]
            or record["prompt_tokens"] != len(manifest["payload"]["inputs"][27]["prompt_tokens"])
            or record["error"]
        ):
            raise AssertionError("retain the same complete Q28 prompt and successful independent generation")
        result_path = directory / f"{variant}.shard0.group0.result.json"
        result = json.loads(result_path.read_bytes())
        if result["indices"] != [27] or result["capacity_tokens"] != 4:
            raise AssertionError("this attribution freezes B1/Q4")
        lifecycle[label] = audit_group(manifest, result_path)
        records[label] = record
        heads[label] = []
        for rank in range(2):
            head = json.loads((directory / f"{variant}.head-audit.rank{rank}.json").read_bytes())
            audit_head(head, dtype, record["q4_steps"])
            heads[label].append(head)
        score = score_output(rules, record["scorer_text"], str(target["target"]))
        scores[label] = dict(
            **score,
            correct=bool(score["matched"]),
            finish_reason=record["finish_reason"],
            length_limited=record["length_limited"],
            generated_token_count=record["generated_token_count"],
            q4_steps=record["q4_steps"],
            fallback_calls=record["fallback_calls"],
            final_text=record["scorer_text"][-1600:],
        )
        hashes[label] = hashlib.sha256(path.read_bytes()).hexdigest()
    texts = json.loads((args.bf16_evidence / "token-text.json").read_bytes())
    pairs = {}
    for left, right in combinations(records, 2):
        prefix = common_prefix(records[left]["generated_tokens"], records[right]["generated_tokens"])
        text = texts.get(left + "__" + right, texts.get(right + "__" + left))
        if text is None or text["common_output_prefix"] != prefix:
            raise AssertionError("tokenizer evidence must agree with the actual complete token sequences")
        first = None
        if prefix < min(len(records[label]["generated_tokens"]) for label in (left, right)):
            first = dict(output_0based=prefix, output_1based=prefix + 1, variants={})
            for label in (left, right):
                directory, variant, _ = directories[label]
                token = records[label]["generated_tokens"][prefix]
                if token != text["variants"][label]["token"]:
                    raise AssertionError("decoded first token does not match its record")
                path = directory / f"{variant}.shard0.group0.steps.jsonl"
                with path.open(encoding="utf-8") as stream:
                    site = emission_site((json.loads(line) for line in stream), prefix, records[label]["prompt_tokens"])
                first["variants"][label] = dict(**text["variants"][label], site=site)
        pairs[left + "__" + right] = dict(common_output_prefix=prefix, first_difference=first)
    old = json.loads((original / "paired-scores.json").read_bytes())
    old_q28 = next(row for row in old["rows"] if row["index"] == 27)
    return dict(
        scope="one known question, three B1 paths; no aggregate quality or performance qualification",
        scores=scores,
        pairs=pairs,
        lifecycle_audit=lifecycle,
        head_audit=heads,
        output_sha256=hashes,
        old_B6_separate=old_q28,
        compact_B1_head_only_answer_observation=answer_observation(
            scores["compact_fp32"]["correct"], scores["compact_bf16"]["correct"]
        ),
        original_failures_remain_valid=True,
        old_B6_vs_B1_head_attribution_is_confounded=True,
        no_new_head_cost_or_decoder_performance_round=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fp32-evidence", type=Path, required=True)
    parser.add_argument("--bf16-evidence", type=Path, required=True)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("preserve previous results")
    result = analyze(args)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(scores=result["scores"], observation=result["compact_B1_head_only_answer_observation"])))


if __name__ == "__main__":
    main()
