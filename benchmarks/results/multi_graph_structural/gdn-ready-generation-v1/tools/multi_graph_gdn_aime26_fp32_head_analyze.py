# SPDX-License-Identifier: Apache-2.0
"""Offline Q28-only scoring, first-token alignment and lifecycle audit.

References are available only to this postprocessor, never to generation.
This does not produce or replace the original all-30-question AIME report.
"""

import argparse
import hashlib
import json
from pathlib import Path

from multi_graph_gdn_aime26 import load_evalscope_rules, load_manifest, score_output
from multi_graph_gdn_aime26_audit import audit_group
from multi_graph_gdn_aime26_divergence import common_prefix


def emission_site(journal, output_offset, prompt_length):
    if output_offset == 0:
        return dict(stage="prefill", predictor_position=prompt_length - 1)
    before = 1
    for step in journal:
        if step["batch"] != 1 or step["question_indices"] != [27]:
            raise ValueError("this diagnostic freezes B1 and Q28 only")
        decision = step["decisions"][0]
        if before <= output_offset < before + len(decision["tokens"]):
            query_offset = output_offset - before
            position = step["seq_lengths"][0] - 4 + query_offset
            return dict(
                stage="Q4",
                step=step["step"],
                batch=1,
                owner=step["request_ids"][0],
                query_offset=query_offset,
                sequence_length=step["seq_lengths"][0],
                predictor_position=position,
                output_position=position + 1,
                accepted_from_previous=step["accepted_from_previous"][0],
                accepted_this_step=decision["accepted"],
                proposals=step["proposals"][0],
                predictions=step["predictions"],
            )
        before += len(decision["tokens"])
    raise ValueError("first different token is not covered by the actual generation journal")


def analyze(args, tokenizer):
    manifest = load_manifest(args.manifest)
    targets = json.loads(args.targets.read_bytes())
    target = targets[27]
    if target["index"] != 27 or target["id"] != manifest["payload"]["inputs"][27]["id"]:
        raise ValueError("reference/Q28 identity mismatch")
    rules = load_evalscope_rules(args.evalscope_source)
    records, rows, audit, hashes = {}, {}, {}, {}
    for variant in ("official", "compact"):
        directory = args.evidence / "generation" / variant
        path = directory / f"{variant}.question27.json"
        record = json.loads(path.read_bytes())
        if record["manifest_sha256"] != manifest["payload_sha256"] or record["index"] != 27:
            raise ValueError("original prompt and generation settings required")
        records[variant] = record
        hashes[variant] = hashlib.sha256(path.read_bytes()).hexdigest()
        score = score_output(rules, record["scorer_text"], str(target["target"]))
        rows[variant] = dict(
            **score,
            correct=bool(score["matched"] and record["finish_reason"] != "runtime_error"),
            finish_reason=record["finish_reason"],
            length_limited=record["length_limited"],
            generated_token_count=record["generated_token_count"],
            q4_steps=record["q4_steps"],
            fallback_calls=record["fallback_calls"],
            error=record["error"],
            final_text=record["scorer_text"][-1600:],
        )
        result_path = directory / f"{variant}.shard0.group0.result.json"
        audit[variant] = audit_group(manifest, result_path)
        for rank in range(2):
            head = json.loads((directory / f"{variant}.head-audit.rank{rank}.json").read_bytes())
            if head["calls"] != record["q4_steps"] + 1 or head["fallback_calls"]:
                raise AssertionError("complete continuous FP32-head call audit required")
            if head["output_dtype"] != "torch.float32" or head["weight_dtype"] != "torch.bfloat16":
                raise AssertionError("do not substitute another precision or set of head weights")
    tokens = {variant: record["generated_tokens"] for variant, record in records.items()}
    prefix = common_prefix(tokens["official"], tokens["compact"])
    first = None
    if prefix < min(map(len, tokens.values())):
        first = dict(output_0based=prefix, output_1based=prefix + 1, variants={})
        for variant, ids in tokens.items():
            path = args.evidence / "generation" / variant / f"{variant}.shard0.group0.steps.jsonl"
            with path.open(encoding="utf-8") as stream:
                site = emission_site((json.loads(line) for line in stream), prefix, records[variant]["prompt_tokens"])
            first["variants"][variant] = dict(
                token=ids[prefix],
                text=tokenizer.decode([ids[prefix]]) if tokenizer is not None else None,
                context=tokenizer.decode(ids[max(0, prefix - 20) : prefix + 21]) if tokenizer is not None else None,
                context_token_ids=ids[max(0, prefix - 20) : prefix + 21],
                site=site,
            )
    return dict(
        scope="one known failed question, Q28 only; diagnostic, no aggregate quality qualification",
        batch=1,
        query=4,
        head_dtype="float32",
        original_generation_manifest=manifest["payload_sha256"],
        target=target["target"],
        scores=rows,
        common_output_prefix=prefix,
        first_difference=first,
        one_output_is_prefix_of_other=prefix == min(map(len, tokens.values()))
        and tokens["official"] != tokens["compact"],
        lifecycle_audit=audit,
        output_sha256=hashes,
        original_B6_comparison_is_a_confound=True,
        original_failures_remain_valid=True,
        new_full_aime_or_performance_evaluation=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--evalscope-source", type=Path, required=True)
    parser.add_argument("--model", default="/models/Qwen3.5-27B-original")
    parser.add_argument("--tokens-only", action="store_true", help="offline scoring without installing a tokenizer")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("preserve previous results")
    tokenizer = None
    if not args.tokens_only:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model)
    result = analyze(args, tokenizer)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(scores=result["scores"], common_output_prefix=result["common_output_prefix"])))


if __name__ == "__main__":
    main()
