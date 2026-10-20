# SPDX-License-Identifier: Apache-2.0
"""Offline paired scoring; this entry point alone receives AIME26 references."""

import argparse
import json
from pathlib import Path

from multi_graph_gdn_aime26 import load_evalscope_rules, load_manifest, paired_summary, score_output


def score_all(manifest, targets, output_dir, rules):
    if len(targets) != 30 or [r["index"] for r in targets] != list(range(30)):
        raise ValueError("exactly 30 frozen targets required")
    rows = []
    for prompt, target in zip(manifest["payload"]["inputs"], targets, strict=True):
        if prompt["id"] != target["id"]:
            raise ValueError("target/question identity mismatch")
        index = prompt["index"]
        row = dict(index=index, id=prompt["id"], target=target["target"])
        for variant in ("official", "compact"):
            path = Path(output_dir) / f"{variant}.question{index:02}.json"
            if not path.exists():
                # Failure placeholders preserve the denominator. They cannot
                # be reported as a completed 30-pair generation experiment.
                row[variant] = dict(
                    correct=False,
                    finish_reason="runtime_error",
                    prediction_valid=False,
                    error="generation output missing",
                    generated_token_count=0,
                    fallback_calls=0,
                    q4_steps=0,
                    extracted="",
                    reason="missing_output",
                )
                continue
            generation = json.loads(path.read_text(encoding="utf-8"))
            if generation["manifest_sha256"] != manifest["payload_sha256"] or generation["index"] != index:
                raise ValueError("output was generated under a different frozen protocol")
            score = score_output(rules, generation["scorer_text"], str(target["target"]))
            row[variant] = dict(
                **score,
                correct=score["matched"] and generation["finish_reason"] != "runtime_error",
                finish_reason=generation["finish_reason"],
                error=generation.get("error"),
                generated_token_count=generation["generated_token_count"],
                q4_steps=generation["q4_steps"],
                fallback_calls=generation["fallback_calls"],
                output_file=path.name,
                output_stop_reason=generation.get("output_stop_reason"),
            )
        row["answer_changed"] = row["official"]["extracted"] != row["compact"]["extracted"]
        row["correctness_changed"] = row["official"]["correct"] != row["compact"]["correct"]
        rows.append(row)
    return dict(
        manifest_sha256=manifest["payload_sha256"],
        rows=rows,
        summary=paired_summary(rows),
        dataset="aime26",
        evaluation="one paired actual generation per question; rule-only",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--evalscope-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite a prior score report")
    result = score_all(
        load_manifest(args.manifest),
        json.loads(args.targets.read_text(encoding="utf-8")),
        args.generations,
        load_evalscope_rules(args.evalscope_source),
    )
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
