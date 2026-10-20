# SPDX-License-Identifier: Apache-2.0
"""Freeze real-text requests without loading weights or observing predictions."""

import argparse
import hashlib
import json
from pathlib import Path

from multi_graph_gdn_q4_full_metadata import load_fixed_trace
from multi_graph_gdn_quality import BOOTSTRAP_DRAWS, BOOTSTRAP_SEED, LIMITS, SCHEMA, digest, score_layout


def articles_from_rows(rows):
    articles, current = [], []
    for row in rows:
        stripped = row.strip()
        title = stripped.startswith("= ") and not stripped.startswith("= =") and stripped.endswith(" =")
        if title and current:
            articles.append("".join(current))
            current = []
        current.append(row)
    if current:
        articles.append("".join(current))
    return articles


def allocate_windows(articles, lengths, tokenize):
    windows, article, offset = [], 0, 0
    tokenized = [tokenize(text) for text in articles]
    for owner, length in enumerate(lengths):
        while article < len(tokenized) and len(tokenized[article]) - offset < length:
            article, offset = article + 1, 0
        if article == len(tokenized):
            raise ValueError("not enough nonoverlapping within-article text for the fixed Q4 trace")
        tokens = tokenized[article][offset:offset + length]
        windows.append(dict(owner=owner, article=article, token_offset=offset, tokens=tokens, token_sha256=digest(tokens)))
        offset += length
    return windows


def prepare(args):
    from transformers import AutoTokenizer

    trace = load_fixed_trace(args.trace)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    rows = json.loads((args.inputs / "wikitext-rows.json").read_text())
    articles = articles_from_rows(rows)
    lengths = list(trace.initial_prefix_by_request)
    for step in trace.steps:
        for owner, length in zip(step["request_ids"], step["seq_lengths"], strict=True):
            lengths[owner] = max(lengths[owner], length + 1)
    windows = allocate_windows(articles, lengths, lambda text: tokenizer.encode(text, add_special_tokens=False))
    groups = len({window["article"] for window in windows})
    if groups < LIMITS["minimum_source_articles"]:
        raise ValueError("insufficient source articles for the predeclared cluster uncertainty check")
    lambada, skipped = [], []
    raw_task = [json.loads(line)["text"] for line in (args.inputs / "lambada.jsonl").read_text().splitlines()]
    for index, text in enumerate(raw_task):
        prompt, separator, word = text.rpartition(" ")
        all_ids = tokenizer.encode(text, add_special_tokens=False)
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        word_ids = tokenizer.encode(separator + word, add_special_tokens=False)
        if not separator or not 1 <= len(word_ids) <= 4 or all_ids != prompt_ids + word_ids or not 2 <= len(prompt_ids) - 1 <= 2044:
            skipped.append(index)
            continue
        ids = [*all_ids, *([tokenizer.eos_token_id] * 5)]
        lambada.append(dict(owner=len(lambada), row=index, tokens=ids, prefix=len(prompt_ids)-1,
                            target_tokens=word_ids, article=index, text_sha256=hashlib.sha256(text.encode()).hexdigest()))
        if len(lambada) == 128:
            break
    if len(lambada) != 128:
        raise ValueError("insufficient eligible task examples")
    tokenizer_files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(args.model.glob('*token*')) if p.is_file()}
    payload = dict(
        schema=SCHEMA, trace_sha256=trace.digest, screen=windows, task=lambada, task_skipped_rows=skipped,
        screen_expected_tokens=sum(map(len, score_layout(trace))), limits=LIMITS,
        bootstrap=dict(seed=BOOTSTRAP_SEED, draws=BOOTSTRAP_DRAWS, unit="source article"),
        sources={name: json.loads((args.inputs / (name + '-source.json')).read_text()) for name in ('wikitext','lambada')},
        article_sha256=[hashlib.sha256(text.encode()).hexdigest() for text in articles],
        tokenizer_files=tokenizer_files,
        preprocessing_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        protocol_sha256=hashlib.sha256(args.protocol.read_bytes()).hexdigest(),
    )
    document = dict(payload=payload, payload_sha256=digest(payload))
    if args.output.exists():
        raise FileExistsError("refusing to replace the frozen quality dataset")
    args.output.write_text(json.dumps(document, indent=2) + '\n')
    print(json.dumps(dict(manifest_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(),
                          payload_sha256=document['payload_sha256'], source_articles=groups,
                          screen_tokens=payload['screen_expected_tokens'], task_examples=len(lambada))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=Path('/models/Qwen3.5-27B-original'))
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    prepare(parser.parse_args())


if __name__ == '__main__':
    main()
