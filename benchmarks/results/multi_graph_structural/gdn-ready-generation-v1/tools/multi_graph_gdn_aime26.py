# SPDX-License-Identifier: Apache-2.0
"""Frozen AIME26 greedy Q4 generation bookkeeping and rule-only scoring.

This module has no model imports. References are consumed only by the separate
offline scorer; the generation manifest deliberately contains no answer field.
"""

import ast
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

QUERY = 4
MAX_NEW_TOKENS = 81920
MAX_MODEL_LEN = 82944
GROUP_SIZE = 8
SCHEMA = "gdn-aime26-generation-v1"


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


def load_manifest(path):
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    payload = document["payload"]
    if document["payload_sha256"] != digest(payload) or payload["schema"] != SCHEMA:
        raise ValueError("frozen AIME26 manifest changed")
    rows = payload["inputs"]
    if len(rows) != 30 or [r["index"] for r in rows] != list(range(30)):
        raise ValueError("all 30 ordered AIME26 questions required")
    if any(set(row) != {"index", "id", "question", "prompt", "prompt_tokens"} for row in rows):
        raise ValueError("generation inputs must contain no targets or extra fields")
    if any(not 2 <= len(r["prompt_tokens"]) <= MAX_MODEL_LEN - MAX_NEW_TOKENS - QUERY for r in rows):
        raise ValueError("prompt and full Q4 speculation must fit the frozen context capacity")
    if payload["generation"] != generation_config():
        raise ValueError("generation configuration differs from the predeclared protocol")
    return document


def generation_config():
    return dict(
        enable_thinking=True,
        do_sample=False,
        temperature=0,
        top_p=1,
        top_k=-1,
        min_p=0,
        presence_penalty=0,
        frequency_penalty=0,
        repetition_penalty=1,
        seed=42,
        max_new_tokens=MAX_NEW_TOKENS,
        max_model_len=MAX_MODEL_LEN,
        eos_token_ids=[248046, 248044],
        stop_strings=[],
        ignore_eos=False,
        ngram_min=2,
        ngram_max=4,
        speculative_tokens=3,
        draft_source="installed vLLM longest matched ngram over this request's own prompt and output",
        missing_draft="pad verifier to Q4; never accept unproposed filler",
        group_size=GROUP_SIZE,
        execution="eager quality harness, unchanged Q4 kernels; no performance measurements",
        scorer_text="tokenizer.decode(generated_tokens, skip_special_tokens=True); no reasoning stripping",
    )


def terminal_metadata(accepted):
    if not 1 <= len(accepted) <= 128 or any(type(n) is not int or not 1 <= n <= QUERY for n in accepted):
        raise ValueError("one actual accepted prefix in 1..4 per owner required")
    # Native compact ABI is [initial offset, row length, row length, ...],
    # whereas vLLM common attention metadata begins with cumulative offsets.
    return dict(
        sequence=[0, *([QUERY] * len(accepted))],
        ids=[owner + 1 for owner in range(len(accepted)) for _ in range(QUERY)],
        accepted=list(accepted),
    )


def verify_greedy(drafts, predictions, *, remaining, eos):
    """Emit matched drafts plus the first target correction/bonus token.

    Each emitted token has exactly one verified predictor. If EOS/length ends
    inside a matching prefix, commit only inputs preceding the final emitted
    token, as on an ordinary decode endpoint. It remains the unprocessed token.
    """
    if len(drafts) > QUERY - 1 or len(predictions) != QUERY or remaining < 1:
        raise ValueError("Q4 verifier needs up to three drafts, four predictions and positive remaining budget")
    emitted, matched = [], 0
    reason = None
    for index in range(len(drafts) + 1):
        token = int(predictions[index])
        emitted.append(token)
        if token in eos:
            reason = "eos"
            break
        if len(emitted) == remaining:
            reason = "length"
            break
        if index == len(drafts) or token != drafts[index]:
            break
        matched += 1
    # The count is input last_output + accepted new drafts before the last
    # output. This is also correct for early EOS and length truncation.
    return dict(tokens=emitted, accepted=len(emitted), matched_drafts=matched, finish_reason=reason)


@dataclass
class GenerationRequest:
    index: int
    prompt: tuple
    max_new_tokens: int = MAX_NEW_TOKENS
    eos: tuple = (248046, 248044)
    tokens: list = field(default_factory=list)
    processed: int = 0
    accepted: int = 1
    finish_reason: str | None = None
    q4_steps: int = 0

    def begin(self, first_token):
        if self.tokens or self.processed or self.finish_reason:
            raise ValueError("request prefill may run exactly once")
        self.processed = len(self.prompt)
        self.tokens.append(int(first_token))
        if first_token in self.eos:
            self.finish_reason = "eos"
        elif self.max_new_tokens == 1:
            self.finish_reason = "length"

    def input_tokens(self, drafts, filler=248044):
        if self.finish_reason or not self.tokens or len(drafts) > 3:
            raise ValueError("only unfinished initialized requests accept up to three proposals")
        self.require_consistent()
        return [self.tokens[-1], *drafts, *([filler] * (3 - len(drafts)))]

    def advance(self, drafts, predictions):
        if self.finish_reason:
            raise ValueError("finished request cannot advance")
        result = verify_greedy(drafts, predictions, remaining=self.max_new_tokens - len(self.tokens), eos=self.eos)
        self.tokens.extend(result["tokens"])
        self.accepted = result["accepted"]
        self.processed += result["accepted"]
        self.finish_reason = result["finish_reason"]
        self.q4_steps += 1
        self.require_consistent()
        return result

    def require_consistent(self):
        if self.processed != len(self.prompt) + len(self.tokens) - 1 or not 1 <= self.accepted <= QUERY:
            raise AssertionError("continuous state must end immediately before the last emitted token")


def load_evalscope_rules(source):
    """Load unchanged 1.13 math functions without the service/agent imports.

    Executed function/class bodies are exactly those of the pinned wheel.
    Dependency-only import edges are resolved here: metric's exception class
    is loaded from its AST, then contracts, then parser. No parser code changes.
    """
    source = Path(source) / "evalscope"
    namespace = {"__name__": "gdn_aime26_frozen_evalscope"}
    metric = ast.parse((source / "api/metric/metric.py").read_text(encoding="utf-8"))
    exception = next(
        node for node in metric.body if isinstance(node, ast.ClassDef) and node.name == "MetricUnavailableError"
    )
    exec(compile(ast.Module(body=[exception], type_ignores=[]), "evalscope/api/metric/metric.py", "exec"), namespace)
    for name in ("contracts", "parser"):
        path = source / f"metrics/math/{name}.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        tree.body = [
            node
            for node in tree.body
            if not isinstance(node, ast.ImportFrom) or node.module not in ("evalscope.api.metric", "contracts")
        ]
        exec(compile(tree, str(path), "exec"), namespace)
    return namespace


def score_output(rules, text, target):
    extracted = rules["extract_answer"](text)
    result = rules["compare_answers"](extracted, target, integer_only=True)
    return dict(filtered_prediction=extracted, **result.model_dump())


def paired_summary(rows):
    if len(rows) != 30 or {r["index"] for r in rows} != set(range(30)):
        raise ValueError("all 30 questions, including errors and truncations, must be retained")
    categories = dict(both_correct=0, both_wrong=0, official_only=0, compact_only=0)
    for row in rows:
        a, b = row["official"]["correct"], row["compact"]["correct"]
        key = (
            "both_correct" if a and b else "both_wrong" if not a and not b else "official_only" if a else "compact_only"
        )
        categories[key] += 1
    summary = dict(questions=30, **categories)
    for variant in ("official", "compact"):
        controls = [r[variant] for r in rows]
        correct = sum(r["correct"] for r in controls)
        summary[variant] = dict(
            correct=correct,
            accuracy=correct / 30,
            length_limited=sum(r["finish_reason"] == "length" for r in controls),
            runtime_failures=sum(r["finish_reason"] == "runtime_error" for r in controls),
            parse_failures=sum(not r.get("prediction_valid", False) for r in controls),
            fallback_calls=sum(r.get("fallback_calls", 0) for r in controls),
        )
    return summary
