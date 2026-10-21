# SPDX-License-Identifier: Apache-2.0
"""Frozen short generation workload and strict same-algorithm pair checks."""

import json
import math
from pathlib import Path

from multi_graph_gdn_aime26 import digest
from multi_graph_gdn_q4_full_metadata import FrozenTrace
from multi_graph_gdn_ready_range_protocol import TEXTS

SCHEMA = "gdn-ready-offline-autoregressive-v1"
REQUESTS = 56
MAX_NEW_TOKENS = 256
MAX_MODEL_LEN = 2048
MODES = ("delayed", "ready")
PAIRS = 3
COMPLETE = "complete_offline_actual_autoregressive_generation"
COST_KEYS = ("prefill_us", "state_import_us", "generation_us", "terminal_us", "batch_us")


def settings():
    return dict(
        schema=SCHEMA,
        requests=REQUESTS,
        max_new_tokens=MAX_NEW_TOKENS,
        max_model_len=MAX_MODEL_LEN,
        q4=4,
        ngram_min=2,
        ngram_max=4,
        draft_tokens=3,
        greedy=True,
        enable_thinking=False,
        eos_token_ids=[248046, 248044],
        ignore_eos=False,
        seed=42,
        head="original BF16 operands and BF16 logits",
        algorithm="unchanged compact FP32 state and recurrence",
        graph_catalog="installed defaults; all possible live batches 1..56; no added sizes",
    )


def prompt_texts():
    return [
        TEXTS[owner % len(TEXTS)].strip() + " Give a concise answer with one concrete example."
        for owner in range(REQUESTS)
    ]


def freeze(tokenizer, model_files):
    rows = []
    for owner, text in enumerate(prompt_texts()):
        tokens = tokenizer.apply_chat_template(
            [dict(role="user", content=text)],
            tokenize=True,
            return_dict=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        rows.append(dict(owner=owner, text=text, prompt_tokens=tokens))
    payload = dict(settings=settings(), inputs=rows, model_files=model_files)
    return dict(payload=payload, payload_sha256=digest(payload))


def load_manifest(path):
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    payload = document["payload"]
    if document["payload_sha256"] != digest(payload) or payload["settings"] != settings():
        raise ValueError("generation protocol changed after freeze")
    rows = payload["inputs"]
    if len(rows) != REQUESTS or [r["owner"] for r in rows] != list(range(REQUESTS)):
        raise ValueError("all 56 ordered resident requests required")
    if [r["text"] for r in rows] != prompt_texts():
        raise ValueError("frozen natural-language prompts changed")
    if any(not 2 <= len(r["prompt_tokens"]) <= MAX_MODEL_LEN - MAX_NEW_TOKENS - 4 for r in rows):
        raise ValueError("prompt, output and speculative tail must fit the existing context limit")
    return document


def capacity_trace(document):
    """Reserve maximum legal histories; never prescribe runtime acceptance."""
    lengths = tuple(len(r["prompt_tokens"]) for r in document["payload"]["inputs"])
    steps = tuple(
        dict(
            step=live - 1,
            batch=live,
            request_ids=tuple(range(live)),
            seq_lengths=tuple(length + MAX_NEW_TOKENS + 4 for length in lengths[:live]),
            accepted_from_previous=(1,) * live,
        )
        for live in range(1, REQUESTS + 1)
    )
    return FrozenTrace(steps, lengths, 42, document["payload_sha256"])


def run_plan():
    return [
        dict(pair=pair, mode=mode, order=order)
        for pair in range(PAIRS)
        for order, mode in enumerate(MODES if pair % 2 == 0 else reversed(MODES))
    ]


def critical_cost(ranks):
    if len(ranks) != 2 or {r["rank"] for r in ranks} != {0, 1}:
        raise ValueError("exactly two TP endpoints required")
    for row in ranks:
        if any(not math.isfinite(row[k]) or row[k] <= 0 for k in COST_KEYS):
            raise ValueError("costs must be positive and finite")
        if abs(sum(row[k] for k in COST_KEYS[:-1]) - row["batch_us"]) > 0.1:
            raise ValueError("batch endpoint must include all four lifecycle phases")
    row = max(ranks, key=lambda r: r["batch_us"])
    return {k: row[k] for k in ("rank", *COST_KEYS)}


def check_pair(delayed, ready):
    for reports, mode in ((delayed, "delayed"), (ready, "ready")):
        if len(reports) != 2:
            raise ValueError("both TP reports required")
        for rank, report in enumerate(reports):
            if report["status"] != COMPLETE or report["mode"] != mode or report["rank"] != rank:
                raise ValueError("incomplete or mismatched generation")
            if not report["timed_matches_validation"] or report["fallback_calls"]:
                raise ValueError("timed run diverged from validation or used a fallback")
            if report["critical_cost"] != critical_cost(report["cost_ranks"]):
                raise ValueError("invalid lifecycle cost aggregation")
    for rank in (0, 1):
        for key in (
            "workload_sha256",
            "selection",
            "loader",
            "dense_weights",
            "head",
            "initial_state",
            "validation",
            "generation",
            "terminal_contract",
            "kernel_identities",
        ):
            if delayed[rank][key] != ready[rank][key]:
                raise ValueError(f"rank{rank}/{key}: schedules did not execute identical work and state")
    if delayed[0]["generation"] != delayed[1]["generation"] or ready[0]["generation"] != ready[1]["generation"]:
        raise ValueError("TP ranks disagree on generated work")
    return True
