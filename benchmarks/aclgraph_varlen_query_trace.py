# SPDX-License-Identifier: Apache-2.0
"""Freeze synthetic variable-query snapshots before any device measurement."""

import argparse
import hashlib
import itertools
import json
import random
from pathlib import Path

Q4_DIGEST = "0519bc80410bd15d0893e0283189425ec285d53003198c163183b79e6249e524"
TRACE_SEED = 20261020
PHYSICAL_BATCH = 128
PHYSICAL_TOKENS = 512
PROBE_STEPS = tuple(range(8))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def make_trace(original):
    source_trace = dict(original["trace"])
    source_trace["initial_prefix_by_request"] = {
        int(key): value for key, value in source_trace["initial_prefix_by_request"].items()
    }
    if original["trace_sha256"] != Q4_DIGEST or digest(source_trace) != Q4_DIGEST:
        raise ValueError("original Q4 trace digest changed")
    rng = random.Random(TRACE_SEED)
    steps = []
    for source in original["trace"]["steps"]:
        # 5/10/15/70 percent for Q=1/2/3/4. A fixed synthetic application
        # distribution, not observed draft confidence or an acceptance model.
        q = [rng.choices((1, 2, 3, 4), weights=(5, 10, 15, 70))[0] for _ in source["request_ids"]]
        kv = [length - 4 + length_q for length, length_q in zip(source["seq_lengths"], q, strict=True)]
        q += [0] * (PHYSICAL_BATCH - len(q))
        kv += [0] * (PHYSICAL_BATCH - len(kv))
        steps.append(
            dict(
                step=source["step"],
                batch=source["batch"],
                request_ids=source["request_ids"],
                q_lengths=q,
                q_endpoints=list(itertools.accumulate(q)),
                kv_lengths=kv,
                live_tokens=sum(q),
                input_seed=source["input_seed"],
            )
        )
    trace = dict(
        version=1,
        application="synthetic per-request variable-length speculative verification snapshots",
        provenance=(
            "request order and pre-query context lengths from original synthetic Q4 trace; Q sampled independently"
        ),
        limitation="not an observed service/draft trace; independent snapshots, no acceptance/throughput claim",
        seed=TRACE_SEED,
        q_weights_percent=[5, 10, 15, 70],
        physical_batch=PHYSICAL_BATCH,
        physical_tokens=PHYSICAL_TOKENS,
        geometry=dict(query_heads=12, kv_heads=2, head_dim=256, page_size=128, dtype="bfloat16"),
        contract=dict(
            query="compact active prefix of fixed TND storage; zero-query inactive requests",
            q_descriptor="int64 cumulative endpoints, WITHOUT leading zero",
            kv_descriptor="int64 local length per request, including current query",
            causal="right-down: k <= kv_length - q_length + q_index",
            output="all physical rows finite, inactive tail exactly zero",
            pages="fixed request-owned disjoint pages, source request IDs preserve cache ownership",
            capacity="512 from original default capture catalog; one fixed graph, no new capture size",
        ),
        probe_steps=list(PROBE_STEPS),
        profile_steps=[0, 7],
        predeclared_gate=dict(
            candidate="only a new repeated cost remaining after community shared device metadata",
            required_fraction=0.05,
            required_us_per_attention=10,
            rule="both required; attributable removable cost, not total compute or an already shared producer",
            no_budget="close screening without candidate implementation or kernel tuning enumeration",
        ),
        steps=steps,
    )
    return dict(trace=trace, trace_sha256=digest(trace), q4_trace_sha256=Q4_DIGEST)


def validate_trace(record):
    trace = record["trace"]
    if digest(trace) != record["trace_sha256"] or record["q4_trace_sha256"] != Q4_DIGEST:
        raise ValueError("trace identity mismatch")
    for step in trace["steps"]:
        q, kv = step["q_lengths"], step["kv_lengths"]
        if len(q) != PHYSICAL_BATCH or len(kv) != PHYSICAL_BATCH:
            raise ValueError("descriptor extent mismatch")
        if list(itertools.accumulate(q)) != step["q_endpoints"] or sum(q) != step["live_tokens"]:
            raise ValueError("query endpoint semantics mismatch")
        if not 0 <= sum(q) <= PHYSICAL_TOKENS or any(not 0 <= a <= 4 or b < a for a, b in zip(q, kv, strict=True)):
            raise ValueError("invalid query/KV domain")
        if any(q[step["batch"] :]) or any(kv[step["batch"] :]):
            raise ValueError("inactive descriptors must be zero")
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--q4", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("frozen trace already exists")
    result = validate_trace(make_trace(json.loads(args.q4.read_bytes())))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes((json.dumps(result, indent=2) + "\n").encode())
    print(json.dumps(dict(trace_sha256=result["trace_sha256"], steps=len(result["trace"]["steps"]))))


if __name__ == "__main__":
    main()
