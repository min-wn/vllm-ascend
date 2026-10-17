"""Predeclare dynamic replay inputs independently of optimization results.

This is a workload manifest, not a serving scheduler. Stable request IDs can
leave and rejoin a batch; implementations must retain their logical state.
Accepted-prefix counts exercise state correctness, not draft-model quality.
"""

import argparse
import hashlib
import json
import random
from pathlib import Path


def make_trace(*, seed=20261018, steps=128, min_batch=64, max_batch=128, min_prefix=512, max_prefix=1536):
    if (
        any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in (seed, steps, min_batch, max_batch, min_prefix, max_prefix)
        )
        or steps < 1
        or not 1 <= min_batch <= max_batch
        or not 0 <= min_prefix <= max_prefix
    ):
        raise ValueError("integer seed, positive steps/batches and valid prefix range required")
    rng = random.Random(seed)
    prefix = {request: rng.randint(min_prefix, max_prefix) for request in range(max_batch)}
    initial_prefix = dict(prefix)
    pending = {request: 1 for request in range(max_batch)}
    seen, records = set(), []
    for step in range(steps):
        batch = rng.randint(min_batch, max_batch)
        requests = rng.sample(range(max_batch), batch)
        accepted = [pending[request] for request in requests]
        after = [rng.choice((1, 2, 3, 4)) for _ in requests]
        # A resumed request consumes its pending prefix exactly once. Its
        # history does not advance while absent from the batch.
        for request in requests:
            if request in seen:
                prefix[request] += pending[request]
        records.append(
            dict(
                step=step,
                batch=batch,
                request_ids=requests,
                first_use=[request not in seen for request in requests],
                seq_lengths=[prefix[request] + 4 for request in requests],
                accepted_from_previous=accepted,
                accepted_after_step=after,
                input_seed=rng.getrandbits(31),
            )
        )
        for request, count in zip(requests, after, strict=True):
            pending[request] = count
            seen.add(request)
    return dict(
        version=1,
        seed=seed,
        query=4,
        steps=records,
        initial_prefix_by_request=initial_prefix,
        parameters=dict(
            steps=steps, min_batch=min_batch, max_batch=max_batch, min_prefix=min_prefix, max_prefix=max_prefix
        ),
        contract=[
            "No custom capture capacities: choose each implementation's legal default dispatcher buckets at runtime.",
            "Request ID is independent of batch position; absent requests retain state and pending accepted count.",
            "Sequence length includes four current query tokens; prefill/initialization occurs before measured decode.",
            "First use consumes initialized base state; accepted_from_previous is ignored only for that first use.",
            "Accepted counts are fixed test inputs, not measured acceptance rate or model quality.",
            "Include current-step preparation, host updates, all device work/joins and hidden/residual endpoints.",
        ],
    )


def trace_digest(trace):
    return hashlib.sha256(json.dumps(trace, sort_keys=True, separators=(",", ":")).encode("ascii")).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20261018)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refuse to overwrite an existing workload manifest")
    trace = make_trace(seed=args.seed, steps=args.steps)
    report = dict(
        trace=trace,
        trace_sha256=trace_digest(trace),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="ascii")
    print(json.dumps(dict(trace_sha256=report["trace_sha256"], output=str(args.output.resolve()))))


if __name__ == "__main__":
    main()
