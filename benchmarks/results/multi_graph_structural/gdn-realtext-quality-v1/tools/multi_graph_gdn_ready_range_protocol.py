# SPDX-License-Identifier: Apache-2.0
"""Predeclared Q4 load ranges with row order separate from request identity."""

import hashlib
import json

PROFILES = {
    "low": (32, (24, 22, 26, 23, 25, 24, 22, 26)),
    "medium": (64, (56, 60, 58, 54, 60, 56, 58, 54)),
    "high": (100, (98, 94, 100, 96, 98, 100, 94, 96)),
}
STEPS = 40
PREFIX_TOKENS = 16
TEXTS = (
    "Explain how a recurrent neural network keeps a state across successive inputs. ",
    "A research report should separate measured results, assumptions, and remaining limitations. ",
    "Write a short Python example that computes a moving average over a sequence of numbers. ",
    "Describe why asynchronous producers and consumers need explicit data readiness signals. ",
)


def frozen_profiles():
    profiles = []
    for name, (pool, loads) in PROFILES.items():
        generations, committed = [0] * 100, [0] * 100
        entries = []
        for step in range(STEPS):
            live = loads[step % len(loads)]
            requests = [(step * 7 + offset) % pool for offset in range(live)]
            if step % 2:
                requests.reverse()
            accepted = [1 + (owner + generations[owner]) % 4 for owner in requests]
            for owner, count in zip(requests, accepted, strict=True):
                if generations[owner]:
                    committed[owner] += count
            entries.append(
                dict(
                    step=step,
                    live=live,
                    request_ids=requests,
                    accepted=accepted,
                    seq_lengths=[PREFIX_TOKENS + committed[owner] + 4 for owner in requests],
                )
            )
            for owner in requests:
                generations[owner] += 1
        profiles.append(
            dict(
                name=name,
                pool=pool,
                trace=entries,
                terminal_counts=[1 + (owner + generation) % 4 for owner, generation in enumerate(generations)],
            )
        )
    return profiles


def protocol():
    body = dict(
        profiles=frozen_profiles(),
        query=4,
        capacity_tokens=400,
        owners=100,
        prefix_tokens=PREFIX_TOKENS,
        texts=TEXTS,
        tokens="local checkpoint tokenizer; owner prefix plus repeated text; accepted-prefix windows",
        activation="embedding and official layers 0..3; target layer 4 prefilled with same text",
        modes=("official", "compact_delayed", "compact_ready", "ring_delayed", "ring_ready"),
        threshold_nrmse=0.005,
        rounds=3,
    )
    digest = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return dict(**body, sha256=digest)
