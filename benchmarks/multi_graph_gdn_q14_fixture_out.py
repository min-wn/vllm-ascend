# SPDX-License-Identifier: Apache-2.0
"""One fixture-only change; frozen recurrence and original Trial stay intact."""

import statistics

from multi_graph_gdn_q14_probe import Trial
from multi_graph_gdn_q14_protocol import metadata, stability_screen

CONFIGS = ("legacy_forced_q4", "legacy_q14_ready", "out_forced_q4", "out_q14_ready")
REPEATS = 8
ROUNDS = 3


class OutTrial(Trial):
    def prepare(self, row):
        torch = self.torch
        owners, actual = row["request_ids"], row["query_lengths"]
        previous = row["accepted_from_previous"]
        for owner, accepted in zip(owners, previous):
            if owner in self.last_width and accepted > self.last_width[owner]:
                raise ValueError("acceptance exceeds the actual previous query width")
        lengths = [4] * len(owners) if self.mode == "forced_q4" else actual
        meta = metadata(owners, lengths, previous, self.capacity)
        source_indices = [4 * o + t for o, q in zip(owners, lengths) for t in range(q)]
        source_indices += [0] * (4 * self.capacity - len(source_indices))
        payload = [*meta["sequence"], *meta["ids"], *meta["accepted"], *source_indices]
        self.meta.copy_(torch.tensor(payload, dtype=torch.int32))
        count = sum(lengths)
        select = self.source_indices[:count].long()
        for name, source in self.fixture.operands.items():
            torch.index_select(source, 0, select, out=self.buffers[name][:count])
        self.count = count
        self.key = (sum(q == 1 for q in actual), sum(q == 4 for q in actual)) if self.mode == "native" else "compact"
        if self.mode == "native":
            self.prepare_native(owners, actual, previous)
        self.last_width.update(dict(zip(owners, actual)))
        self.last_accept.update({o: d["accepted"] for o, d in zip(owners, row["decisions"])})
        return meta


def round_order(round_id):
    return CONFIGS if round_id % 2 == 0 else tuple(reversed(CONFIGS))


def summarize(samples):
    expected = {(c, r, s) for c in CONFIGS for r in range(ROUNDS) for s in range(REPEATS)}
    if len(samples) != len(expected):
        raise ValueError("exactly 96 complete samples required")
    observed = {(s["config"], s["round"], s["sample"]) for s in samples}
    if observed != expected:
        raise ValueError("sample identity is duplicated or missing")
    medians = {
        c: [
            statistics.median(s["lifecycle_ms"] for s in samples if s["config"] == c and s["round"] == r)
            for r in range(ROUNDS)
        ]
        for c in CONFIGS
    }
    return dict(
        sample_count=len(samples),
        round_medians_lifecycle_ms=medians,
        primary_out_vs_out=stability_screen(medians["out_q14_ready"], medians["out_forced_q4"]),
        legacy_vs_legacy=stability_screen(medians["legacy_q14_ready"], medians["legacy_forced_q4"]),
        preparation_change={
            mode: stability_screen(medians["out_" + mode], medians["legacy_" + mode])
            for mode in ("forced_q4", "q14_ready")
        },
    )
