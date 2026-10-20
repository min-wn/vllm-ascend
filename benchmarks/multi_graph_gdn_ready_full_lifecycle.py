# SPDX-License-Identifier: Apache-2.0
"""Final SSM acceptance and paired whole-model lifecycle accounting."""

import math

MODES = ("deferred_tile_delayed", "deferred_tile_ready")


def terminal_counts(owners):
    if owners != 128:
        raise ValueError("the unchanged full trajectory has 128 owners")
    return tuple(1 + owner % 4 for owner in range(owners))


class TerminalDrain:
    """Pay each owner's last SSM acceptance, including currently paused owners.

    Conv/FIA storage stays intact; ``accepted`` remains the next conv history
    selector. Clearing SSM validity prevents committing that same prefix twice.
    No request eviction, page reuse or generation is simulated here.
    """

    def __init__(self, torch, bundle, registry):
        self.torch, self.bundle, self.registry = torch, bundle, registry
        count = len(bundle.pages.pages)
        accepted = terminal_counts(count)
        self.sequences = torch.tensor([0, *([4] * count)], dtype=torch.int32, device=bundle.device)
        self.ids = torch.arange(1, count + 1, dtype=torch.int32, device=bundle.device).repeat_interleave(4)
        self.accepted = torch.tensor(accepted, dtype=torch.int32, device=bundle.device)
        state = bundle.gdn_entries[0].state
        self.empty_logs, self.empty_valid = torch.zeros_like(state.logs), torch.zeros_like(state.valid)
        self.contract = dict(
            accepted=list(accepted),
            owners=count,
            gdn_layers=len(bundle.gdn_entries),
            selected_state="every owner's accepted final candidate; canonical logs/valid cleared",
            conv_and_fia="retained without mutation; final accepted counts retained for conv history selection",
            implementation="same deferred commit with zero new logs, one captured drain shared by both controls",
        )

    def run(self):
        self.registry._require_joined()
        for item in self.bundle.gdn_entries:
            state = item.state
            backend = self.registry.backends[state.handle]
            backend.deferred_module.commit(
                state.base,
                self.sequences,
                self.ids,
                self.accepted,
                state.logs,
                state.valid,
                self.empty_logs,
                self.empty_valid,
            )

    def require_closed(self):
        for item in self.bundle.gdn_entries:
            if self.torch.count_nonzero(item.state.valid).item() or self.torch.count_nonzero(item.state.logs).item():
                raise AssertionError(f"layer {item.index}: final SSM acceptance debt remains")


def paired_lifecycles(ranks, trials):
    if len(ranks) != 2 or {rank["rank"] for rank in ranks} != {0, 1}:
        raise ValueError("TP2 ranks required")
    expected = {(trial, mode) for trial in range(trials) for mode in MODES}
    for rank in ranks:
        rows = rank["lifecycle_samples"]
        if len(rows) != len(expected) or {(r["trial"], r["mode"]) for r in rows} != expected:
            raise ValueError("incomplete paired lifecycle trials")
        for row in rows:
            if not row["terminal_exact"] or row["steps"] != 128:
                raise ValueError("unclosed or incomplete trajectory")
            if not all(
                math.isfinite(row[key]) and row[key] > 0
                for key in ("wall_us", "closed_wall_us", "terminal_wall_us", "transition_us")
            ) or not math.isclose(row["closed_wall_us"], row["wall_us"] + row["transition_us"]):
                raise ValueError("transition cost missing or invalid timing")
    paired = []
    for trial in range(trials):
        times = {
            mode: max(
                row["closed_wall_us"]
                for rank in ranks
                for row in rank["lifecycle_samples"]
                if row["trial"] == trial and row["mode"] == mode
            )
            for mode in MODES
        }
        paired.append(
            dict(trial=trial, closed_wall_us=times, ready_percent=100 * (1 - times[MODES[1]] / times[MODES[0]]))
        )
    return paired
