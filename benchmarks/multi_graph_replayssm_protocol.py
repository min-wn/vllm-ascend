# SPDX-License-Identifier: Apache-2.0
"""Frozen Q4 ReplaySSM ring protocol, independent of Torch and NPU runtime."""

from dataclasses import dataclass

WINDOW = 4
LOGICAL_LENGTH = 20
RING_LENGTH = 32
STEPS = 40
COMMUNITY_COMMIT = "5b96cdbceefcdec6d60d588420ee127d76bd8f97"


@dataclass(frozen=True)
class Cursor:
    write_pos: int = 0
    cache_base: int = 0
    is_flush: bool = False
    pending: bool = False

    def advance(self, accepted):
        if isinstance(accepted, bool) or not isinstance(accepted, int) or not 1 <= accepted <= WINDOW:
            raise ValueError("accepted includes bonus and must be in 1..4")
        if not self.pending:
            return Cursor(pending=True)
        base = (self.cache_base + self.write_pos) % RING_LENGTH if self.is_flush else self.cache_base
        position = accepted if self.is_flush else self.write_pos + accepted
        if position + WINDOW > LOGICAL_LENGTH:
            raise ValueError("speculative window exceeds the logical cache extent")
        return Cursor(position, base, position + 2 * WINDOW > LOGICAL_LENGTH, True)


def frozen_trace(capacity=100, steps=STEPS):
    if capacity != 100 or steps != STEPS:
        raise ValueError("this qualification freezes 100 owners and 40 steps")
    generations = [0] * capacity
    result = []
    for step in range(steps):
        live = (98, 97, 99, 98, 98, 97, 99, 98)[step] if step < 8 else (98, 100, 99, 100, 98, 100, 99, 100)[step % 8]
        accepted = [1 + (owner + generation) % WINDOW for owner, generation in enumerate(generations)]
        result.append(dict(step=step, live=live, accepted=accepted, seed=20719 + step))
        for owner in range(live):
            generations[owner] += 1
    return result, [1 + (owner + generation) % WINDOW for owner, generation in enumerate(generations)]


def coverage(trace):
    cursors = [Cursor() for _ in range(100)]
    flushes, wraps, pauses, resumes = ([0] * 100 for _ in range(4))
    previous_live = 100
    for entry in trace:
        for owner in range(100):
            if owner >= entry["live"]:
                pauses[owner] += 1
                continue
            resumes[owner] += int(owner >= previous_live)
            old = cursors[owner]
            current = old.advance(entry["accepted"][owner])
            wraps[owner] += int(current.cache_base < old.cache_base)
            flushes[owner] += int(current.is_flush)
            cursors[owner] = current
        previous_live = entry["live"]
    return dict(flushes=flushes, wraps=wraps, pauses=pauses, resumes=resumes)
