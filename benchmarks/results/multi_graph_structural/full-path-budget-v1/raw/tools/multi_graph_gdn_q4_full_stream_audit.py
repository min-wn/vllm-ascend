# SPDX-License-Identifier: Apache-2.0
"""Exact continued-state audits with one mutable CPU checkpoint.

After an eager step, each chunk is copied to CPU, its device storage restored
from the preceding checkpoint, and that checkpoint chunk replaced by the eager
result. Replay then starts from exactly the same state and is compared against
the new checkpoint. No complete expected/actual snapshots or hashes are needed.
The caller owns prepare/finish lifecycle calls and output comparisons.
"""

from dataclasses import replace
from math import prod

from multi_graph_gdn_q4_full_checks import HOST_FIELDS, STATE_FIELDS, assert_complete_finite

SCAN_ELEMENTS = 1 << 20


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _host(state):
    return dict(
        state_identity=state._snapshot_identity,
        generations=tuple(state.generations),
        initialized=tuple(state.initialized),
        last_step=state.last_step,
        pending=state.pending,
    )


def _restore_host(state, fields):
    state.generations[:] = fields["generations"]
    state.initialized[:] = fields["initialized"]
    state.last_step, state.pending = fields["last_step"], fields["pending"]


def _clone_snapshot(torch, saved):
    assert_complete_finite(torch, saved, "stream_checkpoint/source")

    def clone(tensor):
        _require(tensor.device.type == "cpu", "stream checkpoint requires CPU source tensors")
        return tensor.clone()

    compact = saved["compact"]
    return dict(
        compact=replace(
            compact,
            states=tuple(replace(state, tensors=tuple(map(clone, state.tensors))) for state in compact.states),
            staging=tuple(
                tuple((identity, clone(logs), clone(valid)) for identity, logs, valid in staging)
                for staging in compact.staging
            ),
        ),
        conv=tuple(map(clone, saved["conv"])),
        fia=tuple(tuple(map(clone, pair)) for pair in saved["fia"]),
    )


def _paired_chunks(live, saved, limit, start=0):
    if live.is_contiguous() and saved.is_contiguous():
        left, right = live.view(-1), saved.view(-1)
        for offset in range(0, left.numel(), limit):
            yield left[offset : offset + limit], right[offset : offset + limit], start + offset
    elif live.ndim:
        for left, right in zip(live.unbind(0), saved.unbind(0), strict=True):
            yield from _paired_chunks(left, right, limit, start)
            start += left.numel()
    else:
        yield live.reshape(-1), saved.reshape(-1), start


def _protected_intervals(rows, row_elements):
    """Join neighboring protected rows before scanning bounded flat chunks."""
    intervals = []
    for row in sorted(rows):
        start, stop = row * row_elements, (row + 1) * row_elements
        if intervals and intervals[-1][1] == start:
            intervals[-1] = (intervals[-1][0], stop)
        else:
            intervals.append((start, stop))
    return intervals


def _identical(torch, left, right):
    # Signed zero can affect later arithmetic despite ordinary torch.equal.
    return torch.equal(left.contiguous().view(torch.uint8), right.contiguous().view(torch.uint8))


class StreamingCheckpoint:
    """Retain one independent CPU trajectory and at most one chunk of scratch.

    ``initial`` remains immutable for other modes/trials. This object clones its
    tensors once, while retaining the exact opaque registry/state identities.
    Every method is outside replay timing and requires all device work joined.
    A failed audit is terminal for this checkpoint; discard the experiment.
    Candidate reconstruction/finite checks and endpoint checks remain with the
    caller; finite canonical records alone cannot rule out candidate overflow.
    """

    def __init__(self, torch, initial, *, scan_elements=SCAN_ELEMENTS):
        if isinstance(scan_elements, bool) or not isinstance(scan_elements, int) or scan_elements < 1:
            raise ValueError("scan_elements must be a positive integer")
        self.torch, self.scan_elements = torch, scan_elements
        self.saved = _clone_snapshot(torch, initial)
        self._awaiting_replay = False
        self._failed = False
        self.completed_steps = 0

    def _bindings(self, bundle, registry, label):
        registry._require_joined()
        compact = self.saved["compact"]
        _require(tuple(registry.backends) == compact.registry_identity, f"{label}: registry identity changed")
        _require(
            len(registry.entries) == len(bundle.gdn_entries) == len(compact.states) == len(compact.staging),
            f"{label}: GDN layer count changed",
        )
        _require(len(bundle.fia_entries) == len(self.saved["fia"]), f"{label}: FIA layer count changed")
        _require(len(bundle.gdn_entries) == len(self.saved["conv"]), f"{label}: conv layer count changed")
        items, states = [], []
        for index, (entry, registered, saved, staging, conv) in enumerate(
            zip(
                bundle.gdn_entries,
                registry.entries,
                compact.states,
                compact.staging,
                self.saved["conv"],
                strict=True,
            )
        ):
            handle, backend, _, _ = registered
            state = backend.state
            prefix = f"{label}/gdn{index}"
            _require(entry.state is state and handle is state.handle, f"{prefix}: layer owner changed")
            _require(
                entry.gdn.kv_cache[0] is entry.conv and entry.gdn.kv_cache[1] is handle,
                f"{prefix}: live cache binding changed",
            )
            _require(state._snapshot_identity == saved.state_identity, f"{prefix}: state identity changed")
            _require(len(state.mutable_tensors) == len(saved.tensors) == len(STATE_FIELDS), f"{prefix}: state fields")
            states.append(state)
            for name, live, wanted in zip(STATE_FIELDS, state.mutable_tensors, saved.tensors, strict=True):
                items.append((f"{prefix}/{name}", live, wanted, "owner", saved.state_identity[1]))
            buffers = backend._staging_buffers()
            wanted_staging = {identity: (logs, valid) for identity, logs, valid in staging}
            _require(len(wanted_staging) == len(staging), f"{prefix}: duplicate staging identity")
            _require(buffers.keys() == wanted_staging.keys(), f"{prefix}: staging identities changed")
            for identity, lane in buffers.items():
                for name, wanted in zip(("new_logs", "new_valid"), wanted_staging[identity], strict=True):
                    items.append((f"{prefix}/staging{identity}/{name}", lane[name], wanted, "owner", ()))
            items.append((f"{label}/conv{index}", entry.conv, conv, "conv", saved.state_identity[1]))
        for index, (entry, pair) in enumerate(zip(bundle.fia_entries, self.saved["fia"], strict=True)):
            _require(
                entry.backend.kv_cache[0] is entry.key
                and entry.backend.kv_cache[1] is entry.value
                and entry.impl.key_cache is entry.key
                and entry.impl.value_cache is entry.value,
                f"{label}/fia{index}: live cache binding changed",
            )
            for name, live, wanted in zip(("key", "value"), (entry.key, entry.value), pair, strict=True):
                items.append((f"{label}/fia{index}/{name}", live, wanted, "fia", ()))
        for prefix, live, wanted, _, _ in items:
            _require(
                live.shape == wanted.shape and live.dtype == wanted.dtype and wanted.device.type == "cpu",
                f"{prefix}: tensor contract changed",
            )
            _require(live.ndim >= 1 and live.shape[0] > 0, f"{prefix}: row-addressable tensor required")
        return items, states

    def _schedule(self, requests, pages, label):
        rows = tuple(tuple(row) for row in getattr(pages, "pages", pages))
        requests = tuple(requests)
        _require(len(set(requests)) == len(requests), f"{label}: duplicate active request")
        _require(
            all(
                isinstance(owner, int) and not isinstance(owner, bool) and 0 <= owner < len(rows) for owner in requests
            ),
            f"{label}: invalid active request",
        )
        reserved = tuple(page for row in rows for page in row)
        _require(
            all(isinstance(page, int) and not isinstance(page, bool) and page > 0 for page in reserved)
            and len(set(reserved)) == len(reserved),
            f"{label}: invalid or overlapping FIA owner pages",
        )
        return set(requests), rows

    def _copy(self, live, wanted, label, *, protected=None, restore=False):
        torch = self.torch
        row_elements = prod(wanted.shape[1:])
        intervals = _protected_intervals(protected or (), row_elements)
        count = 0
        for device_chunk, checkpoint_chunk, start in _paired_chunks(live, wanted, self.scan_elements):
            observed = device_chunk.to(device="cpu", copy=True)
            if observed.is_floating_point() or observed.is_complex():
                _require(torch.isfinite(observed).all().item(), f"{label}: nonfinite state")
            if restore:
                for first, stop in intervals:
                    left, right = max(first, start) - start, min(stop, start + observed.numel()) - start
                    if left < right:
                        _require(
                            _identical(torch, observed[left:right], checkpoint_chunk[left:right]),
                            f"{label}: inactive state changed",
                        )
                # The null row stays zero even when both executions share a bug.
                stop = min(row_elements, start + observed.numel()) - start
                if stop > 0:
                    _require(torch.count_nonzero(observed[:stop]).item() == 0, f"{label}: null state must stay zero")
                    _require(
                        torch.count_nonzero(checkpoint_chunk[:stop]).item() == 0,
                        f"{label}: checkpoint null state must stay zero",
                    )
                device_chunk.copy_(checkpoint_chunk, non_blocking=False)
                checkpoint_chunk.copy_(observed)
            else:
                _require(_identical(torch, observed, checkpoint_chunk), f"{label}: replay tensor values differ")
            count += observed.numel() * observed.element_size()
        return count

    def capture_eager_and_restore(self, bundle, registry, active_requests, pages=None, *, label="stream/eager"):
        """Save eager expected bytes while restoring the device's previous step.

        Call after eager ``prepare_step -> forward -> synchronize -> finish_step``.
        On return, device tensors and host lifecycle hold the preceding step;
        ``saved`` contains eager results. Run replay prepare/finish next, then
        call :meth:`compare_replay`. No model execution may overlap this audit.
        """
        _require(not self._failed and not self._awaiting_replay, f"{label}: checkpoint is not ready for eager")
        try:
            items, states = self._bindings(bundle, registry, label)
            active, owner_pages = self._schedule(active_requests, bundle.pages if pages is None else pages, label)
            old_hosts = [
                {field: getattr(saved, field) for field in HOST_FIELDS} for saved in self.saved["compact"].states
            ]
            expected_hosts = [_host(state) for state in states]
            for index, (before, after) in enumerate(zip(old_hosts, expected_hosts, strict=True)):
                prefix = f"{label}/gdn{index}"
                _require(
                    before["pending"] is None and after["pending"] is None, f"{prefix}: incomplete host transaction"
                )
                _require(after["last_step"] > before["last_step"], f"{prefix}: lifecycle did not advance")
                _require(
                    len(before["generations"]) == len(after["generations"]) == len(owner_pages),
                    f"{prefix}: lifecycle owner count changed",
                )
                _require(after["initialized"] == before["initialized"], f"{prefix}: initialization changed")
                for owner, (previous, current) in enumerate(
                    zip(before["generations"], after["generations"], strict=True)
                ):
                    _require(current == previous + int(owner in active), f"{prefix}/owner{owner}: generation differs")
            copied = 0
            for prefix, live, wanted, kind, candidates in items:
                if kind == "owner":
                    _require(live.shape[0] == len(owner_pages) + 1, f"{prefix}: compact owner count changed")
                    changed_rows = {owner + 1 for owner in active}
                elif kind == "conv":
                    _require(len(candidates) == len(owner_pages), f"{prefix}: convolution owner count changed")
                    changed_rows = {slot for owner in active for slot in candidates[owner]}
                else:
                    _require(
                        all(page < live.shape[0] for row in owner_pages for page in row),
                        f"{prefix}: FIA page outside cache",
                    )
                    changed_rows = {page for owner in active for page in owner_pages[owner]}
                _require(all(0 < row < live.shape[0] for row in changed_rows), f"{prefix}: invalid active row")
                protected = set(range(live.shape[0])) - changed_rows
                copied += self._copy(live, wanted, prefix, protected=protected, restore=True)
            for state, host in zip(states, old_hosts, strict=True):
                _restore_host(state, host)
            compact = self.saved["compact"]
            self.saved["compact"] = replace(
                compact,
                states=tuple(
                    replace(saved, **host) for saved, host in zip(compact.states, expected_hosts, strict=True)
                ),
            )
            self._awaiting_replay = True
            return dict(
                copied_to_cpu_bytes=copied, restored_to_device_bytes=copied, scratch_elements=self.scan_elements
            )
        except BaseException:
            self._failed = True
            raise

    def compare_replay(self, bundle, registry, *, label="stream/replay"):
        """Compare all live bytes and host lifecycle; retain replay continuation."""
        _require(not self._failed and self._awaiting_replay, f"{label}: eager checkpoint is not pending")
        try:
            items, states = self._bindings(bundle, registry, label)
            for index, (state, saved) in enumerate(zip(states, self.saved["compact"].states, strict=True)):
                for field, actual in _host(state).items():
                    _require(actual == getattr(saved, field), f"{label}/gdn{index}/{field}: host contract differs")
            copied = sum(self._copy(live, wanted, prefix) for prefix, live, wanted, _, _ in items)
            self._awaiting_replay = False
            self.completed_steps += 1
            return dict(
                copied_to_cpu_bytes=copied, completed_steps=self.completed_steps, scratch_elements=self.scan_elements
            )
        except BaseException:
            self._failed = True
            raise
