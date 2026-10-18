# SPDX-License-Identifier: Apache-2.0
"""CPU audits of complete hybrid snapshots, outside replay timing.

Include canonical state, deferred staging and host ownership. Chunked scans
avoid allocating a mask the size of a whole model cache.
"""

SCAN_ELEMENTS = 1 << 20
STATE_FIELDS = ("base", "logs", "valid")
HOST_FIELDS = ("state_identity", "generations", "initialized", "last_step", "pending")


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _chunks(tensor):
    if tensor.is_contiguous():
        flat = tensor.view(-1)
        for start in range(0, flat.numel(), SCAN_ELEMENTS):
            yield flat[start : start + SCAN_ELEMENTS]
    elif tensor.ndim:
        for child in tensor.unbind(0):
            yield from _chunks(child)
    else:
        yield tensor.reshape(-1)


def _finite(torch, tensor, label):
    _require(tensor.device.type == "cpu", f"{label}: audit requires a CPU snapshot")
    if tensor.is_floating_point() or tensor.is_complex():
        for chunk in _chunks(tensor):
            _require(torch.isfinite(chunk).all().item(), f"{label}: nonfinite state")


def _equal(torch, actual, expected, label):
    _require(actual.shape == expected.shape, f"{label}: shape changed")
    _require(actual.dtype == expected.dtype, f"{label}: dtype changed")
    _finite(torch, actual, f"{label}/actual")
    _finite(torch, expected, f"{label}/expected")
    _require(torch.equal(actual, expected), f"{label}: tensor values differ")


def _zero(torch, tensor, label):
    _finite(torch, tensor, label)
    for chunk in _chunks(tensor):
        _require(torch.count_nonzero(chunk).item() == 0, f"{label}: null state must stay zero")


def _staging_map(staging, label):
    result = {}
    for identity, logs, valid in staging:
        _require(identity not in result, f"{label}: duplicate staging identity")
        result[identity] = (logs, valid)
    return result


def _structure(actual, expected, label):
    for snapshot in (actual, expected):
        _require(set(snapshot) == {"compact", "conv", "fia"}, f"{label}: incomplete snapshot components")
        compact = snapshot["compact"]
        _require(len(compact.states) == len(compact.staging) == len(snapshot["conv"]), f"{label}: GDN layer mismatch")
    left, right = actual["compact"], expected["compact"]
    _require(left.registry_identity == right.registry_identity, f"{label}: registry identity changed")
    _require(len(left.states) == len(right.states), f"{label}: GDN layer count changed")
    _require(len(actual["fia"]) == len(expected["fia"]), f"{label}: FIA layer count changed")


def assert_complete_finite(torch, snapshot, label="complete"):
    """Reject nonfinite values anywhere, including currently inactive caches."""
    _structure(snapshot, snapshot, label)
    compact = snapshot["compact"]
    for index, (state, staging, conv) in enumerate(zip(compact.states, compact.staging, snapshot["conv"], strict=True)):
        _require(len(state.tensors) == len(STATE_FIELDS), f"{label}/gdn{index}: missing canonical state")
        for name, tensor in zip(STATE_FIELDS, state.tensors, strict=True):
            _finite(torch, tensor, f"{label}/gdn{index}/{name}")
        for lane, (logs, valid) in _staging_map(staging, f"{label}/gdn{index}").items():
            _finite(torch, logs, f"{label}/gdn{index}/staging{lane}/logs")
            _finite(torch, valid, f"{label}/gdn{index}/staging{lane}/valid")
        _finite(torch, conv, f"{label}/conv{index}")
    for index, pair in enumerate(snapshot["fia"]):
        _require(len(pair) == 2, f"{label}/fia{index}: key/value pair required")
        for name, tensor in zip(("key", "value"), pair, strict=True):
            _finite(torch, tensor, f"{label}/fia{index}/{name}")


def compare_complete_snapshots(torch, actual, expected, label="complete"):
    """Require identical cache contents, staging identities and host lifecycle."""
    _structure(actual, expected, label)
    for index, (got, wanted) in enumerate(zip(actual["compact"].states, expected["compact"].states, strict=True)):
        prefix = f"{label}/gdn{index}"
        for field in HOST_FIELDS:
            _require(getattr(got, field) == getattr(wanted, field), f"{prefix}/{field}: host contract differs")
        _require(len(got.tensors) == len(wanted.tensors) == len(STATE_FIELDS), f"{prefix}: missing canonical state")
        for name, left, right in zip(STATE_FIELDS, got.tensors, wanted.tensors, strict=True):
            _equal(torch, left, right, f"{prefix}/{name}")
        got_staging = _staging_map(actual["compact"].staging[index], prefix)
        wanted_staging = _staging_map(expected["compact"].staging[index], prefix)
        _require(got_staging.keys() == wanted_staging.keys(), f"{prefix}: staging identities changed")
        for identity, pair in got_staging.items():
            for name, left, right in zip(("logs", "valid"), pair, wanted_staging[identity], strict=True):
                _equal(torch, left, right, f"{prefix}/staging{identity}/{name}")
    for index, (left, right) in enumerate(zip(actual["conv"], expected["conv"], strict=True)):
        _equal(torch, left, right, f"{label}/conv{index}")
    for index, (left, right) in enumerate(zip(actual["fia"], expected["fia"], strict=True)):
        _require(len(left) == len(right) == 2, f"{label}/fia{index}: key/value pair required")
        for name, got, wanted in zip(("key", "value"), left, right, strict=True):
            _equal(torch, got, wanted, f"{label}/fia{index}/{name}")


def verify_untouched_owners(torch, before, after, active_requests, pages, label="untouched"):
    """Check paused owners, every null row and all non-active FIA pages.

    ``pages`` is OwnerPages or its ``pages`` tuple, indexed by stable request ID.
    Convolution slots come from each state's registered candidates. Compact rows
    are request ID + 1. Active rows may change freely.
    """
    _structure(after, before, label)
    owner_pages = tuple(tuple(row) for row in getattr(pages, "pages", pages))
    active = tuple(active_requests)
    _require(len(set(active)) == len(active), f"{label}: duplicate active request")
    _require(
        all(
            isinstance(owner, int) and not isinstance(owner, bool) and 0 <= owner < len(owner_pages) for owner in active
        ),
        f"{label}: invalid active request",
    )
    reserved = [page for row in owner_pages for page in row]
    _require(
        all(isinstance(page, int) and not isinstance(page, bool) and page > 0 for page in reserved)
        and len(set(reserved)) == len(reserved),
        f"{label}: FIA owner pages overlap or use null",
    )
    paused = set(range(len(owner_pages))) - set(active)
    assert_complete_finite(torch, before, f"{label}/before")
    assert_complete_finite(torch, after, f"{label}/after")
    for index, (left, right) in enumerate(zip(before["compact"].states, after["compact"].states, strict=True)):
        prefix = f"{label}/gdn{index}"
        _require(left.state_identity == right.state_identity, f"{prefix}: state identity changed")
        candidates = left.state_identity[1]
        _require(len(candidates) == len(owner_pages), f"{prefix}: owner count differs")
        for field in ("generations", "initialized"):
            first, second = getattr(left, field), getattr(right, field)
            _require(len(first) == len(second) == len(owner_pages), f"{prefix}/{field}: owner count differs")
            for owner in paused:
                _require(first[owner] == second[owner], f"{prefix}/{field}/owner{owner}: paused lifecycle changed")
        banks = [
            (name, got, wanted) for name, got, wanted in zip(STATE_FIELDS, right.tensors, left.tensors, strict=True)
        ]
        left_staging = _staging_map(before["compact"].staging[index], prefix)
        right_staging = _staging_map(after["compact"].staging[index], prefix)
        _require(left_staging.keys() == right_staging.keys(), f"{prefix}: staging identities changed")
        for identity, pair in right_staging.items():
            banks.extend(
                (f"staging{identity}/{name}", got, wanted)
                for name, got, wanted in zip(("logs", "valid"), pair, left_staging[identity], strict=True)
            )
        for name, got, wanted in banks:
            _require(
                got.shape == wanted.shape and got.shape[0] == len(owner_pages) + 1, f"{prefix}/{name}: owner shape"
            )
            _equal(torch, got[0], wanted[0], f"{prefix}/{name}/null")
            _zero(torch, got[0], f"{prefix}/{name}/null")
            for owner in paused:
                _equal(torch, got[owner + 1], wanted[owner + 1], f"{prefix}/{name}/owner{owner}")
        got_conv, wanted_conv = after["conv"][index], before["conv"][index]
        _equal(torch, got_conv[0], wanted_conv[0], f"{label}/conv{index}/null")
        _zero(torch, got_conv[0], f"{label}/conv{index}/null")
        for owner in paused:
            for slot in candidates[owner]:
                _equal(torch, got_conv[slot], wanted_conv[slot], f"{label}/conv{index}/owner{owner}/slot{slot}")
    active_pages = {page for owner in active for page in owner_pages[owner]}
    for index, (left, right) in enumerate(zip(before["fia"], after["fia"], strict=True)):
        for name, wanted, got in zip(("key", "value"), left, right, strict=True):
            prefix = f"{label}/fia{index}/{name}"
            _require(got.shape == wanted.shape, f"{prefix}: shape changed")
            _require(all(page < got.shape[0] for page in reserved), f"{prefix}: reserved page outside cache")
            for page in range(got.shape[0]):
                if page not in active_pages:
                    _equal(torch, got[page], wanted[page], f"{prefix}/page{page}")
            _zero(torch, got[0], f"{prefix}/null")
