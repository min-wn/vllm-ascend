# SPDX-License-Identifier: Apache-2.0
"""Independent official dense and compact full-model continued trajectories.

The official FP32 candidate banks live on CPU and are transferred unchanged,
one real layer at a time. These transfers are a correctness harness, never a
performance baseline. Conv/FIA caches remain independently resident on NPU.
"""

import hashlib
from contextlib import contextmanager

from multi_graph_gdn_cpu_audit import finite, identical
from multi_graph_gdn_q4_full_probe import deterministic_token_ids, tensor_record
from multi_graph_gdn_ready_full_lifecycle import TerminalDrain, terminal_counts

SCREEN_NRMSE = 0.005


def error_record(torch, actual, reference, *, label, step, owners, prefix=None):
    if actual.shape != reference.shape or actual.shape[0] != len(owners):
        raise ValueError("numerical comparison must retain the owner axis")
    a, b = actual.float(), reference.float()
    if not torch.isfinite(a).all().item() or not torch.isfinite(b).all().item():
        raise AssertionError(f"nonfinite numerical observation {label}/step{step}")
    dimensions = tuple(range(1, a.ndim))
    delta = a - b
    numerator, denominator = delta.square().sum(dimensions), b.square().sum(dimensions)
    ratios = (numerator / denominator.clamp_min(1e-30)).sqrt()
    worst = int(ratios.argmax().item())
    return dict(
        label=label,
        step=step,
        prefix=prefix,
        worst_owner=owners[worst],
        worst_request_nrmse=ratios[worst].item(),
        pooled_nrmse=(numerator.sum() / denominator.sum().clamp_min(1e-30)).sqrt().item(),
        max_abs=delta.abs().max().item(),
        reference_rms=b.square().mean().sqrt().item(),
        elements=a.numel(),
        screen_passed=ratios[worst].item() <= SCREEN_NRMSE,
    )


def candidate_records(torch, actual, reference, *, label, step):
    """Reduce all owner/prefix errors before one small host observation."""
    a, b = actual.float(), reference.float()
    if a.shape != (128, 4, 24, 128, 128) or b.shape != a.shape:
        raise ValueError("all 128 owners and four logical prefixes are required")
    if not torch.isfinite(a).all().item() or not torch.isfinite(b).all().item():
        raise AssertionError("nonfinite logical candidate")
    dims = (2, 3, 4)
    delta = a - b
    numerator, denominator = delta.square().sum(dims), b.square().sum(dims)
    maxima = delta.abs().amax(dims)
    ratios = (numerator / denominator.clamp_min(1e-30)).sqrt()
    packed = torch.stack((numerator, denominator, maxima, ratios)).cpu()
    rows = []
    for prefix in range(4):
        n, d, maximum, ratio = packed[:, :, prefix]
        worst = int(ratio.argmax().item())
        rows.append(
            dict(
                label=label,
                step=step,
                prefix=prefix + 1,
                worst_owner=worst,
                worst_request_nrmse=ratio[worst].item(),
                pooled_nrmse=(n.sum() / d.sum().clamp_min(1e-30)).sqrt().item(),
                max_abs=maximum.max().item(),
                elements=128 * 24 * 128 * 128,
                screen_passed=ratio[worst].item() <= SCREEN_NRMSE,
            )
        )
    return rows


class OfficialBanks:
    def __init__(self, torch, bundle):
        self.torch, self.bundle = torch, bundle
        self.dense = {}
        self.conv = {}
        for item in bundle.gdn_entries:
            base = item.base.cpu()
            bank = torch.empty((513, *base.shape[1:]), dtype=torch.float32, device="cpu")
            bank[0].zero_()
            bank[1:].view(128, 4, *base.shape[1:]).copy_(base[1:, None])
            self.dense[item.index] = bank
            self.conv[item.index] = item.conv.clone()
        self.fia = {item.index: (item.key.clone(), item.value.clone()) for item in bundle.fia_entries}
        self.scratch = torch.empty_like(next(iter(self.dense.values())), device=bundle.device)
        self.active, self.visits = set(), []

    @contextmanager
    def installed(self, model, requests):
        """Only storage movement surrounds unmodified official GDN operators."""
        torch, handles = self.torch, []
        self.active, self.visits = set(requests), []
        old_gdn, old_fia = [], []
        try:
            for item in self.bundle.gdn_entries:
                layer = model.language_model.model.layers[item.index]
                old_gdn.append((item, item.gdn.kv_cache))

                def load(module, inputs, *, item=item):
                    self.scratch.copy_(self.dense[item.index])
                    item.gdn.kv_cache = (self.conv[item.index], self.scratch)
                    self.visits.append(item.index)

                def store(module, inputs, output, *, item=item):
                    # Full model execution is serial by layer. Complete this
                    # official bank before the same scratch serves another.
                    observed = self.scratch.cpu()
                    prior = self.dense[item.index]
                    protected = [
                        0,
                        *(
                            1 + owner * 4 + token
                            for owner in range(128)
                            if owner not in self.active
                            for token in range(4)
                        ),
                    ]
                    if not identical(torch, observed[protected], prior[protected]):
                        raise AssertionError("official inactive state changed")
                    if torch.count_nonzero(observed[0]).item() or not finite(torch, observed):
                        raise AssertionError("official null or finite state contract failed")
                    prior.copy_(observed)

                handles.extend((layer.register_forward_pre_hook(load), layer.register_forward_hook(store)))
            for item in self.bundle.fia_entries:
                old_fia.append((item, item.backend.kv_cache, item.impl.key_cache, item.impl.value_cache))
                item.backend.kv_cache = self.fia[item.index]
                item.impl.key_cache, item.impl.value_cache = self.fia[item.index]
            yield
            if self.visits != [item.index for item in self.bundle.gdn_entries]:
                raise AssertionError("official recurrence did not visit every real GDN exactly once")
        finally:
            for hook in reversed(handles):
                hook.remove()
            for item, cache in old_gdn:
                item.gdn.kv_cache = cache
            for item, cache, key, value in old_fia:
                item.backend.kv_cache, item.impl.key_cache, item.impl.value_cache = cache, key, value


def qualify(args, torch, dist, model, bundle, factory, registry, forward, vocabulary, report, *, checkpoint):
    from multi_graph_gdn_q4_log_backend import installed_log_registry

    torch.npu.matmul.allow_hf32 = False
    torch.set_num_threads(8)  # Audit only; no timings in this phase.
    banks = OfficialBanks(torch, bundle)
    export = torch.empty((129, 4, 24, 128, 128), dtype=torch.float32, device=bundle.device)
    registry.mode = "deferred_tile_ready"
    rows, token_hash = [], hashlib.sha256()
    report["numeric_contract"] = dict(
        reference="unmodified installed official dense recurrence, independent 4-candidate FP32 banks",
        official_storage="CPU layer banks copied without conversion through one NPU scratch; not timed",
        candidate="compact FP32 ready, independent conv/FIA/SSM histories",
        capacities="same installed default dispatch and unchanged Q4 128-step trace",
        screen_nrmse=SCREEN_NRMSE,
        no_performance_comparison=True,
        dense_shape=tensor_record(banks.scratch),
        initialization="same complete official prefill, no reset between steps",
    )
    report["numeric_errors"] = rows

    def states(step, *, terminal=False):
        accepted = terminal_counts(128) if terminal else None
        for item in bundle.gdn_entries:
            backend = registry.backends[item.state.handle]
            banks.scratch.copy_(banks.dense[item.index])
            official = banks.scratch[1:].view(128, 4, 24, 128, 128)
            if not terminal:
                backend.deferred_module.export_candidates(item.state.base, item.state.logs, item.state.valid, export)
                if torch.count_nonzero(export[0]).item():
                    raise AssertionError("compact null candidate changed")
            if terminal:
                expected = torch.stack([official[o, accepted[o] - 1] for o in range(128)])
                rows.append(
                    error_record(
                        torch,
                        item.state.base[1:],
                        expected,
                        label=f"gdn{item.index}/terminal",
                        step=step,
                        owners=list(range(128)),
                    )
                )
            else:
                rows.extend(
                    candidate_records(torch, export[1:], official, label=f"gdn{item.index}/candidate", step=step)
                )
            # Conv and FIA differences are cross-algorithm observations too;
            # exact scheduling equivalence is established by the replay audit.
            rows.append(
                error_record(
                    torch,
                    item.conv[1:].view(128, 4, *item.conv.shape[1:]),
                    banks.conv[item.index][1:].view(128, 4, *item.conv.shape[1:]),
                    label=f"gdn{item.index}/conv",
                    step=step,
                    owners=list(range(128)),
                )
            )

    for step in bundle.trace.steps:
        capacity = args.selection["dispatched_tokens"][step["step"]]
        frame = factory.decode(step, capacity).refresh(step)
        ids = torch.zeros(capacity, dtype=torch.int64)
        ids[: step["batch"] * 4] = deterministic_token_ids(torch, step["batch"] * 4, vocabulary, step["input_seed"])
        token_hash.update(ids.numpy().tobytes())
        ids = ids.to(bundle.device)
        with banks.installed(model, step["request_ids"]):
            reference = tuple(value.clone() for value in forward(frame, ids))
            torch.npu.synchronize()
        registry.prepare_step(step["step"], step["request_ids"], step["accepted_from_previous"])
        with installed_log_registry(registry):
            actual = forward(frame, ids, registry=registry)
            torch.npu.synchronize()
        registry.finish_step(step["step"])
        for label, a, b in zip(("hidden", "residual"), actual, reference, strict=True):
            rows.append(
                error_record(
                    torch,
                    a[: step["batch"] * 4].view(step["batch"], 4, -1),
                    b[: step["batch"] * 4].view(step["batch"], 4, -1),
                    label=label,
                    step=step["step"],
                    owners=list(step["request_ids"]),
                )
            )
        states(step["step"])
        for item in bundle.fia_entries:
            expected = banks.fia[item.index]
            for label, a, b in zip(("key", "value"), (item.key, item.value), expected, strict=True):
                # Include all stored pages, including rejected candidates and
                # inactive owners. Page rows are explicitly labeled as pages.
                rows.append(
                    error_record(
                        torch,
                        a,
                        b,
                        label=f"fia{item.index}/{label}/page",
                        step=step["step"],
                        owners=list(range(a.shape[0])),
                    )
                )
        report.update(
            status="partial_independent_official_full_numeric_audit", numeric_completed_steps=step["step"] + 1
        )
        checkpoint()
        current = [r for r in rows if r["step"] == step["step"] and r["label"] in ("hidden", "residual")]
        print(
            f"rank={dist.get_rank()} official numeric step={step['step']} output_nrmse="
            f"{max(r['worst_request_nrmse'] for r in current):.8f}",
            flush=True,
        )
    terminal = TerminalDrain(torch, bundle, registry)
    terminal.run()
    torch.npu.synchronize()
    terminal.require_closed()
    states(128, terminal=True)
    report.update(
        status="complete_independent_official_full_numeric_audit",
        numeric_completed_steps=128,
        numeric_token_sha256=token_hash.hexdigest(),
        terminal_contract=terminal.contract,
        numeric_screen_passed=all(row["screen_passed"] for row in rows),
        numeric_output_screen_passed=all(
            row["screen_passed"] for row in rows if row["label"] in ("hidden", "residual")
        ),
        numerical_limitations=(
            "frozen teacher-forced token trace; hidden/residual/state comparison, not generation quality"
        ),
    )
    checkpoint()
