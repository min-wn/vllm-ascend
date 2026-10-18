# SPDX-License-Identifier: Apache-2.0
"""Explicit benchmark-only compact SSM ownership for real TP2 GDN layers.

The supplied dense cache is an initialization source, never a current-state
consumer after import. Official convolution stays with the caller. Host lifecycle
checks validate a declared fixture schedule; native kernels do not validate the
values of arbitrary device metadata or implement retry/fallback transactions.
"""

import hashlib
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from multi_graph_gdn_q4_backend import load_extension
from multi_graph_gdn_q4_log_probe import request_pipeline_operations

TILE_MODES = ("deferred_tile_delayed", "deferred_tile_ready", "deferred_tile_separate")
COMPACT_MODES = (
    "padding",
    "serial",
    "pipeline",
    "deferred_serial",
    "deferred_overlap",
    "deferred_pipeline",
) + TILE_MODES


class CompactSSMHandle:
    """Opaque registry identity, deliberately without any Tensor interface."""

    __slots__ = ()


def _snapshot_tensor(torch, tensor, destination):
    if destination is None:
        return tensor.clone()
    if torch.device(destination).type != "cpu":
        raise ValueError("snapshot destination must be CPU or the default live device")
    return tensor.to(device="cpu", copy=True)


@dataclass(frozen=True)
class Q4LogSnapshot:
    tensors: tuple
    state_identity: tuple
    generations: tuple
    initialized: tuple
    last_step: int
    pending: tuple | None


@dataclass(frozen=True)
class CompactRegistrySnapshot:
    states: tuple
    staging: tuple
    registry_identity: tuple


class Q4LogState:
    """One FP32 base and pending logs per explicitly declared request owner."""

    def __init__(self, torch, dense_state, request_candidate_slots, initial_accepted=None):
        self.torch = torch
        if (
            not isinstance(dense_state, torch.Tensor)
            or dense_state.dtype != torch.float32
            or dense_state.ndim != 4
            or dense_state.shape[1:] != (24, 128, 128)
            or dense_state.stride()[1:] != (16384, 128, 1)
            or dense_state.stride(0) < 24 * 16384
        ):
            raise ValueError("compact state requires an FP32 TP2 dense initialization cache")
        candidates = tuple(tuple(slots) for slots in request_candidate_slots)
        if not 1 <= len(candidates) <= 128 or any(len(slots) != 4 for slots in candidates):
            raise ValueError("declare four candidate slots for each of 1..128 request owners")
        flat = tuple(slot for slots in candidates for slot in slots)
        if any(
            isinstance(slot, bool) or not isinstance(slot, int) or not 1 <= slot < dense_state.shape[0] for slot in flat
        ) or len(set(flat)) != len(flat):
            raise ValueError("candidate slots must be unique in-range non-null integers")
        accepted = tuple(initial_accepted) if initial_accepted is not None else (1,) * len(candidates)
        if len(accepted) != len(candidates) or any(
            isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 4 for count in accepted
        ):
            raise ValueError("one initial accepted count in 1..4 per declared request required")
        self.dense_state = dense_state
        self.handle = None
        self.request_candidate_slots = candidates
        self.initial_accepted = accepted
        self.dense_identity = (dense_state.data_ptr(), tuple(dense_state.shape), tuple(dense_state.stride()))
        count, device = len(candidates), dense_state.device
        # Import the actually selected dense prefix once, including fixtures
        # whose initial four candidates are not equal. No per-step state gather.
        selected = [0, *(slots[count - 1] for slots, count in zip(candidates, accepted, strict=True))]
        self.base = dense_state.index_select(0, torch.tensor(selected, dtype=torch.long, device=device)).contiguous()
        self.logs = torch.zeros((count + 1, 24, 1056), dtype=torch.float32, device=device)
        self.valid = torch.zeros((count + 1, 24, 8), dtype=torch.int32, device=device)
        lookup = [-1] * dense_state.shape[0]
        lookup[0] = 0
        for owner, slots in enumerate(candidates, start=1):
            for slot in slots:
                lookup[slot] = owner
        self.device_owner_lookup = torch.tensor(lookup, dtype=torch.int32, device=device)
        self.mutable_tensors = (self.base, self.logs, self.valid)
        self.generations = [0] * count
        self.initialized = [True] * count
        self.last_step = -1
        self.pending = None
        self._export = None
        self._snapshot_identity = (self.dense_identity, candidates, accepted)
        self._base_identity = self._current_base_identity()

    @classmethod
    def from_base(cls, torch, base, request_candidate_slots):
        """Adopt a coherently initialized FP32 base without any dense source.

        Row zero is the preserved null state. Request owner i uses base row
        i+1 and the four explicit metadata candidate IDs supplied by the caller.
        Convolution is independently allocated and retains those candidate IDs.
        """
        candidates = tuple(tuple(slots) for slots in request_candidate_slots)
        if (
            not isinstance(base, torch.Tensor)
            or base.dtype != torch.float32
            or base.ndim != 4
            or base.shape != (len(candidates) + 1, 24, 128, 128)
            or base.stride()[1:] != (16384, 128, 1)
            or base.stride(0) < 24 * 16384
            or not 1 <= len(candidates) <= 128
            or any(len(slots) != 4 for slots in candidates)
        ):
            raise ValueError("direct compact ownership requires FP32 [owners+1,24,128,128] and explicit quartets")
        flat = tuple(slot for slots in candidates for slot in slots)
        if any(isinstance(slot, bool) or not isinstance(slot, int) or slot < 1 for slot in flat) or len(
            set(flat)
        ) != len(flat):
            raise ValueError("candidate slots must be unique positive non-null integers")
        state = cls.__new__(cls)
        state.torch, state.base = torch, base
        state.dense_state = state.dense_identity = None
        state.handle = CompactSSMHandle()
        state.request_candidate_slots = candidates
        state.initial_accepted = (1,) * len(candidates)
        state.logs = torch.zeros((len(candidates) + 1, 24, 1056), dtype=torch.float32, device=base.device)
        state.valid = torch.zeros((len(candidates) + 1, 24, 8), dtype=torch.int32, device=base.device)
        lookup = [-1] * (max(flat) + 1)
        lookup[0] = 0
        for owner, slots in enumerate(candidates, start=1):
            for slot in slots:
                lookup[slot] = owner
        state.device_owner_lookup = torch.tensor(lookup, dtype=torch.int32, device=base.device)
        state.mutable_tensors = (state.base, state.logs, state.valid)
        state.generations = [0] * len(candidates)
        state.initialized = [True] * len(candidates)
        state.last_step, state.pending = -1, None
        state._export = None
        state._snapshot_identity = (state.handle, candidates)
        state._base_identity = state._current_base_identity()
        return state

    def _current_base_identity(self):
        return (
            self.base.data_ptr(),
            tuple(self.base.shape),
            tuple(self.base.stride()),
            self.base.dtype,
            self.base.device,
        )

    def snapshot(self, destination=None):
        return Q4LogSnapshot(
            tensors=tuple(_snapshot_tensor(self.torch, tensor, destination) for tensor in self.mutable_tensors),
            state_identity=self._snapshot_identity,
            generations=tuple(self.generations),
            initialized=tuple(self.initialized),
            last_step=self.last_step,
            pending=self.pending,
        )

    def _validate_snapshot(self, snapshot):
        if not isinstance(snapshot, Q4LogSnapshot) or snapshot.state_identity != self._snapshot_identity:
            raise ValueError("snapshot belongs to a different compact cache")
        if (
            len(snapshot.generations) != len(self.generations)
            or len(snapshot.initialized) != len(self.initialized)
            or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in snapshot.generations)
            or any(not isinstance(value, bool) for value in snapshot.initialized)
        ):
            raise ValueError("snapshot lifecycle contract changed")
        if len(snapshot.tensors) != len(self.mutable_tensors) or any(
            actual.shape != saved.shape
            or actual.dtype != saved.dtype
            or (actual.device != saved.device and saved.device.type != "cpu")
            for actual, saved in zip(self.mutable_tensors, snapshot.tensors, strict=True)
        ):
            raise ValueError("snapshot tensor contract changed")

    def restore(self, snapshot):
        self._validate_snapshot(snapshot)
        for target, saved in zip(self.mutable_tensors, snapshot.tensors, strict=True):
            target.copy_(saved)
        self.generations[:] = snapshot.generations
        self.initialized[:] = snapshot.initialized
        self.last_step, self.pending = snapshot.last_step, snapshot.pending

    def prepare_step(self, step_id, request_ids, accepted, generations=None):
        """Validate host fixture ownership before one eager/graph execution.

        Request IDs are zero-based indices into request_candidate_slots. Call
        finish_step only after device completion. These are optional host guards,
        not code executed during replay or native stale-generation detection.
        """
        if self.pending is not None:
            raise RuntimeError("compact transaction already in flight")
        if isinstance(step_id, bool) or not isinstance(step_id, int) or step_id <= self.last_step:
            raise ValueError("compact step must be newer than the last completed step")
        requests, accepted = tuple(request_ids), tuple(accepted)
        if len(requests) != len(accepted) or len(set(requests)) != len(requests):
            raise ValueError("each active owner needs exactly one acceptance decision")
        if any(
            isinstance(owner, bool) or not isinstance(owner, int) or not 0 <= owner < len(self.generations)
            for owner in requests
        ):
            raise ValueError("unknown compact request owner")
        if any(isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 4 for count in accepted):
            raise ValueError("accepted counts must be in 1..4")
        expected = tuple(self.generations[owner] for owner in requests)
        supplied = tuple(generations) if generations is not None else expected
        if supplied != expected or any(isinstance(value, bool) or not isinstance(value, int) for value in supplied):
            raise ValueError("acceptance belongs to a stale or future request generation")
        self.pending = (step_id, requests, accepted, expected)

    def finish_step(self, step_id):
        if (
            isinstance(step_id, bool)
            or not isinstance(step_id, int)
            or self.pending is None
            or step_id != self.pending[0]
        ):
            raise RuntimeError("no matching compact transaction in flight")
        for owner in self.pending[1]:
            self.generations[owner] += 1
        self.last_step, self.pending = step_id, None

    def export(self, module):
        """Diagnostic all-prefix materialization, explicitly outside recurrence."""
        if self._export is None:
            self._export = self.torch.empty(
                (self.base.shape[0], 4, 24, 128, 128), dtype=self.torch.float32, device=self.base.device
            )
        module.export_candidates(self.base, self.logs, self.valid, self._export)
        return self._export


class Q4LogBackend:
    def __init__(self, state, log_module, product_module, mode="padding", deferred_module=None, tile_module=None):
        self.state, self.log_module, self.product_module = state, log_module, product_module
        self.deferred_module = deferred_module
        self.tile_module = tile_module
        self.buffers = {}
        self._exclusive_buffers = None
        self._exclusive_capacity = None
        self._exclusive_commit_stream = None
        self.calls = 0
        self.sources = None
        self.original = None
        self._pending_commit = None
        self._failed = False
        self.mode = mode

    def reserve_exclusive_capacity(self, max_batch=128, commit_stream=None):
        """Reserve one workspace for metadata lanes that never execute together.

        The caller promises that eager invocations and captured graphs sharing
        this backend never overlap and join all commits before the next use.
        A supplied commit stream is shared by all metadata lanes and may also
        serve independent layers in the same scope. Their readiness/completion
        events remain distinct. Without it each lane allocates its own stream.
        Request halves use disjoint product/output ranges and unique owners.
        Reservation must precede every recurrence and cannot be resized later.
        """
        if isinstance(max_batch, bool) or not isinstance(max_batch, int) or not 1 <= max_batch <= 128:
            raise ValueError("exclusive compact capacity must be an integer in 1..128")
        if self._exclusive_buffers is not None or self.buffers or self.calls or self._pending_commit is not None:
            raise RuntimeError("reserve exclusive compact workspace exactly once before any recurrence")
        if self._failed:
            raise RuntimeError("failed compact state transaction cannot reserve workspace")
        torch, device = self.state.torch, self.state.base.device
        reserved = dict(
            products=torch.empty((max_batch, 24, 128, 8), dtype=torch.float32, device=device),
            output=torch.zeros((max_batch * 4, 24, 128), dtype=torch.bfloat16, device=device),
            ids=torch.empty((max_batch * 4,), dtype=torch.int32, device=device),
        )
        if self.deferred_module is not None:
            reserved.update(new_logs=torch.zeros_like(self.state.logs), new_valid=torch.zeros_like(self.state.valid))
        self._exclusive_buffers, self._exclusive_capacity = reserved, max_batch
        self._exclusive_commit_stream = commit_stream
        return self

    def _staging_buffers(self):
        if self._exclusive_buffers is not None:
            return {None: self._exclusive_buffers} if "new_logs" in self._exclusive_buffers else {}
        return {identity: lane for identity, lane in self.buffers.items() if "new_logs" in lane}

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, mode):
        if self._pending_commit is not None:
            raise RuntimeError("cannot change compact mode before the pending state commit is joined")
        if mode not in COMPACT_MODES:
            raise ValueError("unknown compact backend execution mode")
        if mode.startswith("deferred_") and self.deferred_module is None:
            raise ValueError("deferred compact execution requires an explicit deferred extension")
        if mode in TILE_MODES and self.tile_module is None:
            raise ValueError("tile compact execution requires an explicit tile extension")
        self._mode = mode

    def __call__(self, query, key, value, initial_state, **kwargs):
        if self._failed:
            raise RuntimeError("failed compact state transaction cannot be reused")
        if self._pending_commit is not None:
            raise RuntimeError("previous compact state commit must be launched and joined before another recurrence")
        torch = self.state.torch
        accepted = kwargs.get("num_accepted_tokens")
        if accepted is None:
            raise ValueError("compact backend refuses non-spec or implicit dense-state fallback")
        if set(kwargs) - {"num_accepted_tokens", "actual_seq_lengths", "ssm_state_indices", "g", "beta", "scale"}:
            raise ValueError("compact backend refuses unknown recurrence options")
        if self.state.handle is not None:
            if initial_state is not self.state.handle:
                raise ValueError("recurrence must use its exact registered compact SSM handle")
        else:
            if initial_state is not self.state.dense_state:
                raise ValueError("recurrence must identify the explicitly registered dense initialization source")
            if (
                initial_state.data_ptr(),
                tuple(initial_state.shape),
                tuple(initial_state.stride()),
            ) != self.state.dense_identity:
                raise ValueError("registered dense cache storage or layout changed")
        if self.state._current_base_identity() != self.state._base_identity:
            raise ValueError("compact base storage, precision or layout changed")
        batch = accepted.numel()
        sequence, ids = kwargs.get("actual_seq_lengths"), kwargs.get("ssm_state_indices")
        gate, beta = kwargs.get("g"), kwargs.get("beta")
        if not 1 <= batch <= 128 or query.shape != (batch * 4, 8, 128) or key.shape != query.shape:
            raise ValueError("compact recurrence requires Q4 TP2 Q/K at fixed capacity")
        if (
            value.shape != (batch * 4, 24, 128)
            or beta is None
            or gate is None
            or beta.shape != gate.shape
            or beta.shape != (batch * 4, 24)
        ):
            raise ValueError("compact recurrence V/beta/gate dimensions disagree")
        if kwargs.get("scale") != 128**-0.5:
            raise ValueError("compact recurrence requires the official query scale")
        if sequence is None or ids is None or sequence.numel() != batch + 1 or ids.numel() != batch * 4:
            raise ValueError("complete fixed-address compact metadata required")
        if any(t.dtype != torch.bfloat16 for t in (query, key, value, beta)) or gate.dtype != torch.float32:
            raise ValueError("compact recurrence requires BF16 Q/K/V/beta and FP32 gate")
        if any(t.dtype != torch.int32 or t.ndim != 1 for t in (sequence, ids, accepted)):
            raise ValueError("compact metadata must be one-dimensional int32 tensors")
        inputs = (query, key, value, beta, gate, sequence, ids, accepted)
        if any(not tensor.is_contiguous() or tensor.device != self.state.base.device for tensor in inputs):
            raise ValueError("contiguous inputs and metadata on the compact state device required")
        if self.mode in ("serial", "pipeline", "deferred_pipeline") and batch < 2:
            raise ValueError("two-range compact modes require at least two captured requests")
        if self._exclusive_capacity is not None and batch > self._exclusive_capacity:
            raise ValueError("captured requests exceed the reserved exclusive compact capacity")
        identity = (sequence.data_ptr(), ids.data_ptr(), accepted.data_ptr(), batch)
        if identity not in self.buffers:
            if self._exclusive_buffers is None:
                self.buffers[identity] = dict(
                    products=torch.empty((batch, 24, 128, 8), dtype=torch.float32, device=query.device),
                    output=torch.zeros_like(value),
                    ids=torch.empty_like(ids),
                )
            else:
                reserved = self._exclusive_buffers
                self.buffers[identity] = dict(
                    products=reserved["products"][:batch],
                    output=reserved["output"][: batch * 4],
                    ids=reserved["ids"][: batch * 4],
                )
        buffers = self.buffers[identity]
        products, output, compact_ids = (buffers[name] for name in ("products", "output", "ids"))

        def prepare():
            # Only tiny metadata/Q/K tensors are materialized during execution.
            # The fixture must declare valid device values: kernels skip invalid
            # IDs rather than reporting lifecycle or ownership failures.
            torch.index_select(self.state.device_owner_lookup, 0, ids.long(), out=compact_ids)
            output.zero_()
            return query.float(), key.float()

        def product_range(q_float, k_float, start, stop):
            self.product_module.products(
                self.state.base, q_float, k_float, sequence, compact_ids, accepted, products, 20, start, stop
            )

        def step_range(start, stop):
            self.log_module.step(
                products,
                query,
                key,
                value,
                beta,
                gate,
                self.state.base,
                sequence,
                compact_ids,
                accepted,
                output,
                self.state.logs,
                self.state.valid,
                40,
                start,
                stop,
            )

        if self.mode.startswith("deferred_"):
            if "new_logs" not in buffers:
                reserved = self._exclusive_buffers
                buffers.update(
                    new_logs=reserved["new_logs"] if reserved is not None else torch.zeros_like(self.state.logs),
                    new_valid=reserved["new_valid"] if reserved is not None else torch.zeros_like(self.state.valid),
                    commit_stream=(
                        self._exclusive_commit_stream
                        if self._exclusive_commit_stream is not None
                        else torch.npu.Stream()
                    ),
                    solve_ready=torch.npu.Event(enable_timing=False),
                    commit_done=torch.npu.Event(enable_timing=False),
                )

            def solve_range(start, stop):
                self.deferred_module.solve(
                    products,
                    query,
                    key,
                    value,
                    beta,
                    gate,
                    self.state.base,
                    sequence,
                    compact_ids,
                    accepted,
                    output,
                    self.state.logs,
                    self.state.valid,
                    buffers["new_logs"],
                    buffers["new_valid"],
                    40,
                    start,
                    stop,
                )

            if self.mode in TILE_MODES:
                q_float, k_float = prepare()
                if self.mode == "deferred_tile_separate":
                    product_range(q_float, k_float, 0, batch)
                    self.tile_module.consume(
                        products,
                        query,
                        key,
                        value,
                        beta,
                        gate,
                        self.state.base,
                        sequence,
                        compact_ids,
                        accepted,
                        output,
                        self.state.logs,
                        self.state.valid,
                        buffers["new_logs"],
                        buffers["new_valid"],
                        40,
                    )
                else:
                    # Both mixed-core controls share task ownership and math;
                    # only ready mode exposes each request's product early.
                    self.tile_module.solve(
                        self.state.base,
                        q_float,
                        k_float,
                        query,
                        key,
                        value,
                        beta,
                        gate,
                        sequence,
                        compact_ids,
                        accepted,
                        products,
                        output,
                        self.state.logs,
                        self.state.valid,
                        buffers["new_logs"],
                        buffers["new_valid"],
                        20,
                        self.mode == "deferred_tile_ready",
                    )
            elif self.mode == "deferred_pipeline":
                # Distinct dispatch from fused range execution: a captured
                # deferred solve must never retain a later fused-step closure.
                if "deferred_range_dispatch" not in buffers:
                    buffers["deferred_range_dispatch"] = {}
                    deferred_dispatch = buffers["deferred_range_dispatch"]
                    buffers["deferred_range_operations"] = request_pipeline_operations(
                        torch,
                        batch,
                        lambda: deferred_dispatch["prepare"](),
                        lambda *args: deferred_dispatch["products"](*args),
                        lambda *args: deferred_dispatch["solve"](*args),
                        output,
                    )
                buffers["deferred_range_dispatch"].update(prepare=prepare, products=product_range, solve=solve_range)
                # P_A -> (S_A || P_B) -> S_B -> root output-ready join.
                # Official gated Norm may consume every output row on return.
                buffers["deferred_range_operations"]["compact_pipeline"]()
            else:
                q_float, k_float = prepare()
                product_range(q_float, k_float, 0, batch)
                solve_range(0, batch)

            def commit():
                self.deferred_module.commit(
                    self.state.base,
                    sequence,
                    compact_ids,
                    accepted,
                    self.state.logs,
                    self.state.valid,
                    buffers["new_logs"],
                    buffers["new_valid"],
                )

            if self.mode == "deferred_serial":
                commit()
            else:
                # The layer's output-projection prehook launches this after its
                # gated Norm. Its endpoint must join the transaction explicitly.
                self._pending_commit = dict(
                    operation=commit, buffers=buffers, root=torch.npu.current_stream(), launched=False
                )
        elif self.mode == "padding":
            q_float, k_float = prepare()
            product_range(q_float, k_float, 0, batch)
            step_range(0, batch)
        else:
            # Streams/events persist per metadata lane. Current tensor operands
            # are supplied afresh so eager calls never retain old Q/K temporaries.
            if "range_dispatch" not in buffers:
                buffers["range_dispatch"] = {}
                dispatch = buffers["range_dispatch"]
                buffers["range_operations"] = request_pipeline_operations(
                    torch,
                    batch,
                    lambda: dispatch["prepare"](),
                    lambda *args: dispatch["products"](*args),
                    lambda *args: dispatch["step"](*args),
                    output,
                )
            buffers["range_dispatch"].update(prepare=prepare, products=product_range, step=step_range)
            buffers["range_operations"][f"compact_{self.mode}"]()
        self.calls += 1
        return output

    def launch_pending_commit(self):
        """Output-projection prehook: publish gated-Norm readiness to AIV."""
        pending = self._pending_commit
        if pending is None:
            return False
        if pending["launched"]:
            raise RuntimeError("compact state commit was already launched")
        torch, buffers = self.state.torch, pending["buffers"]
        root = torch.npu.current_stream()
        buffers["solve_ready"].record(root)
        try:
            with torch.npu.stream(buffers["commit_stream"]):
                buffers["commit_stream"].wait_event(buffers["solve_ready"])
                pending["operation"]()
                buffers["commit_done"].record(buffers["commit_stream"])
            pending["launched"] = True
        except BaseException:
            # A partially submitted commit cannot safely be retried. Error-only
            # synchronization closes outstanding work before restoring the op.
            self._failed = True
            try:
                torch.npu.synchronize()
            finally:
                self._pending_commit = None
            raise
        return True

    def finish_pending_commit(self):
        """Layer endpoint: join state publication before output masking/return."""
        pending = self._pending_commit
        if pending is None:
            return False
        if not pending["launched"]:
            raise RuntimeError("output-projection hook did not launch the pending compact state commit")
        self.state.torch.npu.current_stream().wait_event(pending["buffers"]["commit_done"])
        self._pending_commit = None
        return True

    def _cleanup_pending_commit(self):
        """Error/undrained-context cleanup; never silently count it as a result."""
        pending = self._pending_commit
        if pending is None:
            return
        self._failed = True
        with self.state.torch.npu.stream(pending["root"]):
            if not pending["launched"]:
                self.launch_pending_commit()
            self.finish_pending_commit()
        self.state.torch.npu.synchronize()

    def describe(self):
        scratch = [self._exclusive_buffers] if self._exclusive_buffers is not None else self.buffers.values()
        return dict(
            backend="experimental_explicit_q4_log_state",
            mode=self.mode,
            observed_python_calls=self.calls,
            graph_replay_calls_are_not_counted=True,
            request_owners=len(self.state.request_candidate_slots),
            persistent_lanes=len(self.buffers),
            compact_state_bytes=sum(t.numel() * t.element_size() for t in self.state.mutable_tensors),
            scratch_bytes=sum(
                t.numel() * t.element_size()
                for lane in scratch
                for key, t in lane.items()
                if key in ("products", "output", "ids", "new_logs", "new_valid")
            ),
            exclusive_workspace_capacity=self._exclusive_capacity,
            exclusive_commit_stream_shared=self._exclusive_commit_stream is not None,
            exclusive_workspace_contract=(
                "caller guarantees metadata lanes and captured graphs never overlap; commits joined before reuse"
                if self._exclusive_buffers is not None
                else None
            ),
            dense_initialization_source_becomes_stale=self.state.dense_state is not None,
            dense_initialization_source_retained=self.state.dense_state is not None,
            opaque_compact_handle=self.state.handle is not None,
            non_spec_and_fallback="rejected",
            all_prefix_export="explicit diagnostic only; not included in recurrence",
            device_metadata_value_validation=False,
            lifecycle="optional host prepare/finish guards; no native generation or retry validation",
            preparation="device owner lookup, output clear, Q/K casts included in callable",
            deferred_extension_loaded=self.deferred_module is not None,
            tile_extension_loaded=self.tile_module is not None,
            deferred_commit_pending=self._pending_commit is not None,
            deferred_staging="scratch only; active new logs published into canonical old bank by commit",
            deferred_overlap="output-projection prehook launches commit; full-layer endpoint must join",
            deferred_pipeline="fixed request halves overlap products/solve; all solves joined before gated Norm",
            deferred_tile="request tasks; separate/delayed/ready controls share deferred commit and endpoint join",
            libraries=self.sources,
            production_default_changed=False,
        )

    @contextmanager
    def suspended(self):
        """Run the explicit official dense control while retaining our scratch."""
        if self._pending_commit is not None:
            raise RuntimeError("cannot enter official dense control before joining compact state publication")
        import vllm_ascend.ops.gdn as gdn

        if self.original is None or gdn.recurrent_gated_delta_rule is not self:
            raise RuntimeError("only the currently installed compact backend can be suspended")
        gdn.recurrent_gated_delta_rule = self.original
        try:
            yield self.original
        finally:
            gdn.recurrent_gated_delta_rule = self


class CompactSSMRegistry:
    """Exact-handle dispatcher for a manually owned full hybrid benchmark.

    This registry does not modify the production KV manager or implement dense
    consumers. The caller binds each real conv tensor and opaque SSM handle,
    refreshes declared metadata, and joins all commits at the model endpoint.
    """

    def __init__(self):
        self.entries = []
        self.backends = {}
        self.calls = 0
        self._installed = False

    def register(self, handle, backend, out_proj, name=None):
        if self._installed:
            raise RuntimeError("cannot change an installed compact registry")
        if (
            type(handle) is not CompactSSMHandle
            or backend.state.handle is not handle
            or backend.state.dense_state is not None
        ):
            raise ValueError("register an exact direct-base compact handle and its backend")
        if handle in self.backends or any(backend is entry[1] or out_proj is entry[2] for entry in self.entries):
            raise ValueError("compact handle, backend and output projection must each have one owner")
        if not callable(getattr(out_proj, "register_forward_pre_hook", None)):
            raise ValueError("compact layer needs a real output projection hook interface")
        name = str(name) if name is not None else f"gdn_{len(self.entries)}"
        if any(name == entry[3] for entry in self.entries):
            raise ValueError("compact layer names must be unique")
        for _, previous, _, _ in self.entries:
            if previous.state.base.device != backend.state.base.device:
                raise ValueError("all compact layers in one registry must share a device")
            for tensor in backend.state.mutable_tensors:
                for old in previous.state.mutable_tensors:

                    def span(value):
                        elements = (
                            (value.shape[0] - 1) * value.stride(0) + 24 * 128 * 128
                            if value.ndim == 4
                            else value.numel()
                        )
                        return value.data_ptr(), value.data_ptr() + elements * value.element_size()

                    left, right = span(tensor), span(old)
                    if left[0] < right[1] and right[0] < left[1]:
                        raise ValueError("different GDN layers must not alias compact state storage")
        self.entries.append((handle, backend, out_proj, name))
        self.backends[handle] = backend
        return backend

    def __call__(self, query, key, value, initial_state, **kwargs):
        if type(initial_state) is not CompactSSMHandle or initial_state not in self.backends:
            raise ValueError("unknown compact handle or unexpected dense Tensor fallback")
        if kwargs.get("num_accepted_tokens") is None:
            raise ValueError("multilayer compact dispatcher refuses non-spec execution")
        result = self.backends[initial_state](query, key, value, initial_state, **kwargs)
        self.calls += 1
        return result

    @property
    def mode(self):
        modes = {backend.mode for _, backend, _, _ in self.entries}
        if len(modes) > 1:
            raise RuntimeError("registered compact layers have inconsistent execution modes")
        return next(iter(modes), None)

    @mode.setter
    def mode(self, mode):
        if any(backend._pending_commit is not None for _, backend, _, _ in self.entries):
            raise RuntimeError("join all compact commits before changing registry mode")
        if mode not in COMPACT_MODES:
            raise ValueError("unknown compact backend execution mode")
        if mode.startswith("deferred_") and any(backend.deferred_module is None for _, backend, _, _ in self.entries):
            raise ValueError("every compact layer needs the deferred extension")
        if mode in TILE_MODES and any(backend.tile_module is None for _, backend, _, _ in self.entries):
            raise ValueError("every compact layer needs the tile extension")
        for _, backend, _, _ in self.entries:
            backend.mode = mode

    def finish_pending_commits(self):
        # Validate every hook before joining any state transaction, so a missing
        # projection cannot silently be turned into a different execution path.
        if any(
            backend._pending_commit is not None and not backend._pending_commit["launched"]
            for _, backend, _, _ in self.entries
        ):
            raise RuntimeError("a registered GDN output projection did not launch its pending commit")
        return sum(backend.finish_pending_commit() for _, backend, _, _ in self.entries)

    def _require_joined(self):
        if any(backend._pending_commit is not None for _, backend, _, _ in self.entries):
            raise RuntimeError("all compact commits must be joined before snapshots or lifecycle changes")

    def prepare_step(self, step_id, request_ids, accepted, generations=None):
        self._require_joined()
        requests, accepted = tuple(request_ids), tuple(accepted)
        generations = tuple(generations) if generations is not None else None
        previous = [backend.state.pending for _, backend, _, _ in self.entries]
        try:
            for _, backend, _, _ in self.entries:
                backend.state.prepare_step(step_id, requests, accepted, generations)
        except BaseException:
            for (_, backend, _, _), pending in zip(self.entries, previous, strict=True):
                backend.state.pending = pending
            raise

    def finish_step(self, step_id):
        self._require_joined()
        if (
            isinstance(step_id, bool)
            or not isinstance(step_id, int)
            or any(
                backend.state.pending is None or backend.state.pending[0] != step_id
                for _, backend, _, _ in self.entries
            )
        ):
            raise RuntimeError("every compact layer must have the matching host transaction")
        for _, backend, _, _ in self.entries:
            backend.state.finish_step(step_id)

    def snapshot(self, destination=None):
        self._require_joined()
        states, staging = [], []
        for _, backend, _, _ in self.entries:
            states.append(backend.state.snapshot(destination=destination))
            staging.append(
                tuple(
                    (
                        identity,
                        _snapshot_tensor(backend.state.torch, lane["new_logs"], destination),
                        _snapshot_tensor(backend.state.torch, lane["new_valid"], destination),
                    )
                    for identity, lane in backend._staging_buffers().items()
                )
            )
        return CompactRegistrySnapshot(tuple(states), tuple(staging), tuple(self.backends))

    def restore(self, snapshot):
        self._require_joined()
        if (
            not isinstance(snapshot, CompactRegistrySnapshot)
            or snapshot.registry_identity != tuple(self.backends)
            or len(snapshot.states) != len(self.entries)
            or len(snapshot.staging) != len(self.entries)
        ):
            raise ValueError("snapshot belongs to a different compact registry")
        for (_, backend, _, _), state, staging in zip(self.entries, snapshot.states, snapshot.staging, strict=True):
            backend.state._validate_snapshot(state)
            buffers = backend._staging_buffers()
            for identity, logs, valid in staging:
                if identity not in buffers:
                    raise ValueError("snapshot staging lane is no longer registered")
                lane = buffers[identity]
                for target, saved in ((lane["new_logs"], logs), (lane["new_valid"], valid)):
                    if (
                        target.shape != saved.shape
                        or target.dtype != saved.dtype
                        or (target.device != saved.device and saved.device.type != "cpu")
                    ):
                        raise ValueError("snapshot staging tensor contract changed")
        for (_, backend, _, _), state, staging in zip(self.entries, snapshot.states, snapshot.staging, strict=True):
            backend.state.restore(state)
            saved_lanes = {identity: (logs, valid) for identity, logs, valid in staging}
            for identity, lane in backend._staging_buffers().items():
                if identity in saved_lanes:
                    lane["new_logs"].copy_(saved_lanes[identity][0])
                    lane["new_valid"].copy_(saved_lanes[identity][1])
                else:
                    # A coherent-prefix snapshot can precede graph capture and
                    # allocation of scratch. New inactive staging starts empty.
                    lane["new_logs"].zero_()
                    lane["new_valid"].zero_()

    def describe(self):
        return dict(
            registry="experimental_exact_compact_handle_registry",
            layers=len(self.entries),
            observed_python_calls=self.calls,
            graph_replay_calls_are_not_counted=True,
            production_kv_manager_support=False,
            layer_backends=[dict(name=name, **backend.describe()) for _, backend, _, name in self.entries],
        )


@contextmanager
def installed_log_registry(registry):
    """Install exact-handle dispatch and every GDN projection readiness hook."""
    import vllm_ascend.ops.gdn as gdn

    if not isinstance(registry, CompactSSMRegistry) or not registry.entries or registry._installed:
        raise ValueError("install a nonempty compact registry exactly once")
    registry._require_joined()
    original = gdn.recurrent_gated_delta_rule
    hooks = []
    counts = [backend.calls for _, backend, _, _ in registry.entries]
    registry._installed = True
    try:
        for _, backend, projection, _ in registry.entries:

            def launch_commit(module, inputs, backend=backend):
                backend.launch_pending_commit()

            hooks.append(projection.register_forward_pre_hook(launch_commit))
        gdn.recurrent_gated_delta_rule = registry
        yield registry
        registry._require_joined()
        if any(backend.calls == count for (_, backend, _, _), count in zip(registry.entries, counts, strict=True)):
            raise RuntimeError("a registered compact GDN layer was never called")
    finally:
        try:
            # Each layer has independent base/log ownership; drain every known
            # transaction even when an earlier layer's cleanup reports failure.
            error = None
            for _, backend, _, _ in registry.entries:
                try:
                    backend._cleanup_pending_commit()
                except BaseException as exception:
                    error = error or exception
            if error is not None:
                raise error
        finally:
            for hook in reversed(hooks):
                hook.remove()
            gdn.recurrent_gated_delta_rule = original
            registry._installed = False


@contextmanager
def installed_log_backend(state, logpath, productpath, mode="padding", deferred_path=None, tile_path=None):
    import vllm_ascend.ops.gdn as gdn

    original = gdn.recurrent_gated_delta_rule
    backend = Q4LogBackend(
        state,
        load_extension("gdn_q4_log", logpath),
        load_extension("gdn_q4_products", productpath),
        mode,
        deferred_module=load_extension("gdn_q4_log_deferred", deferred_path) if deferred_path is not None else None,
        tile_module=load_extension("gdn_q4_tile_pipeline", tile_path) if tile_path is not None else None,
    )
    backend.original = original
    backend.sources = {
        name: dict(path=str(path), sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())
        for name, path in (
            ("log", logpath),
            ("products", productpath),
            ("deferred", deferred_path),
            ("tile", tile_path),
        )
        if path is not None
    }
    gdn.recurrent_gated_delta_rule = backend
    try:
        yield backend
        if backend._pending_commit is not None:
            raise RuntimeError("compact backend context ended without joining pending state publication")
        if backend.calls == 0:
            raise RuntimeError("compact recurrence backend was never called")
    finally:
        try:
            backend._cleanup_pending_commit()
        finally:
            gdn.recurrent_gated_delta_rule = original
