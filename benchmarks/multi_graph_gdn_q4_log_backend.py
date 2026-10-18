# SPDX-License-Identifier: Apache-2.0
"""Explicit benchmark-only compact SSM ownership for one real TP2 GDN layer.

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


@dataclass(frozen=True)
class Q4LogSnapshot:
    tensors: tuple
    state_identity: tuple
    generations: tuple
    initialized: tuple
    last_step: int
    pending: tuple | None


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

    def snapshot(self):
        return Q4LogSnapshot(
            tensors=tuple(tensor.clone() for tensor in self.mutable_tensors),
            state_identity=self._snapshot_identity,
            generations=tuple(self.generations),
            initialized=tuple(self.initialized),
            last_step=self.last_step,
            pending=self.pending,
        )

    def restore(self, snapshot):
        if not isinstance(snapshot, Q4LogSnapshot) or snapshot.state_identity != self._snapshot_identity:
            raise ValueError("snapshot belongs to a different compact cache")
        if len(snapshot.tensors) != len(self.mutable_tensors) or any(
            actual.shape != saved.shape or actual.dtype != saved.dtype or actual.device != saved.device
            for actual, saved in zip(self.mutable_tensors, snapshot.tensors, strict=True)
        ):
            raise ValueError("snapshot tensor contract changed")
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
    def __init__(self, state, log_module, product_module, mode="padding", deferred_module=None):
        self.state, self.log_module, self.product_module = state, log_module, product_module
        self.deferred_module = deferred_module
        self.buffers = {}
        self.calls = 0
        self.sources = None
        self.original = None
        self._pending_commit = None
        self._failed = False
        self.mode = mode

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, mode):
        if self._pending_commit is not None:
            raise RuntimeError("cannot change compact mode before the pending state commit is joined")
        if mode not in ("padding", "serial", "pipeline", "deferred_serial", "deferred_overlap", "deferred_pipeline"):
            raise ValueError("unknown compact backend execution mode")
        if mode.startswith("deferred_") and self.deferred_module is None:
            raise ValueError("deferred compact execution requires an explicit deferred extension")
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
        if initial_state is not self.state.dense_state:
            raise ValueError("recurrence must identify the explicitly registered dense initialization source")
        if (
            initial_state.data_ptr(),
            tuple(initial_state.shape),
            tuple(initial_state.stride()),
        ) != self.state.dense_identity:
            raise ValueError("registered dense cache storage or layout changed")
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
        identity = (sequence.data_ptr(), ids.data_ptr(), accepted.data_ptr(), batch)
        if identity not in self.buffers:
            self.buffers[identity] = dict(
                products=torch.empty((batch, 24, 128, 8), dtype=torch.float32, device=query.device),
                output=torch.zeros_like(value),
                ids=torch.empty_like(ids),
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
                buffers.update(
                    new_logs=torch.zeros_like(self.state.logs),
                    new_valid=torch.zeros_like(self.state.valid),
                    commit_stream=torch.npu.Stream(),
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

            if self.mode == "deferred_pipeline":
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
                for lane in self.buffers.values()
                for key, t in lane.items()
                if key in ("products", "output", "ids", "new_logs", "new_valid")
            ),
            dense_initialization_source_becomes_stale=True,
            non_spec_and_fallback="rejected",
            all_prefix_export="explicit diagnostic only; not included in recurrence",
            device_metadata_value_validation=False,
            lifecycle="optional host prepare/finish guards; no native generation or retry validation",
            preparation="device owner lookup, output clear, Q/K casts included in callable",
            deferred_extension_loaded=self.deferred_module is not None,
            deferred_commit_pending=self._pending_commit is not None,
            deferred_staging="scratch only; active new logs published into canonical old bank by commit",
            deferred_overlap="output-projection prehook launches commit; full-layer endpoint must join",
            deferred_pipeline="fixed request halves overlap products/solve; all solves joined before gated Norm",
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


@contextmanager
def installed_log_backend(state, logpath, productpath, mode="padding", deferred_path=None):
    import vllm_ascend.ops.gdn as gdn

    original = gdn.recurrent_gated_delta_rule
    backend = Q4LogBackend(
        state,
        load_extension("gdn_q4_log", logpath),
        load_extension("gdn_q4_products", productpath),
        mode,
        deferred_module=load_extension("gdn_q4_log_deferred", deferred_path) if deferred_path is not None else None,
    )
    backend.original = original
    backend.sources = {
        name: dict(path=str(path), sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())
        for name, path in (("log", logpath), ("products", productpath), ("deferred", deferred_path))
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
