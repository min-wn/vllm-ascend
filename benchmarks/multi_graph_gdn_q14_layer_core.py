# SPDX-License-Identifier: Apache-2.0
"""Benchmark-only Q1/Q4 adapter around one real GDN layer's official core.

The caller owns fixed-address metadata, the coherent conv/compact caches and
the full-layer endpoint. No kernel, model class, precision or ABI is changed.
Only the target instance packs its already projected inputs. Module-level op
dispatch is replaced only while that instance's original core is executing;
concurrent Python forwards in the same process are therefore unsupported.
"""

from contextlib import contextmanager
from dataclasses import dataclass
from types import MethodType

from multi_graph_gdn_q4_log_backend import Q4LogBackend

MODES = ("forced_q4", "q14_ready", "q14_delayed")


@dataclass(frozen=True)
class PackedLayerDescriptor:
    """Caller-updated tensors for a fixed capacity of T=4*B projected rows.

    pack_indices selects dense quartet rows; ignored packed rows can select any
    in-range source row and are explicitly zeroed using active. inverse_indices
    maps each dense row to a packed row, or to the zero sentinel T. sequence is
    [0, q0, q1, ...] (lengths, not cumulative offsets), ids contains owner+1 for
    each packed token and zero in its tail, and accepted belongs to the owner's
    *previous* query. In particular previous acceptance 4 is valid for current
    Q1. The caller validates values and copies all tensors in place before use.

    Official conv metadata is separate: cumulative query_start_loc, the same
    previous acceptance values, and cache_indices[:,0] == 1+4*owner. It must
    remain on the speculative branch even for a batch containing only Q1.
    """

    pack_indices: object
    inverse_indices: object
    active: object
    sequence: object
    ids: object
    accepted: object

    @property
    def capacity_tokens(self):
        return self.pack_indices.numel()

    def validate(self, torch, device):
        tokens = self.capacity_tokens
        if not 4 <= tokens <= 512 or tokens % 4:
            raise ValueError("layer descriptor requires 4*B rows with B in 1..128")
        expected = (
            (self.pack_indices, torch.int64, tokens),
            (self.inverse_indices, torch.int64, tokens),
            (self.active, torch.bool, tokens),
            (self.sequence, torch.int32, tokens // 4 + 1),
            (self.ids, torch.int32, tokens),
            (self.accepted, torch.int32, tokens // 4),
        )
        for tensor, dtype, length in expected:
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.dtype != dtype
                or tensor.shape != (length,)
                or tensor.device != device
                or not tensor.is_contiguous()
            ):
                raise ValueError("fixed contiguous descriptor tensors with the declared dtype/shape/device required")
        if (
            bool((self.pack_indices < 0).any())
            or bool((self.pack_indices >= tokens).any())
            or bool((self.inverse_indices < 0).any())
            or bool((self.inverse_indices > tokens).any())
            or self.sequence[0].item() != 0
            or bool(((self.sequence[1:] != 0) & (self.sequence[1:] != 1) & (self.sequence[1:] != 4)).any())
            or int(self.sequence[1:].sum()) > tokens
            or bool((self.accepted < 1).any())
            or bool((self.accepted > 4).any())
            or bool((self.ids < 0).any())
            or bool((self.ids > tokens).any())
        ):
            raise ValueError("descriptor indices, Q1/Q4 lengths, owner IDs or acceptance are invalid")
        active_count = int(self.active.sum())
        if active_count != int(self.sequence[1:].sum()):
            raise ValueError("active rows must equal the sum of declared Q1/Q4 lengths")


class _FrozenModuleMetadata:
    """Keep frozen arithmetic but bypass the Q4 backend's physical-ID lookup."""

    def __init__(self, descriptor, solve_module, commit_module):
        self.descriptor, self.solve_module, self.commit_module = descriptor, solve_module, commit_module

    def solve(self, *args):
        arguments = list(args)
        arguments[8:11] = [self.descriptor.sequence, self.descriptor.ids, self.descriptor.accepted]
        return self.solve_module.solve(*arguments)

    def commit(self, base, sequence, ids, accepted, logs, valid, new_logs, new_valid):
        return self.commit_module.commit(
            base,
            self.descriptor.sequence,
            self.descriptor.ids,
            self.descriptor.accepted,
            logs,
            valid,
            new_logs,
            new_valid,
        )

    def __getattr__(self, name):
        return getattr(self.commit_module, name)


class LayerCoreAdapter:
    def __init__(
        self,
        torch,
        gdn,
        metadata,
        state,
        descriptor,
        old_module,
        new_module,
        deferred_module,
        *,
        mode="q14_ready",
        endpoint=None,
        gdn_ops=None,
    ):
        if mode not in MODES:
            raise ValueError("unknown full-layer Q1/Q4 mode")
        descriptor.validate(torch, state.base.device)
        self.torch, self.gdn, self.metadata = torch, gdn, metadata
        self.descriptor, self.state, self.mode = descriptor, state, mode
        self.endpoint = gdn if endpoint is None else endpoint
        self.tokens = descriptor.capacity_tokens
        self._validate_metadata()
        if gdn.kv_cache[1] is not state.handle:
            raise ValueError("target GDN must own this exact compact SSM handle")
        if gdn_ops is None:
            import vllm_ascend.ops.gdn as gdn_ops
        self.ops = gdn_ops
        solve_module = old_module if mode == "forced_q4" else new_module
        commit_module = deferred_module if mode == "forced_q4" else new_module
        if not callable(getattr(solve_module, "solve", None)) or not callable(getattr(commit_module, "commit", None)):
            raise ValueError("explicit frozen solve and commit modules required")
        shim = _FrozenModuleMetadata(descriptor, solve_module, commit_module)
        self.backend = Q4LogBackend(
            state,
            None,
            None,
            mode="deferred_tile_delayed" if mode == "q14_delayed" else "deferred_tile_ready",
            deferred_module=shim,
            tile_module=shim,
        )
        self._packed = {}
        self._output = self._inactive = None
        self._running = False

    def _validate_metadata(self):
        metadata = self.metadata
        spec = getattr(metadata, "spec_decode_metadata", None)
        if (
            getattr(metadata, "num_prefills", None) != 0
            or getattr(metadata, "num_decodes", None) != 0
            or getattr(metadata, "spec_sequence_masks", None) is None
            or spec is None
            or metadata.num_actual_tokens != self.tokens
        ):
            raise ValueError("the target core requires fixed-capacity pure speculative metadata")
        causal = spec.spec_causal_conv1d
        if (
            causal.query_start_loc.numel() != self.tokens // 4 + 1
            or causal.cache_indices.numel() != self.tokens
            or causal.num_accepted_tokens.numel() != self.tokens // 4
            or spec.actual_seq_lengths.numel() != self.tokens // 4 + 1
        ):
            raise ValueError("official conv and recurrence metadata must retain fixed request capacity")

    def _pack(self, name, source):
        torch = self.torch
        if source.shape[0] != self.tokens:
            raise ValueError("dense projected inputs must retain four rows per captured request")
        if name not in self._packed:
            self._packed[name] = torch.empty(source.shape, dtype=source.dtype, device=source.device)
        target = self._packed[name]
        if target.shape != source.shape or target.dtype != source.dtype or target.device != source.device:
            raise ValueError("projected input shape, precision or device changed after reservation")
        torch.index_select(source, 0, self.descriptor.pack_indices, out=target)
        target.masked_fill_(self._inactive.view(self.tokens, *([1] * (source.ndim - 1))), 0)
        return target

    def _core(self, original, mixed_qkv, b, a, output):
        if self._running:
            raise RuntimeError("the target core cannot execute recursively or concurrently")
        self._validate_metadata()
        self._running = True
        original_conv = self.ops.causal_conv1d_update
        original_recurrence = self.ops.recurrent_gated_delta_rule

        def finite_conv(*args, **kwargs):
            target = kwargs.get("out")
            if target is None:
                target = self.torch.zeros_like(args[0])
                kwargs["out"] = target
            else:
                target.zero_()
            return original_conv(*args, **kwargs)

        try:
            self.ops.causal_conv1d_update = finite_conv
            self.ops.recurrent_gated_delta_rule = self.backend
            if self.mode == "forced_q4":
                return original(mixed_qkv, b, a, output)
            if self._inactive is None:
                self._inactive = self.torch.empty_like(self.descriptor.active)
                self._output = self.torch.zeros(
                    (self.tokens + 1, *output.shape[1:]), dtype=output.dtype, device=output.device
                )
            if self._output.shape[1:] != output.shape[1:] or self._output.dtype != output.dtype:
                raise ValueError("core output shape or precision changed after reservation")
            self.torch.logical_not(self.descriptor.active, out=self._inactive)
            packed = [self._pack(name, value) for name, value in (("mixed_qkv", mixed_qkv), ("b", b), ("a", a))]
            self._output.zero_()
            original(*packed, self._output[: self.tokens])
            # Sentinel T is never written by the official core. Invalid dense
            # rows gather this defined zero instead of unspecified conv lanes.
            self.torch.index_select(self._output, 0, self.descriptor.inverse_indices, out=output)
        finally:
            self.ops.causal_conv1d_update = original_conv
            self.ops.recurrent_gated_delta_rule = original_recurrence
            self._running = False

    def buffer_pointers(self):
        values = dict(self._packed)
        if self._output is not None:
            values.update(output=self._output, inactive=self._inactive)
        return {name: tensor.data_ptr() for name, tensor in values.items()}


@contextmanager
def installed(*args, **kwargs):
    """Install one instance adapter; yield its backend for snapshot diagnostics.

    Pass endpoint=the complete decoder so the commit join follows output GEMM,
    TP reductions and MLP. Warm up each fixed tensor layout before capture. All
    descriptor/official metadata uploads and lifecycle validation remain owned
    by the caller; neither device values nor replay calls are host-validated.
    """
    adapter = kwargs.pop("adapter", None)
    if adapter is None:
        adapter = LayerCoreAdapter(*args, **kwargs)
    elif args or kwargs:
        raise ValueError("a persistent adapter cannot be combined with new constructor arguments")
    gdn = adapter.gdn
    original = gdn._forward_core
    had_instance_core = "_forward_core" in gdn.__dict__
    previous_instance_core = gdn.__dict__.get("_forward_core")

    def core(owner, mixed_qkv, b, a, core_attn_out):
        return adapter._core(original, mixed_qkv, b, a, core_attn_out)

    def launch(module, inputs):
        adapter.backend.launch_pending_commit()

    def join(module, inputs, output):
        adapter.backend.finish_pending_commit()

    handles = []
    try:
        gdn._forward_core = MethodType(core, gdn)
        handles.append(gdn.out_proj.register_forward_pre_hook(launch))
        handles.append(adapter.endpoint.register_forward_hook(join))
        yield adapter
    finally:
        try:
            adapter.backend._cleanup_pending_commit()
        finally:
            for handle in reversed(handles):
                handle.remove()
            if had_instance_core:
                gdn._forward_core = previous_instance_core
            elif "_forward_core" in gdn.__dict__:
                del gdn._forward_core
