# SPDX-License-Identifier: Apache-2.0
"""Benchmark-only adjacent-layer request pipeline inside the real model loop.

The predecessor returns a full tensor populated on the communication stream.
Only its paired consumer may read it, after waiting for each row-range event.
No auxiliary observer may consume the intermediate asynchronous tensor.
"""

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass


@dataclass
class PackedProjection:
    owner: object
    name: str
    qkvz: object
    ba: object
    residual: object
    consumed: bool = False


def project_packed(phases, value, prior, name):
    if hasattr(phases.gdn, "in_proj_qkv"):
        raise ValueError("packed join requires the QKVZ/BA projection layout")
    hidden, residual = phases.layer.input_layernorm(value, prior)
    qkvz, _ = phases.gdn.in_proj_qkvz(hidden)
    ba, _ = phases.gdn.in_proj_ba(hidden)
    return PackedProjection(phases, name, qkvz, ba, residual)


def join_packed(projected):
    """Join three packed tensors, then lay out/zero the full core only once."""
    import torch
    from multi_graph_tp_gdn_phases import GDNPhase

    if len(projected) != 2 or tuple(p.name for p in projected) != ("left", "right"):
        raise ValueError("require ordered left/right packed projections")
    phases = projected[0].owner
    for p in projected:
        if p.owner is not phases or p.consumed or not p.qkvz.shape[0] == p.ba.shape[0] == p.residual.shape[0]:
            raise ValueError("packed projections must be unconsumed with matching owner/rows")
    qkvz, ba, residual = (torch.cat([getattr(p, field) for p in projected]) for field in ("qkvz", "ba", "residual"))
    gdn = phases.gdn
    if gdn.gqa_interleaved_layout:
        mixed_qkv, z, b, a = phases.split_layout(
            qkvz,
            ba,
            (gdn.num_k_heads + gdn.tp_size - 1) // gdn.tp_size,
            (gdn.num_v_heads + gdn.tp_size - 1) // gdn.tp_size,
            gdn.head_k_dim,
            gdn.head_v_dim,
        )
    else:
        mixed_qkv, z = qkvz.split(
            [(gdn.key_dim * 2 + gdn.value_dim) // gdn.tp_size, gdn.value_dim // gdn.tp_size], dim=-1
        )
        z = z.reshape(z.shape[0], -1, gdn.head_v_dim)
        b, a = gdn._split_ba_for_tp(ba)
        b, a = b.contiguous(), a.contiguous()
    core_output = torch.zeros(
        (qkvz.shape[0], gdn.num_v_heads // gdn.tp_size, gdn.head_v_dim), dtype=qkvz.dtype, device=qkvz.device
    )
    for p in projected:
        p.consumed = True
    return GDNPhase(phases, "full", mixed_qkv, z, b, a, core_output, residual)


def legal_pair(capacity, catalog):
    sizes = {size for size in catalog if size > 0 and size % 4 == 0}
    pairs = [(left, capacity - left) for left in sizes if capacity - left in sizes]
    if capacity not in sizes or not pairs:
        raise ValueError("both request lanes and full capacity must belong to the default Q4 catalog")
    return min(pairs, key=lambda pair: (abs(pair[0] - pair[1]), -pair[0]))


def validate_boundaries(layers, boundaries):
    if not boundaries or len(set(boundaries)) != len(boundaries):
        raise ValueError("nonempty unique predecessor indices required")
    for index in boundaries:
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(layers) - 1:
            raise ValueError("boundary must have a following decoder")
        previous, following = layers[index : index + 2]
        if previous.layer_type != "full_attention" or following.layer_type != "linear_attention":
            raise ValueError("only adjacent FIA -> GDN boundaries are supported")
        if any(layer.layer_scale or layer.use_attn_reduce_scatter_for_moe for layer in (previous, following)):
            raise ValueError("layer scaling and sequence-parallel MoE are unsupported")


class CoreJoinSchedule:
    modes = ("padding", "split_barrier", "split_early")

    def __init__(self, model, boundaries, catalog, *, packed=False):
        import torch
        from multi_graph_tp_gdn_phases import GDNPhases
        from vllm.distributed.parallel_state import get_tp_group

        decoder = model.language_model.model
        validate_boundaries(decoder.layers, boundaries)
        if decoder.use_sequence_parallel or getattr(decoder, "aux_hidden_state_layers", ()):
            raise ValueError("asynchronous adjacent consumer excludes sequence parallel and auxiliary outputs")
        self.torch, self.group = torch, get_tp_group()
        self.catalog, self.boundaries = tuple(catalog), tuple(boundaries)
        self.mode, self.packed = "padding", packed
        self.comm = torch.npu.Stream()
        self.pairs = []
        for index in boundaries:
            previous, following = decoder.layers[index : index + 2]
            projection = previous.mlp.down_proj
            if (
                projection.bias is not None
                or getattr(projection, "custom_op", None) is not None
                or not projection.reduce_results
            ):
                raise ValueError("require an ordinary bias-free reduced Down projection")
            self.pairs.append(
                dict(
                    index=index,
                    down=projection,
                    layer=following,
                    phases=GDNPhases(following, lambda name: nullcontext()),
                    ready=[torch.npu.Event() for _ in range(2)],
                    done=[torch.npu.Event() for _ in range(2)],
                    pending=None,
                )
            )

    def assert_joined(self):
        if any(pair["pending"] is not None for pair in self.pairs):
            raise RuntimeError("a paired GDN consumer was skipped")

    def down(self, pair, original, value):
        if self.mode == "padding":
            return original(value)
        if pair["pending"] is not None:
            raise RuntimeError("unconsumed previous row-range output")
        torch = self.torch
        left, right = legal_pair(value.shape[0], self.catalog)
        output = torch.empty((left + right, pair["down"].weight.shape[0]), dtype=value.dtype, device=value.device)
        root = torch.npu.current_stream()
        for lane, (start, stop) in enumerate(((0, left), (left, left + right))):
            partial = torch.nn.functional.linear(value[start:stop], pair["down"].weight)
            pair["ready"][lane].record(root)
            with torch.npu.stream(self.comm):
                self.comm.wait_event(pair["ready"][lane])
                reduced = self.group.all_reduce(partial)
                output[start:stop].copy_(reduced)
                pair["done"][lane].record(self.comm)
            # Eager execution may otherwise recycle a temporary before the side stream reads it.
            partial.record_stream(self.comm)
        pair["pending"] = (output, left)
        return output, None

    def consume(self, pair, original, *, hidden_states, residual, positions, **kwargs):
        if self.mode == "padding":
            return original(hidden_states=hidden_states, residual=residual, positions=positions, **kwargs)
        from aclgraph_bounded_lead_probe import join_projected_phases

        torch, phases = self.torch, pair["phases"]
        if pair["pending"] is None or pair["pending"][0] is not hidden_states:
            raise RuntimeError("paired consumer did not receive its predecessor's output tensor")
        _, left = pair["pending"]
        root = torch.npu.current_stream()
        if self.mode == "split_barrier":
            root.wait_event(pair["done"][1])
        projected = []
        for lane, (name, start, stop) in enumerate((("left", 0, left), ("right", left, hidden_states.shape[0]))):
            root.wait_event(pair["done"][lane])
            projected.append(
                project_packed(phases, hidden_states[start:stop], residual[start:stop], name)
                if self.packed
                else phases.project(hidden_states[start:stop], residual[start:stop], name)
            )
        phase = join_packed(projected) if self.packed else join_projected_phases(projected)
        phases.core(phase)
        # The installed registry's out_proj hook publishes state exactly once.
        output, _ = phases.gdn.out_proj(phase.normalized_output)
        output, residual = pair["layer"].post_attention_layernorm(output, phase.residual)
        output = pair["layer"].mlp(output)
        phase.completed = True
        pair["pending"] = None
        return output, residual

    @contextmanager
    def installed(self):
        originals = []
        try:
            for pair in self.pairs:
                for module, method in ((pair["down"], self.down), (pair["layer"], self.consume)):
                    local = module.__dict__.get("forward")
                    original = module.forward
                    originals.append((module, local))

                    def wrapper(*args, pair=pair, method=method, original=original, **kwargs):
                        return method(pair, original, *args, **kwargs)

                    module.forward = wrapper
            yield self
            self.assert_joined()
        finally:
            for module, local in reversed(originals):
                if local is None:
                    del module.forward
                else:
                    module.forward = local
