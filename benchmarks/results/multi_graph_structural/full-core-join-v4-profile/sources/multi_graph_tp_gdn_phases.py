"""Separate official GDN math into scheduling phases without changing state handling.

This benchmark adapter only separates launch boundaries. The registered official
attention core owns all convolution/SSM state access, masking, and updates.
"""

from dataclasses import dataclass

import torch


@dataclass
class GDNPhase:
    owner: object
    name: str
    mixed_qkv: torch.Tensor
    z: torch.Tensor
    b: torch.Tensor
    a: torch.Tensor
    core_output: torch.Tensor
    residual: torch.Tensor
    normalized_output: torch.Tensor | None = None
    completed: bool = False


class GDNPhases:
    """Launch projection, stateful vector core, and local output separately."""

    def __init__(self, layer, context_for_name, *, core_op=None, split_layout=None):
        self.layer = layer
        self.gdn = layer.linear_attn
        self.context_for_name = context_for_name
        self.core_op = core_op if core_op is not None else torch.ops.vllm.qwen_gdn_attention_core
        self.split_layout = split_layout
        projection = self.gdn.out_proj
        if getattr(projection, "custom_op", None) is not None or getattr(projection, "bias", None) is not None:
            raise ValueError("phased GDN requires an ordinary bias-free output projection")
        if not hasattr(projection, "reduce_results"):
            raise ValueError("phased GDN requires an explicit output reduction switch")
        if not hasattr(self.gdn, "in_proj_qkv") and self.gdn.gqa_interleaved_layout and split_layout is None:
            # Lazy import keeps CPU schedule tests independent of the NPU stack.
            from vllm_ascend.ops.triton.fla.fused_qkvzba_split_reshape import fused_qkvzba_split_reshape_cat

            self.split_layout = fused_qkvzba_split_reshape_cat

    def project(self, value, prior, name="full"):
        """Run input Norm and the exact official projection/layout branches."""
        gdn = self.gdn
        with self.context_for_name(name):
            hidden_states, residual = self.layer.input_layernorm(value, prior)
            num_tokens = hidden_states.size(0)
            if hasattr(gdn, "in_proj_qkv"):
                mixed_qkv, _ = gdn.in_proj_qkv(hidden_states)
                ba, _ = gdn.in_proj_ba(hidden_states)
                z, _ = gdn.in_proj_z(hidden_states)
                z = z.reshape(z.size(0), -1, gdn.head_v_dim)
                b, a = gdn._split_ba_for_tp(ba)
                b, a = b.contiguous(), a.contiguous()
            elif not gdn.gqa_interleaved_layout:
                mixed_qkvz, _ = gdn.in_proj_qkvz(hidden_states)
                num_tokens = mixed_qkvz.size(0)
                qkv_size = (gdn.key_dim * 2 + gdn.value_dim) // gdn.tp_size
                z_size = gdn.value_dim // gdn.tp_size
                mixed_qkv, z = mixed_qkvz.split([qkv_size, z_size], dim=-1)
                z = z.reshape(z.size(0), -1, gdn.head_v_dim)
                ba, _ = gdn.in_proj_ba(hidden_states)
                b, a = gdn._split_ba_for_tp(ba)
                b, a = b.contiguous(), a.contiguous()
            else:
                projected_qkvz, _ = gdn.in_proj_qkvz(hidden_states)
                projected_ba, _ = gdn.in_proj_ba(hidden_states)
                num_tokens = projected_qkvz.size(0)
                mixed_qkv, z, b, a = self.split_layout(
                    projected_qkvz,
                    projected_ba,
                    (gdn.num_k_heads + gdn.tp_size - 1) // gdn.tp_size,
                    (gdn.num_v_heads + gdn.tp_size - 1) // gdn.tp_size,
                    gdn.head_k_dim,
                    gdn.head_v_dim,
                )
            # Official forward uses zeros: inactive rows must not be uninitialized.
            core_output = torch.zeros(
                (num_tokens, gdn.num_v_heads // gdn.tp_size, gdn.head_v_dim),
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
        return GDNPhase(self, name, mixed_qkv, z, b, a, core_output, residual)

    def _check(self, phase, name):
        if phase.owner is not self or phase.name != name:
            raise ValueError("GDN phase belongs to another adapter or metadata lane")
        if phase.completed:
            raise ValueError("GDN phase output has already been consumed")

    def partition(self, phase, names=("left", "right"), boundaries=(0, 200, 400)):
        """Split a full projection into metadata lanes using tensor views only.

        The caller supplies contexts whose state indices describe each global
        request interval. Projection work and residual creation stay shared;
        only the official state core and output projection run per child.
        """
        self._check(phase, phase.name)
        if phase.normalized_output is not None:
            raise ValueError("GDN partition must precede the stateful core")
        names, boundaries = tuple(names), tuple(boundaries)
        if not names or any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(names):
            raise ValueError("GDN partition requires unique nonempty metadata lane names")
        count = phase.core_output.shape[0]
        if (
            len(boundaries) != len(names) + 1
            or any(not isinstance(value, int) or isinstance(value, bool) for value in boundaries)
            or boundaries[0] != 0
            or boundaries[-1] != count
            or any(start >= stop for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True))
        ):
            raise ValueError("GDN partition requires nonempty disjoint intervals covering every token")
        fields = (phase.mixed_qkv, phase.z, phase.b, phase.a, phase.core_output, phase.residual)
        if any(value.shape[0] != count for value in fields):
            raise ValueError("GDN projected phase tensors must share the token extent")
        children = tuple(
            GDNPhase(self, name, *(value[start:stop] for value in fields))
            for name, start, stop in zip(names, boundaries[:-1], boundaries[1:], strict=True)
        )
        phase.completed = True
        return children

    def core(self, phase, name=None):
        """Run the official stateful core and gated Norm, without a GEMM."""
        name = phase.name if name is None else name
        self._check(phase, name)
        if phase.normalized_output is not None:
            raise ValueError("GDN core may only run once per projected phase")
        with self.context_for_name(name):
            self.core_op(phase.mixed_qkv, phase.b, phase.a, phase.core_output, self.gdn.prefix, False)
            z_shape = phase.z.shape
            normalized = self.gdn.norm(
                phase.core_output.reshape(-1, phase.core_output.shape[-1]),
                phase.z.reshape(-1, phase.z.shape[-1]),
            )
            normalized = normalized.reshape(z_shape)
            phase.normalized_output = normalized.reshape(*normalized.shape[:-2], -1)
        return phase.normalized_output

    def output(self, phase, name=None):
        """Run the official output projection locally; the caller schedules AR."""
        name = phase.name if name is None else name
        self._check(phase, name)
        if phase.normalized_output is None:
            raise ValueError("GDN output requires the stateful core first")
        projection = self.gdn.out_proj
        with self.context_for_name(name):
            previous_reduce_results = projection.reduce_results
            projection.reduce_results = False
            try:
                projected, _ = projection(phase.normalized_output)
            finally:
                projection.reduce_results = previous_reduce_results
        phase.completed = True
        return projected, phase.residual


def make_gdn_phases(layer, context_for_name, *, core_op=None, split_layout=None):
    """Construct an adapter; optional CPU dependencies preserve test isolation."""
    return GDNPhases(layer, context_for_name, core_op=core_op, split_layout=split_layout)
