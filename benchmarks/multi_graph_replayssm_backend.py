# SPDX-License-Identifier: Apache-2.0
"""Isolated Q4 real-layer adapter for the frozen ReplaySSM A2 control."""

from contextlib import contextmanager


class RingState:
    def __init__(self, torch, dense, candidates):
        self.torch, self.dense = torch, dense
        selected = torch.tensor([0, *(slots[0] for slots in candidates)], device=dense.device)
        self.base = dense.index_select(0, selected).contiguous()
        owners = len(candidates) + 1
        self.ring = torch.zeros((owners, 24, 32, 256), dtype=torch.float32, device=dense.device)
        self.gates = torch.zeros((owners, 24, 32, 8), dtype=torch.float32, device=dense.device)
        self.cursors = torch.zeros((owners, 4, 8), dtype=torch.int32, device=dense.device)
        lookup = [0] * dense.shape[0]
        for owner, slots in enumerate(candidates, 1):
            for slot in slots:
                lookup[slot] = owner
        self.lookup = torch.tensor(lookup, dtype=torch.int32, device=dense.device)
        self.mutable = (self.base, self.ring, self.gates, self.cursors)

    def snapshot(self):
        return tuple(tensor.clone() for tensor in self.mutable)

    def restore(self, values):
        for target, value in zip(self.mutable, values, strict=True):
            target.copy_(value)


class RingBackend:
    def __init__(self, state, module):
        self.state, self.module, self.streaming = state, module, True
        self.buffers = {}
        self.inputs = None

    def __call__(self, query, key, value, initial_state, **kwargs):
        torch, state = self.state.torch, self.state
        if initial_state is not state.dense or kwargs.get("scale") != 128**-0.5:
            raise ValueError("exact initialization handle and official Q4 scale required")
        sequence, ids, accepted = (
            kwargs[name] for name in ("actual_seq_lengths", "ssm_state_indices", "num_accepted_tokens")
        )
        batch = accepted.numel()
        identity = (sequence.data_ptr(), ids.data_ptr(), accepted.data_ptr(), batch)
        if identity not in self.buffers:
            self.buffers[identity] = dict(
                products=torch.empty((batch, 24, 128, 8), device=query.device),
                output=torch.zeros_like(value),
                ids=torch.empty_like(ids),
            )
        buffers = self.buffers[identity]
        torch.index_select(state.lookup, 0, ids.long(), out=buffers["ids"])
        buffers["output"].zero_()
        q_float, k_float = query.float(), key.float()
        operands = (state.base, state.ring, state.gates, state.cursors, sequence, buffers["ids"], accepted)
        self.module.state(*operands, 0)
        # A flush changes the checkpoint used by THIS step's products. It is
        # serialized before Cube, with no out-projection deferred-commit hook.
        self.module.state(*operands, 1)
        self.module.solve(
            state.base,
            q_float,
            k_float,
            query,
            key,
            value,
            kwargs["beta"],
            kwargs["g"],
            sequence,
            buffers["ids"],
            accepted,
            buffers["products"],
            buffers["output"],
            state.ring,
            state.gates,
            state.cursors,
            self.streaming,
        )
        self.inputs = (query, key, value, kwargs["beta"], kwargs["g"], sequence, buffers["ids"], accepted)
        return buffers["output"]

    def drain(self, sequence, ids, accepted):
        state = self.state
        operands = (state.base, state.ring, state.gates, state.cursors, sequence, ids, accepted)
        self.module.state(*operands, 0)
        self.module.state(*operands, 2)
        self.module.state(*operands, 3)


@contextmanager
def installed_ring(backend):
    import vllm_ascend.ops.gdn as gdn

    original = gdn.recurrent_gated_delta_rule
    gdn.recurrent_gated_delta_rule = backend
    try:
        yield backend
    finally:
        gdn.recurrent_gated_delta_rule = original
