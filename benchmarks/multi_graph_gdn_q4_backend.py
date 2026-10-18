"""Explicit benchmark-only Q4 recurrence replacement for real GDN layers.

Convolution, gating, normalization, cache ownership and attention metadata stay
with the official GDN implementation. Only the recurrence call is replaced.
"""

import hashlib
import importlib.util
from contextlib import contextmanager
from pathlib import Path


def load_extension(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Q4Backend:
    def __init__(self, block, products, torch_module, original):
        self.block, self.products, self.torch, self.original = block, products, torch_module, original
        self.buffers = {}
        self.calls = 0
        self.original_calls = 0
        self.layouts = set()
        self.sources = None

    def __call__(self, query, key, value, initial_state, **kwargs):
        accepted = kwargs.get("num_accepted_tokens")
        if accepted is None:
            self.original_calls += 1
            return self.original(query, key, value, initial_state, **kwargs)
        torch = self.torch
        batch = accepted.numel()
        sequence, ids = kwargs.get("actual_seq_lengths"), kwargs.get("ssm_state_indices")
        gate, beta = kwargs.get("g"), kwargs.get("beta")
        if set(kwargs) - {"num_accepted_tokens", "actual_seq_lengths", "ssm_state_indices", "g", "beta", "scale"}:
            raise ValueError("Q4 benchmark backend refuses unknown recurrence options")
        if not 1 <= batch <= 128 or tuple(query.shape) != (4 * batch, 8, 128) or key.shape != query.shape:
            raise ValueError("Q4 benchmark backend requires fixed-capacity four-token 8/24-head input")
        if value.shape != (4 * batch, 24, 128) or initial_state.shape[1:] != (24, 128, 128):
            raise ValueError("Q4 benchmark backend requires the TP2-local state layout")
        if (
            initial_state.dtype not in (torch.bfloat16, torch.float32)
            or initial_state.stride()[1:] != (16384, 128, 1)
            or initial_state.stride(0) < 24 * 128 * 128
            or initial_state.shape[0] < 1 + 4 * batch
        ):
            raise ValueError(
                f"Q4 state requires BF16/FP32 contiguous inner dimensions and enough nonoverlapping slots; got "
                f"dtype={initial_state.dtype}, shape={tuple(initial_state.shape)}, "
                f"stride={initial_state.stride()}"
            )
        if sequence is None or ids is None or sequence.numel() != batch + 1 or ids.numel() != batch * 4:
            raise ValueError("Q4 benchmark backend requires complete fixed-address device metadata")
        if gate is None or beta is None or kwargs.get("scale") != 128**-0.5:
            raise ValueError("Q4 benchmark backend requires scalar gating and the official query scale")
        if gate.shape != beta.shape or beta.shape != (4 * batch, 24):
            raise ValueError("Q4 gating dimensions disagree")
        if any(t.dtype != torch.bfloat16 for t in (query, key, value, beta)) or gate.dtype != torch.float32:
            raise ValueError("Q4 recurrence requires BF16 Q/K/V/beta and FP32 gating")
        if any(t.dtype != torch.int32 or t.ndim != 1 for t in (sequence, ids, accepted)):
            raise ValueError("Q4 metadata must be one-dimensional int32 tensors")
        if any(not t.is_contiguous() for t in (query, key, value, beta, gate, sequence, ids, accepted)):
            raise ValueError("Q4 inputs and metadata must be contiguous except for the state page stride")
        if any(t.device != query.device for t in (key, value, beta, gate, initial_state, sequence, ids, accepted)):
            raise ValueError("Q4 inputs, state and metadata must share a device")
        # Metadata pointers distinguish concurrently executing graph branches.
        # Q/K projection temporaries may change across eager calls/capture.
        identity = (
            initial_state.data_ptr(),
            tuple(initial_state.shape),
            tuple(initial_state.stride()),
            initial_state.dtype,
            query.device,
            sequence.data_ptr(),
            ids.data_ptr(),
            accepted.data_ptr(),
            batch,
        )
        if identity not in self.buffers:
            self.buffers[identity] = (
                torch.empty((batch, 24, 128, 8), dtype=torch.float32, device=query.device),
                torch.zeros_like(value),
            )
        products, output = self.buffers[identity]
        output.zero_()
        # Cube uses one input dtype. Cast only the small Q/K inputs here so
        # FP32-state conversion costs are part of eager/captured execution.
        product_query, product_key = (
            (query.float(), key.float()) if initial_state.dtype == torch.float32 else (query, key)
        )
        self.products.products(initial_state, product_query, product_key, sequence, ids, accepted, products)
        self.block.emit(products, query, key, value, beta, gate, initial_state, sequence, ids, accepted, output)
        self.calls += 1
        self.layouts.add((str(initial_state.dtype), tuple(initial_state.shape), tuple(initial_state.stride())))
        return output

    def describe(self):
        return dict(
            backend="experimental_q4_block_direct_products",
            observed_python_calls=self.calls,
            observed_non_spec_original_calls=self.original_calls,
            graph_replay_calls_are_not_counted=True,
            persistent_buffers=len(self.buffers),
            persistent_bytes=sum(t.numel() * t.element_size() for pair in self.buffers.values() for t in pair),
            state_layouts=[
                dict(dtype=dtype, shape=shape, stride=stride) for dtype, shape, stride in sorted(self.layouts)
            ],
            fp32_product_inputs_cast_inside_execution=True,
            production_default_changed=False,
            libraries=self.sources,
        )


@contextmanager
def installed_backend(block_path, products_path):
    import torch

    import vllm_ascend.ops.gdn as gdn

    original = gdn.recurrent_gated_delta_rule
    backend = Q4Backend(
        load_extension("gdn_q4_block", block_path),
        load_extension("gdn_q4_products", products_path),
        torch,
        original,
    )
    backend.sources = provenance(block_path, products_path)
    gdn.recurrent_gated_delta_rule = backend
    try:
        yield backend
        if backend.calls == 0:
            raise RuntimeError("experimental Q4 backend was never called; refusing an official-only result")
    finally:
        gdn.recurrent_gated_delta_rule = original


def provenance(block_path, products_path):
    return {
        name: dict(path=str(path), sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())
        for name, path in (("block", block_path), ("products", products_path))
    }
