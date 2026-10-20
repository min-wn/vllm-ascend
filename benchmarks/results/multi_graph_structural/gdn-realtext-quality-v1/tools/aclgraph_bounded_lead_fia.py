# SPDX-License-Identifier: Apache-2.0
"""Real full-attention prefix for the bounded-lead two-layer experiment.

Synthetic segment inputs and projected synthetic KV prefix, not model prefill.
All current-token attention math uses the installed official attention module.
"""

from contextlib import contextmanager


def validate_fia_tasks(tasks, operation, lengths):
    if len(tasks) != 1 or tasks[0].operation != operation:
        raise ValueError("every replay must update exactly one actual FIA task")
    if list(tasks[0].kwargs["actual_seq_lengths_kv"]) != lengths:
        raise ValueError("FIA update did not resolve the current true host lengths")


class FIAPrefix:
    def __init__(self, workload, runner, model, layer):
        import torch
        import torch_npu
        from multi_graph_tp_attention import request_values

        from vllm_ascend.attention.attention_v1 import AscendAttentionBackendImpl, AscendAttentionState, AscendMetadata

        self.torch, self.workload = torch, workload
        self.runner, self.model, self.layer = runner, model, layer
        self.attention = attention = layer.self_attn
        backend, impl = attention.attn, attention.attn.impl
        if (
            type(impl) is not AscendAttentionBackendImpl
            or impl.sinks is not None
            or impl.sliding_window is not None
            or impl.use_bnsd_kv_cache
            or impl.pcp_enabled
            or impl.kv_sharing_target_layer_name is not None
            or impl.enable_c8_quant
        ):
            raise ValueError("requires ordinary dense BF16 TND FIA")
        shape = (workload.kv_blocks, workload.block_size, attention.num_kv_heads, attention.head_dim)
        self.caches = tuple(
            torch_npu.npu_format_cast(torch.zeros(shape, dtype=torch.bfloat16, device="npu"), 2) for _ in range(2)
        )
        if any(torch_npu.get_npu_format(t) != 2 or not t.is_contiguous() for t in self.caches):
            raise ValueError("old-driver path requires contiguous ND KV")
        backend.kv_cache = self.caches
        impl.key_cache, impl.value_cache = self.caches
        values = request_values(workload, 0, workload.capacity, workload.batch, 0)
        self.positions = torch.tensor(values["positions"], dtype=torch.int64, device="npu")
        lengths = torch.tensor(values["lengths"], dtype=torch.int32)
        boundaries = torch.tensor(values["boundaries"], dtype=torch.int32)
        self.metadata = AscendMetadata(
            attn_mask=torch.ones((2048, 2048), dtype=torch.bool, device="npu").triu_(1),
            attn_state=AscendAttentionState.ChunkedPrefill,
            num_actual_tokens=workload.capacity * 4,
            num_decode_tokens=workload.capacity * 4,
            num_decodes=workload.capacity,
            seq_lens=lengths,
            seq_lens_cpu=lengths,
            seq_lens_list=values["lengths"],
            seq_lens_gpu=lengths.npu(),
            query_start_loc=boundaries,
            query_start_loc_gpu=boundaries.npu(),
            actual_seq_lengths_q=values["boundaries"][1:],
            max_query_len=4,
            block_tables=torch.tensor(values["pages"], dtype=torch.int32, device="npu"),
            slot_mapping=torch.tensor(values["slots"], dtype=torch.int64, device="npu"),
            model_runner_type="generate",
        )
        self.mapping = {backend.layer_name: self.metadata}
        self.addresses = self.identity()
        self.cpu_inputs = {}
        self.prepare(workload.batch, 0)
        self.stage()
        self.initialize_prefix()
        torch.npu.synchronize()

    def identity(self):
        return tuple(
            t.data_ptr()
            for t in (
                self.positions,
                self.metadata.seq_lens_gpu,
                self.metadata.block_tables,
                self.metadata.slot_mapping,
                *self.caches,
            )
        )

    def prepare(self, live, variant):
        from multi_graph_tp_attention import request_values

        values = request_values(self.workload, 0, self.workload.capacity, live, variant)
        self.metadata.seq_lens_list[:] = values["lengths"]
        self.metadata.seq_lens.copy_(self.torch.tensor(values["lengths"], dtype=self.torch.int32))
        self.cpu_inputs = {
            key: self.torch.tensor(values[key], dtype=dtype).pin_memory()
            for key, dtype in (
                ("lengths", self.torch.int32),
                ("pages", self.torch.int32),
                ("slots", self.torch.int64),
                ("positions", self.torch.int64),
            )
        }
        if self.identity() != self.addresses:
            raise ValueError("FIA metadata/cache addresses changed")

    def stage(self):
        for destination, key in (
            (self.metadata.seq_lens_gpu, "lengths"),
            (self.metadata.block_tables, "pages"),
            (self.metadata.slot_mapping, "slots"),
            (self.positions, "positions"),
        ):
            destination.copy_(self.cpu_inputs[key], non_blocking=True)

    @contextmanager
    def context(self, capturing=False):
        from vllm.config import CUDAGraphMode
        from vllm.forward_context import BatchDescriptor, get_forward_context

        from vllm_ascend.ascend_forward_context import set_ascend_forward_context

        rows = self.workload.capacity * 4
        with set_ascend_forward_context(
            self.mapping,
            self.runner.vllm_config,
            num_tokens=rows,
            num_actual_tokens=rows,
            aclgraph_runtime_mode=CUDAGraphMode.NONE,
            batch_descriptor=BatchDescriptor(num_tokens=rows, num_reqs=self.workload.capacity, uniform=True),
            model_instance=self.model,
            skip_compiled=True,
        ):
            get_forward_context().capturing = capturing
            yield

    def forward(self, hidden, residual, capturing=False):
        with self.context(capturing):
            normalized, residual = self.layer.input_layernorm(hidden, residual)
            return self.attention(positions=self.positions, hidden_states=normalized), residual

    def initialize_prefix(self):
        # Same projection and QK/RoPE transform as the installed official path;
        # only K/V are retained to populate a coherent synthetic layer prefix.
        from vllm_ascend.patch.worker.patch_qwen3_5 import _uses_multimodal_rope

        torch, attention = self.torch, self.attention
        torch.manual_seed(901)
        hidden = (
            torch.randn(
                (self.workload.kv_length, self.runner.vllm_config.model_config.hf_text_config.hidden_size),
                dtype=torch.bfloat16,
                device="npu",
            )
            * 0.1
        )
        positions = torch.arange(self.workload.kv_length, device="npu")
        with self.context():
            normalized = self.layer.input_layernorm(hidden)
            qkv, _ = attention.qkv_proj(normalized)
            if not _uses_multimodal_rope(attention):
                raise ValueError("this fixture qualifies only the current Qwen3.5 MRoPE path")
            cos_sin = attention.rotary_emb.cos_sin_cache[positions.unsqueeze(0).expand(3, -1)].to(qkv)
            _, key, value, _ = torch.ops.vllm.triton_split_qkv_rmsnorm_mrope(
                qkv=qkv,
                q_weight=1.0 + attention.q_norm.weight,
                k_weight=1.0 + attention.k_norm.weight,
                cos_sin=cos_sin,
                num_q_heads=attention.num_heads,
                num_kv_heads=attention.num_kv_heads,
                head_size=attention.head_dim,
                eps=attention.config.rms_norm_eps,
                mrope_section=attention.rotary_emb.mrope_section,
                is_interleaved=attention.rotary_emb.mrope_interleaved,
                rope_dim=attention.rotary_emb.rotary_dim,
                has_gate=True,
            )
        page_rows = self.workload.pages_per_request * self.workload.block_size
        for cache, prefix in zip(self.caches, (key, value), strict=True):
            prefix = prefix.reshape(self.workload.kv_length, attention.num_kv_heads, attention.head_dim)
            view = cache.view(self.workload.capacity, page_rows, attention.num_kv_heads, attention.head_dim)
            view[:, : self.workload.kv_length].copy_(prefix.unsqueeze(0).expand(self.workload.capacity, -1, -1, -1))

    def check_unwritten(self, before, after):
        torch = self.torch
        device = before[0].device
        untouched = torch.ones(self.workload.kv_blocks * self.workload.block_size, dtype=torch.bool, device=device)
        slots = self.cpu_inputs["slots"]
        untouched[slots[slots >= 0].to(device)] = False
        for old, new in zip(before, after, strict=True):
            torch.testing.assert_close(new.flatten(0, 1)[untouched], old.flatten(0, 1)[untouched], rtol=0, atol=0)

    def resolve(self, graph):
        import torch_npu

        from vllm_ascend.compilation.updatable_graph import ContextSource

        tasks = graph.resolve_tasks(ContextSource(self.mapping))
        validate_fia_tasks(tasks, torch_npu.npu_fused_infer_attention_score.out, self.metadata.seq_lens_list)
        return tasks
