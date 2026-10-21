# SPDX-License-Identifier: Apache-2.0
"""Real text, coherent prefill states and target-layer inputs from layers 0..3."""

import hashlib
from types import SimpleNamespace

from multi_graph_gdn_ready_range_protocol import PREFIX_TOKENS, TEXTS


class ActivationSource:
    def __init__(self, torch, model, config, *, owners=100):
        import torch_npu
        from multi_graph_gdn_q4_full_metadata import (
            BLOCK_SIZE,
            FIACacheEntry,
            FullMetadataFactory,
            GDNCacheEntry,
            OwnerPages,
        )
        from transformers import AutoTokenizer

        self.torch, self.model, self.config = torch, model, config
        self.layers = model.language_model.model.layers
        if [layer.layer_type for layer in self.layers[:5]] != [
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
            "linear_attention",
        ]:
            raise ValueError("the existing Qwen3.5 target must be layer 4")
        tokenizer = AutoTokenizer.from_pretrained(config.model_config.model, local_files_only=True)
        self.tokens = []
        for owner in range(owners):
            text = f"Request {owner}. " + TEXTS[owner % len(TEXTS)] * 40
            encoded = tokenizer.encode(text, add_special_tokens=False)
            if len(encoded) < 256:
                raise ValueError("frozen text must provide at least 256 tokens per owner")
            self.tokens.append(encoded[:256])
        self.token_sha256 = hashlib.sha256(str(self.tokens).encode()).hexdigest()
        pages = OwnerPages(
            tuple((1 + owner * 2, 2 + owner * 2) for owner in range(owners)), (256,) * owners, 1 + 2 * owners, 2
        )
        gdn_entries, fia_entries = [], []
        for index, layer in enumerate(self.layers[:5]):
            if layer.layer_type == "linear_attention":
                gdn = layer.linear_attn
                conv_shape, state_shape = gdn.get_state_shape()
                conv_dtype, state_dtype = gdn.get_state_dtype()
                conv = torch.zeros((owners + 1, *conv_shape), dtype=conv_dtype, device="npu")
                base = torch.zeros((owners + 1, *state_shape), dtype=state_dtype, device="npu")
                gdn.kv_cache = (conv, base)
                gdn_entries.append(GDNCacheEntry(index, gdn.prefix, gdn, base, conv))
            else:
                attention = layer.self_attn
                shape = (pages.page_count, BLOCK_SIZE, attention.num_kv_heads, attention.head_dim)
                key, value = (
                    torch_npu.npu_format_cast(torch.zeros(shape, dtype=torch.bfloat16, device="npu"), 2)
                    for _ in range(2)
                )
                attention.attn.kv_cache = (key, value)
                attention.attn.impl.key_cache, attention.attn.impl.value_cache = key, value
                fia_entries.append(FIACacheEntry(index, attention, attention.attn, attention.attn.impl, key, value))
        bundle = SimpleNamespace(
            gdn_entries=gdn_entries,
            fia_entries=fia_entries,
            pages=pages,
            device=torch.device("npu"),
            decode_ready=False,
        )
        self.bundle = bundle
        self.factory = FullMetadataFactory(torch, bundle, config)
        self.target = gdn_entries[-1]

    def batch(self, owners, lengths, *, target=False):
        from vllm.config import CUDAGraphMode
        from vllm.forward_context import get_forward_context

        from vllm_ascend.ascend_forward_context import set_ascend_forward_context

        torch, factory = self.torch, self.factory
        boundaries, flat_tokens, positions, slots, blocks = [0], [], [], [], []
        for owner, length in zip(owners, lengths, strict=True):
            flat_tokens.extend(self.tokens[owner][:length])
            positions.extend(range(length))
            slots.extend(self.bundle.pages.slot(owner, pos) for pos in range(length))
            blocks.append(list(self.bundle.pages.pages[owner]))
            boundaries.append(boundaries[-1] + length)
        positions = torch.tensor(positions, dtype=torch.int64, device="npu")
        slots = torch.tensor(slots, dtype=torch.int64, device="npu")
        for entry in self.bundle.gdn_entries:
            if target or entry.index < 4:
                for owner in owners:
                    entry.base[owner + 1].zero_()
                    entry.conv[owner + 1].zero_()
        common = factory._common(
            boundaries,
            lengths,
            [[owner + 1] for owner in owners],
            positions,
            slots,
            prefilling=True,
            capacity_tokens=len(flat_tokens),
        )
        gdn_meta = factory._builder(len(flat_tokens), prefix=True).build(0, common)
        if gdn_meta.num_prefills != len(owners) or gdn_meta.spec_decode_metadata is not None:
            raise ValueError("activation source must run the official packed prefill path")
        fia_meta = factory._fia(boundaries, lengths, blocks, slots, prefix=True)
        fia_meta.num_prefills = len(owners)
        fia_meta.max_query_len = max(lengths)
        mapping = {entry.prefix: gdn_meta for entry in self.bundle.gdn_entries}
        mapping.update({entry.backend.layer_name: fia_meta for entry in self.bundle.fia_entries})
        ids = torch.tensor(flat_tokens, dtype=torch.int64, device="npu")
        with set_ascend_forward_context(
            mapping,
            self.config,
            num_tokens=len(flat_tokens),
            num_actual_tokens=len(flat_tokens),
            aclgraph_runtime_mode=CUDAGraphMode.NONE,
            model_instance=self.model,
            skip_compiled=True,
        ):
            get_forward_context().slot_mapping = {entry.backend.layer_name: slots for entry in self.bundle.fia_entries}
            get_forward_context().capturing = False
            hidden = self.model.language_model.model.embed_tokens(ids)
            residual = None
            for layer in self.layers[:4]:
                hidden, residual = layer(hidden_states=hidden, residual=residual, positions=positions)
            if target:
                self.layers[4](hidden_states=hidden, residual=residual, positions=positions)
        result = []
        for stop in boundaries[1:]:
            result.append((hidden[stop - 4 : stop].clone(), residual[stop - 4 : stop].clone()))
        return result

    def initial_target(self):
        for start in range(0, 100, 8):
            owners = list(range(start, min(100, start + 8)))
            self.batch(owners, [PREFIX_TOKENS] * len(owners), target=True)
        self.torch.npu.synchronize()
        return tuple(tensor.clone() for tensor in (self.target.conv, self.target.base))

    def records(self, profile, hidden_size):
        torch = self.torch
        result, manifest = [], []
        for entry in profile["trace"]:
            hidden, residual = (torch.zeros(400, hidden_size, dtype=torch.bfloat16, device="npu") for _ in range(2))
            for start in range(0, entry["live"], 8):
                owners = entry["request_ids"][start : start + 8]
                values = self.batch(owners, entry["seq_lengths"][start : start + 8])
                for offset, pair in enumerate(values, start):
                    hidden[offset * 4 : (offset + 1) * 4].copy_(pair[0])
                    residual[offset * 4 : (offset + 1) * 4].copy_(pair[1])
            row = dict(step=entry["step"], fields=[])
            for tensor in (hidden, residual):
                if not torch.isfinite(tensor).all().item():
                    raise ValueError("nonfinite real activation")
                blob = tensor.cpu().view(torch.uint8).numpy().tobytes()
                active = tensor[: entry["live"] * 4].float()
                row["fields"].append(
                    dict(
                        sha256=hashlib.sha256(blob).hexdigest(),
                        rms=active.square().mean().sqrt().item(),
                        max_abs=active.abs().max().item(),
                    )
                )
            result.append((hidden, residual))
            manifest.append(row)
            print(f"activation profile={profile['name']} step={entry['step']} live={entry['live']}", flush=True)
        return result, manifest
