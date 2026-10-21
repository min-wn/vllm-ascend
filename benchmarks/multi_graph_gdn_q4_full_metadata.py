# SPDX-License-Identifier: Apache-2.0
"""Explicit caches and stable metadata for the complete hybrid Q4 experiment.

FIA pages, recurrent owners and convolution candidate slots are independent
namespaces. No allocation here is a four-candidate dense SSM cache.
"""

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from multi_graph_dynamic_trace import trace_digest

FIXED_TRACE_SHA256 = "0519bc80410bd15d0893e0283189425ec285d53003198c163183b79e6249e524"
QUERY = 4
BLOCK_SIZE = 128


@dataclass(frozen=True)
class FrozenTrace:
    steps: tuple
    initial_prefix_by_request: tuple
    seed: int
    digest: str


def load_fixed_trace(path):
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    trace = document["trace"]
    digest = trace_digest(trace)
    if digest != FIXED_TRACE_SHA256 or document.get("trace_sha256") != digest:
        raise ValueError("the complete-model experiment requires the unchanged predeclared trace")
    prefixes = trace["initial_prefix_by_request"]
    if trace["query"] != QUERY or set(prefixes) != {str(index) for index in range(128)}:
        raise ValueError("expected the fixed Q4 trajectory with 128 stable request owners")
    # Freeze nested input vectors as well as the containing record. Runtime
    # metadata refresh may never alter a later control's declared workload.
    steps = tuple(
        MappingProxyType({key: tuple(value) if isinstance(value, list) else value for key, value in step.items()})
        for step in trace["steps"]
    )
    return FrozenTrace(steps, tuple(prefixes[str(index)] for index in range(128)), trace["seed"], digest)


@dataclass(frozen=True)
class OwnerPages:
    pages: tuple
    maximum_lengths: tuple
    page_count: int
    width: int

    def slot(self, request, position):
        if not 0 <= request < len(self.pages) or not 0 <= position < self.maximum_lengths[request]:
            raise ValueError("request or logical token offset is outside its fixed FIA reservation")
        return self.pages[request][position // BLOCK_SIZE] * BLOCK_SIZE + position % BLOCK_SIZE


def reserve_owner_pages(trace):
    maximum = list(trace.initial_prefix_by_request)
    for step in trace.steps:
        for request, length in zip(step["request_ids"], step["seq_lengths"], strict=True):
            maximum[request] = max(maximum[request], length)
    pages, next_page = [], 1
    for length in maximum:
        count = (length + BLOCK_SIZE - 1) // BLOCK_SIZE
        pages.append(tuple(range(next_page, next_page + count)))
        next_page += count
    return OwnerPages(tuple(pages), tuple(maximum), next_page, max(map(len, pages)))


def decode_values(step, capacity_tokens, pages):
    """Derive true host lengths without confusing request identity and row."""
    if (
        isinstance(capacity_tokens, bool)
        or not isinstance(capacity_tokens, int)
        or capacity_tokens % QUERY
        or not 1 <= step["batch"] <= capacity_tokens // QUERY
    ):
        raise ValueError("Q4 batch must fit a positive capture capacity divisible by four")
    live, capacity = step["batch"], capacity_tokens // QUERY
    requests, lengths, accepted = (tuple(step[key]) for key in ("request_ids", "seq_lengths", "accepted_from_previous"))
    if len(requests) != live or len(lengths) != live or len(accepted) != live or len(set(requests)) != live:
        raise ValueError("one unique request ID, true sequence length and accepted prefix per live row required")
    if any(isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= QUERY for value in accepted):
        raise ValueError("accepted prefixes must be integers in 1..4")
    positions, slots, blocks, gdn_blocks = [], [], [], []
    for request, length in zip(requests, lengths, strict=True):
        if not isinstance(length, int) or isinstance(length, bool) or length < QUERY:
            raise ValueError("true sequence lengths must include four current queries")
        current = range(length - QUERY, length)
        slots.extend(pages.slot(request, position) for position in current)
        positions.extend(current)
        owner_pages = pages.pages[request]
        blocks.append([*owner_pages, *([0] * (pages.width - len(owner_pages)))])
        gdn_blocks.append([1 + QUERY * request + token for token in range(QUERY)])
    padding = capacity - live
    return dict(
        lengths=[*lengths, *([0] * padding)],
        accepted=[*accepted, *([1] * padding)],
        positions=[*positions, *([0] * (padding * QUERY))],
        slots=[*slots, *([-1] * (padding * QUERY))],
        pages=[*blocks, *([[0] * pages.width for _ in range(padding)])],
        gdn_blocks=[*gdn_blocks, *([[0] * QUERY for _ in range(padding)])],
        gdn_boundaries=[min(index, live) * QUERY for index in range(capacity + 1)],
        fia_boundaries=[index * QUERY for index in range(capacity + 1)],
        drafts=[*([QUERY - 1] * live), *([-1] * padding)],
        active=[*([True] * (live * QUERY)), *([False] * (padding * QUERY))],
    )


@dataclass
class GDNCacheEntry:
    index: int
    prefix: str
    gdn: object
    base: object
    conv: object
    state: object = None


@dataclass
class FIACacheEntry:
    index: int
    attention: object
    backend: object
    impl: object
    key: object
    value: object


class FullCacheBundle:
    def __init__(self, torch, gdn_entries, fia_entries, pages, capture_sizes, device, trace=None):
        self.torch, self.gdn_entries, self.fia_entries = torch, gdn_entries, fia_entries
        self.pages, self.capture_sizes, self.device = pages, tuple(capture_sizes), device
        self.trace = trace
        self.decode_ready = False

    def transition_to_decode(self):
        from multi_graph_gdn_q4_log_backend import Q4LogState

        if self.decode_ready:
            raise ValueError("compact decode transition must happen exactly once")
        count = len(self.pages.pages)
        candidates = tuple(tuple(1 + QUERY * request + token for token in range(QUERY)) for request in range(count))
        for entry in self.gdn_entries:
            prefix_conv = entry.conv
            conv = self.torch.zeros(
                (1 + count * QUERY, *prefix_conv.shape[1:]), dtype=prefix_conv.dtype, device=self.device
            )
            conv[1:].view(count, QUERY, *prefix_conv.shape[1:]).copy_(prefix_conv[1:].unsqueeze(1))
            entry.state = Q4LogState.from_base(self.torch, entry.base, candidates)
            entry.conv = conv
            entry.gdn.kv_cache = (conv, entry.state.handle)
            # No prefix-conv or dense four-candidate state remains owned here.
            del prefix_conv
        self.decode_ready = True


def allocate_full_caches(torch, model, config, trace, *, device="npu", capture_sizes, format_cast, expected_owners=128):
    pages = reserve_owner_pages(trace)
    owners = len(pages.pages)
    gdn_entries, fia_entries = [], []
    layers = model.language_model.model.layers
    if len(layers) != 64 or not 1 <= expected_owners <= 128 or owners != expected_owners:
        raise ValueError("complete allocation requires 64 real decoders and the declared owner count in 1..128")
    for index, layer in enumerate(layers):
        if layer.layer_type == "linear_attention":
            gdn = layer.linear_attn
            conv_shape, ssm_shape = gdn.get_state_shape()
            conv_dtype, ssm_dtype = gdn.get_state_dtype()
            if tuple(ssm_shape) != (24, 128, 128) or ssm_dtype != torch.float32 or conv_dtype != torch.bfloat16:
                raise ValueError("compact full replay requires the actual TP2 FP32 SSM/BF16 conv layout")
            base = torch.zeros((owners + 1, *ssm_shape), dtype=ssm_dtype, device=device)
            conv = torch.zeros((owners + 1, *conv_shape), dtype=conv_dtype, device=device)
            gdn.kv_cache = (conv, base)
            gdn_entries.append(GDNCacheEntry(index, gdn.prefix, gdn, base, conv))
        elif layer.layer_type == "full_attention":
            attention = layer.self_attn
            backend, impl = attention.attn, attention.attn.impl
            if (
                impl.use_bnsd_kv_cache
                or impl.pcp_enabled
                or impl.enable_c8_quant
                or impl.sinks is not None
                or impl.sliding_window is not None
                or impl.kv_sharing_target_layer_name is not None
                or attention.num_kv_heads != 2
                or attention.head_dim != 256
            ):
                raise ValueError("complete replay currently requires independent TP2 BF16 ND FIA caches")
            shape = (pages.page_count, BLOCK_SIZE, attention.num_kv_heads, attention.head_dim)
            key, value = (format_cast(torch.zeros(shape, dtype=torch.bfloat16, device=device)) for _ in range(2))
            if not key.is_contiguous() or not value.is_contiguous():
                raise ValueError("the old-driver FIA path requires contiguous ND caches")
            backend.kv_cache = (key, value)
            impl.key_cache, impl.value_cache = key, value
            fia_entries.append(FIACacheEntry(index, attention, backend, impl, key, value))
        else:
            raise ValueError("unexpected decoder type in the complete hybrid model")
    if len(gdn_entries) != 48 or len(fia_entries) != 16:
        raise ValueError("complete replay must own all 48 GDN and 16 FIA layer caches")
    return FullCacheBundle(torch, gdn_entries, fia_entries, pages, capture_sizes, torch.device(device), trace)


def _metadata_types():
    from vllm.config import CUDAGraphMode

    from vllm_ascend.attention.attention_v1 import AscendAttentionState, AscendMetadata
    from vllm_ascend.attention.utils import AscendCommonAttentionMetadata
    from vllm_ascend.ops.gdn_attn_builder import AscendGDNAttentionMetadataBuilder

    return (
        CUDAGraphMode,
        AscendAttentionState,
        AscendMetadata,
        AscendCommonAttentionMetadata,
        AscendGDNAttentionMetadataBuilder,
    )


class FullMetadataFrame:
    def __init__(self, factory, capacity_tokens):
        self.factory, self.capacity_tokens = factory, capacity_tokens
        torch, device = factory.torch, factory.bundle.device
        self.positions = torch.zeros(capacity_tokens, dtype=torch.int64, device=device)
        self.slot_mapping = torch.full((capacity_tokens,), -1, dtype=torch.int64, device=device)
        self.active = torch.zeros((capacity_tokens, 1), dtype=torch.bool, device=device)
        self.mapping = {}

    def refresh(self, step):
        self.factory._refresh_decode(self, step)
        return self


class FullMetadataFactory:
    def __init__(self, torch, bundle, config):
        self.torch, self.bundle, self.config = torch, bundle, config
        self.types = _metadata_types()
        self.frames, self.builders = {}, {}
        self.mask = torch.ones((2048, 2048), dtype=torch.bool, device=bundle.device).triu_(1)

    def _builder(self, capacity, *, prefix=False):
        # Eager prefixes need no per-length persistent graph buffers. Reusing
        # one builder avoids retaining 128 copies of its device scratch.
        key = (0 if prefix else capacity, prefix)
        if key not in self.builders:
            mode, _, _, _, builder_type = self.types
            config = copy.copy(self.config)
            config.cache_config = copy.copy(config.cache_config)
            config.cache_config.mamba_cache_mode = "none"
            config.compilation_config = copy.copy(config.compilation_config)
            config.compilation_config.cudagraph_mode = mode.NONE if prefix else mode.FULL
            config.compilation_config.max_cudagraph_capture_size = capacity
            entry = self.bundle.gdn_entries[0]
            self.builders[key] = builder_type(
                entry.gdn.get_kv_cache_spec(config), [entry.prefix], config, self.bundle.device
            )
        return self.builders[key]

    def _common(self, boundaries, lengths, blocks, positions, slots, *, prefilling, capacity_tokens):
        torch = self.torch
        _, _, _, common_type, _ = self.types
        boundaries = torch.tensor(boundaries, dtype=torch.int32)
        lengths = torch.tensor(lengths, dtype=torch.int32)
        computed = lengths - torch.diff(boundaries)
        return common_type(
            query_start_loc=boundaries.to(self.bundle.device),
            query_start_loc_cpu=boundaries,
            seq_lens=lengths.to(self.bundle.device),
            _seq_lens_cpu=lengths,
            seq_lens_cpu=lengths,
            seq_lens_cpu_upper_bound=lengths,
            _num_computed_tokens_cpu=computed,
            num_computed_tokens_cpu=computed,
            num_reqs=len(lengths),
            num_actual_tokens=int(boundaries[-1]),
            num_input_tokens=capacity_tokens,
            max_query_len=int(torch.diff(boundaries).max()),
            max_seq_len=int(lengths.max()),
            block_table_tensor=torch.tensor(blocks, dtype=torch.int32, device=self.bundle.device),
            slot_mapping=slots,
            positions=positions,
            causal=True,
            is_prefilling=torch.full((len(lengths),), prefilling, dtype=torch.bool),
        )

    def _fia(self, boundaries, lengths, blocks, slots, *, prefix):
        torch = self.torch
        _, state_type, metadata_type, _, _ = self.types
        lengths_cpu = torch.tensor(lengths, dtype=torch.int32)
        boundaries_cpu = torch.tensor(boundaries, dtype=torch.int32)
        return metadata_type(
            attn_mask=self.mask,
            attn_state=state_type.PrefillNoCache if prefix else state_type.ChunkedPrefill,
            num_actual_tokens=len(slots),
            num_decode_tokens=0 if prefix else len(slots),
            num_prefills=1 if prefix else 0,
            num_decodes=0 if prefix else len(lengths),
            seq_lens=lengths_cpu,
            seq_lens_cpu=lengths_cpu,
            seq_lens_list=list(lengths),
            seq_lens_gpu=lengths_cpu.to(self.bundle.device),
            query_start_loc=boundaries_cpu,
            query_start_loc_gpu=boundaries_cpu.to(self.bundle.device),
            actual_seq_lengths_q=list(boundaries[1:]),
            max_query_len=boundaries[-1] if prefix else QUERY,
            block_tables=torch.tensor(blocks, dtype=torch.int32, device=self.bundle.device),
            slot_mapping=slots,
            model_runner_type="generate",
        )

    def prefix(self, owner, length):
        if self.bundle.decode_ready:
            raise ValueError("later prefill is unsupported after compact decode activation")
        if not 0 <= owner < len(self.bundle.pages.pages) or not 1 < length <= self.bundle.pages.maximum_lengths[owner]:
            raise ValueError("prefix owner and length must fit their fixed reservation")
        torch = self.torch
        frame = FullMetadataFrame(self, length)
        # A prefix is not a padded decode. Avoid advertising an active mask to
        # the endpoint observer unless it describes every real prefix token.
        frame.active.fill_(True)
        frame.positions.copy_(torch.arange(length, dtype=torch.int64, device=self.bundle.device))
        frame.slot_mapping.copy_(
            torch.tensor(
                [self.bundle.pages.slot(owner, pos) for pos in range(length)],
                dtype=torch.int64,
                device=self.bundle.device,
            )
        )
        common = self._common(
            [0, length],
            [length],
            [[owner + 1]],
            frame.positions,
            frame.slot_mapping,
            prefilling=True,
            capacity_tokens=length,
        )
        gdn = self._builder(length, prefix=True).build(0, common)
        if gdn.num_prefills != 1 or gdn.num_decodes != 0 or gdn.spec_decode_metadata is not None:
            raise ValueError("coherent prefix must use the official non-speculative prefill path")
        pages = self.bundle.pages.pages[owner]
        fia = self._fia(
            [0, length],
            [length],
            [[*pages, *([0] * (self.bundle.pages.width - len(pages)))]],
            frame.slot_mapping,
            prefix=True,
        )
        frame.mapping.update({entry.prefix: gdn for entry in self.bundle.gdn_entries})
        frame.mapping.update({entry.backend.layer_name: fia for entry in self.bundle.fia_entries})
        return frame

    def decode(self, step, capacity_tokens):
        if capacity_tokens not in self.bundle.capture_sizes:
            raise ValueError("decode capacity must come from the resolved installed default catalog")
        if capacity_tokens not in self.frames:
            self.frames[capacity_tokens] = FullMetadataFrame(self, capacity_tokens)
        return self.frames[capacity_tokens].refresh(step)

    def _refresh_decode(self, frame, step):
        if not self.bundle.decode_ready:
            raise ValueError("transition coherent prefix caches before constructing compact decode metadata")
        torch = self.torch
        values = decode_values(step, frame.capacity_tokens, self.bundle.pages)
        for tensor, key, dtype in (
            (frame.positions, "positions", torch.int64),
            (frame.slot_mapping, "slots", torch.int64),
        ):
            tensor.copy_(torch.tensor(values[key], dtype=dtype))
        frame.active.copy_(torch.tensor(values["active"], dtype=torch.bool).view(-1, 1))
        common = self._common(
            values["gdn_boundaries"],
            values["lengths"],
            values["gdn_blocks"],
            frame.positions,
            frame.slot_mapping,
            prefilling=False,
            capacity_tokens=frame.capacity_tokens,
        )
        gdn = self._builder(frame.capacity_tokens).build(
            0,
            common,
            num_accepted_tokens=torch.tensor(values["accepted"], dtype=torch.int32, device=self.bundle.device),
            num_decode_draft_tokens_cpu=torch.tensor(values["drafts"], dtype=torch.int32),
            num_actual_reqs=step["batch"],
        )
        # The Python slice is fixed at capture, while zero-length segments and
        # null cache IDs mask inactive recurrent requests on subsequent replay.
        gdn.num_actual_tokens = frame.capacity_tokens
        if gdn.num_prefills or gdn.num_decodes or gdn.spec_decode_metadata is None:
            raise ValueError("compact decode requires a pure Q4 speculative metadata branch")
        frame.mapping.update({entry.prefix: gdn for entry in self.bundle.gdn_entries})
        if not hasattr(frame, "fia"):
            frame.fia = self._fia(
                values["fia_boundaries"], values["lengths"], values["pages"], frame.slot_mapping, prefix=False
            )
        else:
            fia = frame.fia
            fia.seq_lens_list[:] = values["lengths"]
            fia.seq_lens.copy_(torch.tensor(values["lengths"], dtype=torch.int32))
            fia.seq_lens_gpu.copy_(fia.seq_lens)
            fia.block_tables.copy_(torch.tensor(values["pages"], dtype=torch.int32))
        frame.mapping.update({entry.backend.layer_name: frame.fia for entry in self.bundle.fia_entries})
