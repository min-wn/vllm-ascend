# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for complete-model ownership and immutable dynamic inputs."""

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def metadata(monkeypatch):
    root = Path(__file__).resolve().parents[4]
    monkeypatch.syspath_prepend(str(root / "benchmarks"))
    return importlib.import_module("multi_graph_gdn_q4_full_metadata")


@pytest.fixture
def trace(metadata):
    path = (
        Path(__file__).resolve().parents[4]
        / "benchmarks/results/multi_graph_structural/structural-dynamic-trace-v1.json"
    )
    return metadata.load_fixed_trace(path)


def test_fixed_trace_checks_original_digest_and_freezes_nested_vectors(metadata, trace):
    assert trace.digest == metadata.FIXED_TRACE_SHA256
    assert len(trace.steps) == len(trace.initial_prefix_by_request) == 128
    assert trace.seed == 20261018
    with pytest.raises(TypeError):
        trace.steps[0]["batch"] = 99
    with pytest.raises(TypeError):
        trace.steps[0]["request_ids"][0] = 99


def test_modified_manifest_is_rejected_even_if_its_declared_digest_is_recomputed(metadata, tmp_path):
    from multi_graph_dynamic_trace import trace_digest

    source = (
        Path(__file__).resolve().parents[4]
        / "benchmarks/results/multi_graph_structural/structural-dynamic-trace-v1.json"
    )
    document = json.loads(source.read_text())
    document["trace"]["steps"][0]["seq_lengths"][0] += 1
    document["trace_sha256"] = trace_digest(document["trace"])
    target = tmp_path / "modified.json"
    target.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="unchanged"):
        metadata.load_fixed_trace(target)


def test_fia_reservation_is_stable_disjoint_and_covers_all_true_lengths(metadata, trace):
    pages = metadata.reserve_owner_pages(trace)
    assert pages.page_count == 1293
    assert pages.width == 15
    flat = [page for owner in pages.pages for page in owner]
    assert flat == list(range(1, pages.page_count))
    for step in trace.steps:
        for request, length in zip(step["request_ids"], step["seq_lengths"], strict=True):
            for position in range(length - 4, length):
                slot = pages.slot(request, position)
                assert slot // 128 in pages.pages[request]
                assert slot % 128 == position % 128
    with pytest.raises(ValueError):
        pages.slot(0, pages.maximum_lengths[0])


def test_dynamic_metadata_preserves_shuffled_identity_and_null_padding(metadata, trace):
    pages = metadata.reserve_owner_pages(trace)
    step = trace.steps[0]
    values = metadata.decode_values(step, 400, pages)
    request, length = step["request_ids"][0], step["seq_lengths"][0]
    assert request != 0
    assert values["gdn_blocks"][0] == list(range(1 + request * 4, 5 + request * 4))
    assert values["positions"][:4] == list(range(length - 4, length))
    assert values["slots"][:4] == [pages.slot(request, position) for position in range(length - 4, length)]
    assert values["lengths"][:98] == list(step["seq_lengths"])
    assert values["accepted"][:98] == list(step["accepted_from_previous"])
    assert values["gdn_blocks"][-2:] == [[0] * 4] * 2
    assert values["lengths"][-2:] == [0, 0]
    assert values["accepted"][-2:] == [1, 1]
    assert values["drafts"][-2:] == [-1, -1]
    assert values["slots"][-8:] == [-1] * 8
    assert values["gdn_boundaries"][-3:] == [392, 392, 392]
    assert values["fia_boundaries"][-3:] == [392, 396, 400]
    assert values["active"][-8:] == [False] * 8


def test_trace_first_use_and_pause_resume_use_declared_pending_acceptance(metadata, trace):
    seen, pending = set(), [1] * 128
    for step in trace.steps:
        values = metadata.decode_values(step, 512, metadata.reserve_owner_pages(trace))
        for row, request in enumerate(step["request_ids"]):
            assert values["accepted"][row] == pending[request]
            if request not in seen:
                assert values["accepted"][row] == 1
            seen.add(request)
            pending[request] = step["accepted_after_step"][row]


@pytest.mark.parametrize("capacity", [0, 392 - 4, 399, True])
def test_invalid_capacity_is_rejected(metadata, trace, capacity):
    with pytest.raises(ValueError):
        metadata.decode_values(trace.steps[0], capacity, metadata.reserve_owner_pages(trace))


def make_model():
    layers = []
    for index in range(64):
        if index % 4 != 3:
            gdn = SimpleNamespace(
                prefix=f"model.layers.{index}.linear_attn",
                get_state_shape=lambda: ((6, 5120), (24, 128, 128)),
                get_state_dtype=lambda: (torch.bfloat16, torch.float32),
            )
            layers.append(SimpleNamespace(layer_type="linear_attention", linear_attn=gdn))
        else:
            impl = SimpleNamespace(
                use_bnsd_kv_cache=False,
                pcp_enabled=False,
                enable_c8_quant=False,
                sinks=None,
                sliding_window=None,
                kv_sharing_target_layer_name=None,
            )
            backend = SimpleNamespace(layer_name=f"model.layers.{index}.self_attn.attn", impl=impl)
            attention = SimpleNamespace(attn=backend, num_kv_heads=2, head_dim=256)
            layers.append(SimpleNamespace(layer_type="full_attention", self_attn=attention))
    return SimpleNamespace(language_model=SimpleNamespace(model=SimpleNamespace(layers=layers)))


def test_allocation_binds_all_real_layers_without_dense_candidate_ssm(metadata, trace):
    model = make_model()
    bundle = metadata.allocate_full_caches(
        torch, model, SimpleNamespace(), trace, device="meta", capture_sizes=(384, 400, 512), format_cast=lambda x: x
    )
    assert len(bundle.gdn_entries) == 48
    assert len(bundle.fia_entries) == 16
    bases = [entry.base for entry in bundle.gdn_entries]
    for entry in bundle.gdn_entries:
        assert tuple(entry.base.shape) == (129, 24, 128, 128)
        assert tuple(entry.conv.shape) == (129, 6, 5120)
        assert entry.gdn.kv_cache == (entry.conv, entry.base)
        assert entry.state is None
    for entry in bundle.fia_entries:
        assert tuple(entry.key.shape) == (1293, 128, 2, 256)
        assert entry.impl.key_cache is entry.key
        assert entry.impl.value_cache is entry.value
        assert entry.backend.kv_cache == (entry.key, entry.value)
    assert len({id(entry.key) for entry in bundle.fia_entries}) == 16
    bundle.transition_to_decode()
    assert bundle.decode_ready
    for entry, original in zip(bundle.gdn_entries, bases, strict=True):
        assert entry.base is entry.state.base is original
        assert entry.state.dense_state is None
        assert entry.gdn.kv_cache == (entry.conv, entry.state.handle)
        assert tuple(entry.conv.shape) == (513, 6, 5120)
    with pytest.raises(ValueError, match="exactly once"):
        bundle.transition_to_decode()


def test_transition_copies_real_conv_orientation_into_all_candidate_slots(metadata, monkeypatch):
    backend = importlib.import_module("multi_graph_gdn_q4_log_backend")
    monkeypatch.setattr(
        backend.Q4LogState, "from_base", lambda torch, base, candidates: SimpleNamespace(handle=object(), base=base)
    )
    prefix = torch.arange(3 * 2 * 6, dtype=torch.bfloat16).reshape(3, 2, 6)
    prefix[0].zero_()
    entry = metadata.GDNCacheEntry(0, "gdn", SimpleNamespace(), torch.empty(0), prefix)
    pages = metadata.OwnerPages(((1,), (2,)), (128, 128), 3, 1)
    bundle = metadata.FullCacheBundle(torch, [entry], [], pages, (8,), torch.device("cpu"))
    bundle.transition_to_decode()
    torch.testing.assert_close(entry.conv[0], prefix[0], rtol=0, atol=0)
    for request in range(2):
        for token in range(4):
            torch.testing.assert_close(entry.conv[1 + 4 * request + token], prefix[request + 1], rtol=0, atol=0)


class StubBuilder:
    def __init__(self, spec, names, config, device):
        self.config = config
        self.calls = []

    def build(self, zero, common, **kwargs):
        self.calls.append((common, kwargs))
        prefix = bool(common.is_prefilling[0])
        return SimpleNamespace(
            num_prefills=int(prefix),
            num_decodes=0,
            spec_decode_metadata=None if prefix else SimpleNamespace(),
            num_actual_tokens=common.num_actual_tokens,
        )


@pytest.fixture
def factory(metadata, monkeypatch, trace):
    monkeypatch.setattr(
        metadata,
        "_metadata_types",
        lambda: (
            SimpleNamespace(NONE="NONE", FULL="FULL"),
            SimpleNamespace(PrefillNoCache="PrefillNoCache", ChunkedPrefill="ChunkedPrefill"),
            SimpleNamespace,
            SimpleNamespace,
            StubBuilder,
        ),
    )
    pages = metadata.reserve_owner_pages(trace)
    gdn = SimpleNamespace(get_kv_cache_spec=lambda config: "spec")
    gdns = [SimpleNamespace(prefix=f"gdn{index}", gdn=gdn) for index in range(48)]
    fias = [SimpleNamespace(backend=SimpleNamespace(layer_name=f"fia{index}")) for index in range(16)]
    bundle = metadata.FullCacheBundle(torch, gdns, fias, pages, (384, 400, 512), torch.device("cpu"))
    config = SimpleNamespace(
        cache_config=SimpleNamespace(mamba_cache_mode="align"), compilation_config=SimpleNamespace()
    )
    return metadata.FullMetadataFactory(torch, bundle, config)


def test_prefix_uses_real_owner_complete_token_count_and_zero_computed_history(factory, trace):
    owner, length = 13, trace.initial_prefix_by_request[13]
    frame = factory.prefix(owner, length)
    assert len(frame.mapping) == 64
    assert frame.capacity_tokens == length
    assert frame.active.all()
    assert frame.positions.tolist() == list(range(length))
    builder = factory.builders[(0, True)]
    common, kwargs = builder.calls[-1]
    assert common.num_input_tokens == common.num_actual_tokens == length
    assert common.num_computed_tokens_cpu.tolist() == [0]
    assert common._num_computed_tokens_cpu.tolist() == [0]
    assert common.block_table_tensor.tolist() == [[owner + 1]]
    assert not kwargs
    assert builder.config.compilation_config.cudagraph_mode == "NONE"
    assert builder.config.cache_config.mamba_cache_mode == "none"
    assert factory.config.cache_config.mamba_cache_mode == "align"
    fia = frame.mapping["fia0"]
    assert fia.attn_state == "PrefillNoCache"
    assert fia.seq_lens_list == [length]


def test_decode_keeps_fixed_addresses_and_refreshes_true_lengths_owner_order_and_slots(factory, trace):
    factory.bundle.decode_ready = True
    first = factory.decode(trace.steps[0], 512)
    fia = first.mapping["fia0"]
    tensors = (
        first.positions,
        first.slot_mapping,
        first.active,
        fia.seq_lens,
        fia.seq_lens_gpu,
        fia.block_tables,
        fia.query_start_loc_gpu,
    )
    addresses = [value.data_ptr() for value in tensors]
    second = factory.decode(trace.steps[1], 512)
    assert first is second
    assert second.mapping["fia15"] is fia
    assert [value.data_ptr() for value in tensors] == addresses
    assert fia.seq_lens_list[: trace.steps[1]["batch"]] == list(trace.steps[1]["seq_lengths"])
    assert len(first.mapping) == 64
    common, kwargs = factory.builders[(512, False)].calls[-1]
    assert common.num_input_tokens == 512
    assert kwargs["num_actual_reqs"] == trace.steps[1]["batch"]
    assert kwargs["num_accepted_tokens"][: trace.steps[1]["batch"]].tolist() == list(
        trace.steps[1]["accepted_from_previous"]
    )
    assert first.mapping["gdn0"].num_actual_tokens == 512
    with pytest.raises(ValueError, match="default catalog"):
        factory.decode(trace.steps[0], 396)
    with pytest.raises(ValueError, match="later prefill"):
        factory.prefix(0, 512)


def test_decode_rejects_untransitioned_cache_ownership(factory, trace):
    with pytest.raises(ValueError, match="transition"):
        factory.decode(trace.steps[0], 512)
