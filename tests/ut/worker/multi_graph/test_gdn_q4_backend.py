"""CPU adapter contracts; native kernel and model qualification run separately."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def backend_module():
    path = Path(__file__).resolve().parents[4] / "benchmarks" / "multi_graph_gdn_q4_backend.py"
    spec = importlib.util.spec_from_file_location("q4_backend_contract", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_backend(module, state_dtype=torch.bfloat16):
    calls = []

    def products(state, query, key, sequence, ids, accepted, output):
        calls.append(("products", state, output, query, key))
        output.fill_(3)

    def emit(product, query, key, value, beta, gate, state, sequence, ids, accepted, output):
        calls.append(("emit", state, product, query, key))
        assert not output.count_nonzero()
        output.fill_(7)

    def original(*args, **kwargs):
        calls.append(("original", args, kwargs))
        return "official"

    backend = module.Q4Backend(SimpleNamespace(emit=emit), SimpleNamespace(products=products), torch, original)
    query = torch.ones(8, 8, 128, dtype=torch.bfloat16)
    value = torch.ones(8, 24, 128, dtype=torch.bfloat16)
    # Model-style pages include a gap before the next SSM slot.
    state = torch.empty_strided((9, 24, 128, 128), (400384, 16384, 128, 1), dtype=state_dtype)
    kwargs = dict(
        g=torch.ones(8, 24),
        beta=torch.ones(8, 24, dtype=torch.bfloat16),
        scale=128**-0.5,
        actual_seq_lengths=torch.tensor([0, 4, 4], dtype=torch.int32),
        ssm_state_indices=torch.arange(1, 9, dtype=torch.int32),
        num_accepted_tokens=torch.tensor([1, 4], dtype=torch.int32),
    )
    return backend, calls, (query, query.clone(), value, state), kwargs


@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float32])
def test_adapter_keeps_page_layout_and_reuses_buffers_but_clears_output(backend_module, state_dtype):
    backend, calls, args, kwargs = make_backend(backend_module, state_dtype)
    first = backend(*args, **kwargs)
    second = backend(*args, **kwargs)
    assert first is second and backend.calls == 2 and len(backend.buffers) == 1
    assert [call[0] for call in calls] == ["products", "emit", "products", "emit"]
    assert calls[0][1] is args[3] and calls[1][1] is args[3]
    assert backend.describe()["state_layouts"][0]["stride"][0] == 400384
    assert backend.describe()["state_layouts"][0]["dtype"] == str(state_dtype)
    assert not args[3].is_contiguous()


@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("metadata", ["actual_seq_lengths", "ssm_state_indices", "num_accepted_tokens"])
def test_parallel_metadata_lanes_get_distinct_scratch(backend_module, state_dtype, metadata):
    backend, calls, args, kwargs = make_backend(backend_module, state_dtype)
    first = backend(*args, **kwargs)
    kwargs[metadata] = kwargs[metadata].clone()
    second = backend(*args, **kwargs)
    assert first.data_ptr() != second.data_ptr() and len(backend.buffers) == 2
    assert calls[0][2].data_ptr() != calls[2][2].data_ptr()


@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float32])
def test_product_conversion_preserves_state_and_block_precision(backend_module, state_dtype):
    backend, calls, args, kwargs = make_backend(backend_module, state_dtype)
    state = args[3]
    # The low bits must survive; rounding this state to BF16 would lose them.
    state.fill_(1.0001)
    before = state.clone()
    backend(*args, **kwargs)
    product_query, product_key = calls[0][3:]
    assert calls[0][1] is state and calls[1][1] is state
    assert torch.equal(state, before)
    assert product_query.dtype == product_key.dtype == state_dtype
    assert calls[1][3] is args[0] and calls[1][4] is args[1]
    assert torch.equal(product_query, args[0].to(state_dtype))
    assert torch.equal(product_key, args[1].to(state_dtype))
    if state_dtype == torch.bfloat16:
        assert product_query is args[0] and product_key is args[1]
    else:
        assert product_query is not args[0] and product_key is not args[1]
        args[0].fill_(2)
        args[1].fill_(4)
        backend(*args, **kwargs)
        assert torch.equal(calls[2][3], args[0].float())
        assert torch.equal(calls[2][4], args[1].float())
        assert calls[2][3] is not product_query and calls[2][4] is not product_key
        assert len(backend.buffers) == 1


def test_distinct_state_cache_uses_distinct_scratch(backend_module):
    backend, calls, args, kwargs = make_backend(backend_module, torch.float32)
    first = backend(*args, **kwargs)
    other = torch.empty_strided(args[3].shape, args[3].stride(), dtype=torch.float32)
    second = backend(*args[:3], other, **kwargs)
    assert first.data_ptr() != second.data_ptr() and len(backend.buffers) == 2
    assert calls[0][2].data_ptr() != calls[2][2].data_ptr()


def test_non_spec_dispatch_preserves_official_call(backend_module):
    backend, calls, args, kwargs = make_backend(backend_module)
    kwargs.pop("num_accepted_tokens")
    assert backend(*args, **kwargs) == "official"
    assert backend.calls == 0 and backend.original_calls == 1 and calls[0][0] == "original"


@pytest.mark.parametrize(
    "bad",
    [
        "query",
        "heads",
        "dtype",
        "stride",
        "slot_stride",
        "slots",
        "metadata",
        "metadata_dtype",
        "metadata_rank",
        "metadata_contiguous",
        "gating",
        "gate_dtype",
        "query_dtype",
        "query_contiguous",
        "device",
        "scale",
        "option",
    ],
)
def test_q4_incompatible_contract_never_silently_falls_back(backend_module, bad):
    backend, calls, args, kwargs = make_backend(backend_module)
    args = list(args)
    if bad == "query":
        args[0] = args[0][:4]
    elif bad == "heads":
        args[2] = args[2][:, :12]
    elif bad == "dtype":
        args[3] = args[3].half()
    elif bad == "stride":
        args[3] = args[3].transpose(-1, -2)
    elif bad == "slot_stride":
        args[3] = args[3].as_strided(args[3].shape, (1, 16384, 128, 1))
    elif bad == "slots":
        args[3] = args[3][:-1]
    elif bad == "metadata":
        kwargs["actual_seq_lengths"] = None
    elif bad == "metadata_dtype":
        kwargs["num_accepted_tokens"] = kwargs["num_accepted_tokens"].long()
    elif bad == "metadata_rank":
        kwargs["num_accepted_tokens"] = kwargs["num_accepted_tokens"].view(1, 2)
    elif bad == "metadata_contiguous":
        kwargs["ssm_state_indices"] = torch.arange(16, dtype=torch.int32)[::2]
    elif bad == "gating":
        kwargs["beta"] = None
    elif bad == "gate_dtype":
        kwargs["g"] = kwargs["g"].bfloat16()
    elif bad == "query_dtype":
        args[0] = args[0].float()
    elif bad == "query_contiguous":
        args[0] = torch.empty(8, 8, 256, dtype=torch.bfloat16)[:, :, ::2]
    elif bad == "device":
        args[1] = args[1].to(device="meta")
    elif bad == "scale":
        kwargs["scale"] = 1.0
    else:
        kwargs["unqualified_option"] = True
    with pytest.raises(ValueError):
        backend(*args, **kwargs)
    assert not calls
