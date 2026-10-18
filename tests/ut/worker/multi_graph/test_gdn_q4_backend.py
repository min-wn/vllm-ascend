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


def make_backend(module):
    calls = []

    def products(state, query, key, sequence, ids, accepted, output):
        calls.append(("products", state, output))
        output.fill_(3)

    def emit(product, query, key, value, beta, gate, state, sequence, ids, accepted, output):
        calls.append(("emit", state, product))
        assert not output.count_nonzero()
        output.fill_(7)

    def original(*args, **kwargs):
        calls.append(("original", args, kwargs))
        return "official"

    backend = module.Q4Backend(SimpleNamespace(emit=emit), SimpleNamespace(products=products), torch, original)
    query = torch.ones(8, 8, 128, dtype=torch.bfloat16)
    value = torch.ones(8, 24, 128, dtype=torch.bfloat16)
    # Model-style pages include a gap before the next SSM slot.
    state = torch.empty_strided((9, 24, 128, 128), (400384, 16384, 128, 1), dtype=torch.bfloat16)
    kwargs = dict(
        g=torch.ones(8, 24),
        beta=torch.ones(8, 24, dtype=torch.bfloat16),
        scale=128**-0.5,
        actual_seq_lengths=torch.tensor([0, 4, 4], dtype=torch.int32),
        ssm_state_indices=torch.arange(1, 9, dtype=torch.int32),
        num_accepted_tokens=torch.tensor([1, 4], dtype=torch.int32),
    )
    return backend, calls, (query, query.clone(), value, state), kwargs


def test_adapter_keeps_page_layout_and_reuses_buffers_but_clears_output(backend_module):
    backend, calls, args, kwargs = make_backend(backend_module)
    first = backend(*args, **kwargs)
    second = backend(*args, **kwargs)
    assert first is second and backend.calls == 2 and len(backend.buffers) == 1
    assert [call[0] for call in calls] == ["products", "emit", "products", "emit"]
    assert calls[0][1] is args[3] and calls[1][1] is args[3]
    assert backend.describe()["state_layouts"][0]["stride"][0] == 400384
    assert not args[3].is_contiguous()


def test_parallel_metadata_lanes_get_distinct_scratch(backend_module):
    backend, _, args, kwargs = make_backend(backend_module)
    first = backend(*args, **kwargs)
    kwargs["actual_seq_lengths"] = kwargs["actual_seq_lengths"].clone()
    second = backend(*args, **kwargs)
    assert first.data_ptr() != second.data_ptr() and len(backend.buffers) == 2


def test_non_spec_dispatch_preserves_official_call(backend_module):
    backend, calls, args, kwargs = make_backend(backend_module)
    kwargs.pop("num_accepted_tokens")
    assert backend(*args, **kwargs) == "official"
    assert backend.calls == 0 and backend.original_calls == 1 and calls[0][0] == "original"


@pytest.mark.parametrize("bad", ["query", "heads", "dtype", "stride", "metadata", "gating", "scale", "option"])
def test_q4_incompatible_contract_never_silently_falls_back(backend_module, bad):
    backend, calls, args, kwargs = make_backend(backend_module)
    args = list(args)
    if bad == "query":
        args[0] = args[0][:4]
    elif bad == "heads":
        args[2] = args[2][:, :12]
    elif bad == "dtype":
        args[3] = args[3].float()
    elif bad == "stride":
        args[3] = args[3].transpose(-1, -2)
    elif bad == "metadata":
        kwargs["actual_seq_lengths"] = None
    elif bad == "gating":
        kwargs["beta"] = None
    elif bad == "scale":
        kwargs["scale"] = 1.0
    else:
        kwargs["unqualified_option"] = True
    with pytest.raises(ValueError):
        backend(*args, **kwargs)
    assert not calls
