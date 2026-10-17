"""CPU checks of GDN decomposition against the current official forward body."""

import ast
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def phases_module(monkeypatch):
    directory = Path(__file__).resolve().parents[4] / "benchmarks"
    monkeypatch.syspath_prepend(str(directory))
    import multi_graph_tp_gdn_phases

    return multi_graph_tp_gdn_phases


def cpu_interleaved_layout(qkvz, ba, nk, nv, dk, dv):
    """Reference layout described by the official fused Triton kernel."""
    rows = qkvz.shape[0]
    values_per_group = nv // nk
    groups = qkvz.reshape(rows, nk, dk * 2 + values_per_group * dv * 2)
    q, k, v, z = groups.split([dk, dk, values_per_group * dv, values_per_group * dv], dim=-1)
    b, a = ba.reshape(rows, nk, values_per_group * 2).chunk(2, dim=-1)
    return (
        torch.cat([q.flatten(1), k.flatten(1), v.flatten(1)], dim=-1),
        z.reshape(rows, nv, dv),
        b.reshape(rows, nv).contiguous(),
        a.reshape(rows, nv).contiguous(),
    )


def official_forward(core_op, split_layout):
    """Execute the repository's actual forward math, rather than duplicating it."""
    source = Path(__file__).resolve().parents[4] / "vllm_ascend" / "ops" / "gdn.py"
    definition = next(
        node for node in ast.parse(source.read_text(encoding="utf-8")).body if isinstance(node, ast.ClassDef)
    )
    forward = next(node for node in definition.body if isinstance(node, ast.FunctionDef) and node.name == "forward")
    namespace = dict(
        torch=SimpleNamespace(
            Tensor=torch.Tensor,
            zeros=torch.zeros,
            ops=SimpleNamespace(vllm=SimpleNamespace(qwen_gdn_attention_core=core_op)),
        ),
        fused_qkvzba_split_reshape_cat=split_layout,
        triton=SimpleNamespace(cdiv=lambda a, b: (a + b - 1) // b),
        rearrange=lambda value, pattern: value.flatten(-2),
    )
    exec(compile(ast.Module(body=[forward], type_ignores=[]), str(source), "exec"), namespace)
    return namespace["forward"]


def make_fixture(layout, rows, dtype):
    torch.manual_seed(512)
    tp, nk, nv, dk, dv, hidden = 2, 4, 8, 2, 3, 7
    active, log = [None], []
    state = torch.randn(rows, nv // tp, dv, dtype=dtype)
    initial = state.clone()
    gdn = SimpleNamespace(
        prefix="actual.gdn",
        tp_size=tp,
        num_k_heads=nk,
        num_v_heads=nv,
        head_k_dim=dk,
        head_v_dim=dv,
        key_dim=nk * dk,
        value_dim=nv * dv,
        gqa_interleaved_layout=layout == "interleaved",
    )

    @contextmanager
    def context(name):
        assert active[0] is None
        active[0] = name
        log.append(("enter", name))
        try:
            yield
        finally:
            log.append(("exit", name))
            active[0] = None

    class Projection:
        def __init__(self, name, output_dim, input_dim=hidden):
            self.name = name
            self.weight = torch.randn(output_dim, input_dim, dtype=dtype) * 0.1
            self.reduce_results = True
            self.custom_op = self.bias = None
            self.fail = False

        def __call__(self, value):
            assert active[0] is not None
            log.append((self.name, active[0]))
            if self.name == "output":
                assert not self.reduce_results
            if self.fail:
                raise RuntimeError("projection failure")
            return torch.nn.functional.linear(value, self.weight), None

    qkv_size = (gdn.key_dim * 2 + gdn.value_dim) // tp
    z_size = gdn.value_dim // tp
    if layout == "unpacked":
        gdn.in_proj_qkv = Projection("qkv", qkv_size)
        gdn.in_proj_z = Projection("z", z_size)
    else:
        gdn.in_proj_qkvz = Projection("qkvz", qkv_size + z_size)
    gdn.in_proj_ba = Projection("ba", 2 * nv // tp)
    gdn.out_proj = Projection("output", hidden, z_size)
    gdn._split_ba_for_tp = lambda value: value.chunk(2, dim=-1)

    def gated_norm(core, z):
        assert active[0] is not None
        log.append(("norm", active[0]))
        return torch.tanh(core) * torch.sigmoid(z)

    gdn.norm = gated_norm

    def core_op(qkv, b, a, output, prefix, flag):
        assert active[0] is not None and prefix == gdn.prefix and flag is False
        assert b.is_contiguous() and a.is_contiguous()
        assert not output.count_nonzero()
        log.append(("core", active[0]))
        # Last row models an inactive request, whose output must remain zero.
        update = qkv[:-1, -z_size:].reshape(rows - 1, nv // tp, dv)
        state[:-1].add_(update * torch.sigmoid(b[:-1]).unsqueeze(-1))
        output[:-1].copy_(state[:-1] + a[:-1].unsqueeze(-1))

    def input_norm(value, prior):
        assert active[0] is not None
        log.append(("input_norm", active[0]))
        residual = value + prior
        return residual * 0.5, residual

    def split_layout(qkvz, ba, local_nk, local_nv, head_k, head_v):
        assert (local_nk, local_nv, head_k, head_v) == (nk // tp, nv // tp, dk, dv)
        assert active[0] is not None
        log.append(("layout", active[0]))
        return cpu_interleaved_layout(qkvz, ba, local_nk, local_nv, head_k, head_v)

    layer = SimpleNamespace(linear_attn=gdn, input_layernorm=input_norm)
    value, prior = torch.randn(rows, hidden, dtype=dtype), torch.randn(rows, hidden, dtype=dtype)
    return SimpleNamespace(**locals())


@pytest.mark.parametrize("layout", ["unpacked", "packed", "interleaved"])
@pytest.mark.parametrize("rows", [2, 7])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_gdn_phases_match_actual_official_forward_and_state(phases_module, layout, rows, dtype):
    fixture = make_fixture(layout, rows, dtype)
    adapter = phases_module.make_gdn_phases(
        fixture.layer, fixture.context, core_op=fixture.core_op, split_layout=fixture.split_layout
    )
    with fixture.context("left"):
        normalized, residual = fixture.layer.input_layernorm(fixture.value, fixture.prior)
        fixture.gdn.out_proj.reduce_results = False
        expected = official_forward(fixture.core_op, fixture.split_layout)(fixture.gdn, normalized)
        fixture.gdn.out_proj.reduce_results = True
    expected_state = fixture.state.clone()
    fixture.state.copy_(fixture.initial)
    fixture.log.clear()

    phase = adapter.project(fixture.value, fixture.prior, "left")
    torch.testing.assert_close(fixture.state, fixture.initial, rtol=0, atol=0)
    assert phase.core_output.shape == (rows, fixture.nv // fixture.tp, fixture.dv)
    assert not phase.core_output.count_nonzero()
    assert not any(event[0] in ("core", "norm", "output") for event in fixture.log)
    core = adapter.core(phase)
    assert core.shape == (rows, fixture.z_size)
    assert not any(event[0] == "output" for event in fixture.log)
    actual, actual_residual = adapter.output(phase)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_residual, residual, rtol=0, atol=0)
    torch.testing.assert_close(fixture.state, expected_state, rtol=0, atol=0)
    torch.testing.assert_close(fixture.state[-1], fixture.initial[-1], rtol=0, atol=0)
    assert fixture.gdn.out_proj.reduce_results
    assert fixture.active[0] is None
    assert fixture.log.count(("enter", "left")) == fixture.log.count(("exit", "left")) == 3


@pytest.mark.parametrize("original", [False, True])
def test_gdn_output_restores_reduction_switch_on_failure(phases_module, original):
    fixture = make_fixture("packed", 3, torch.float32)
    adapter = phases_module.make_gdn_phases(fixture.layer, fixture.context, core_op=fixture.core_op)
    phase = adapter.project(fixture.value, fixture.prior, "right")
    adapter.core(phase)
    fixture.gdn.out_proj.reduce_results = original
    fixture.gdn.out_proj.fail = True
    with pytest.raises(RuntimeError, match="projection failure"):
        adapter.output(phase)
    assert fixture.gdn.out_proj.reduce_results is original
    assert fixture.active[0] is None
    assert not phase.completed


def test_gdn_phase_guards_prevent_wrong_metadata_and_duplicate_state_updates(phases_module):
    fixture = make_fixture("packed", 3, torch.float32)
    adapter = phases_module.make_gdn_phases(fixture.layer, fixture.context, core_op=fixture.core_op)
    other = phases_module.make_gdn_phases(fixture.layer, fixture.context, core_op=fixture.core_op)
    phase = adapter.project(fixture.value, fixture.prior, "left")
    with pytest.raises(ValueError, match="metadata lane"):
        adapter.core(phase, "right")
    with pytest.raises(ValueError, match="another adapter"):
        other.core(phase)
    with pytest.raises(ValueError, match="core first"):
        adapter.output(phase)
    adapter.core(phase)
    updated_state = fixture.state.clone()
    with pytest.raises(ValueError, match="only run once"):
        adapter.core(phase)
    torch.testing.assert_close(fixture.state, updated_state, rtol=0, atol=0)
    adapter.output(phase)
    with pytest.raises(ValueError, match="already been consumed"):
        adapter.output(phase)


@pytest.mark.parametrize("unsupported", ["custom_op", "bias", "reduce_results"])
def test_gdn_phases_reject_unsupported_output_projections(phases_module, unsupported):
    fixture = make_fixture("packed", 3, torch.float32)
    if unsupported == "reduce_results":
        del fixture.gdn.out_proj.reduce_results
    else:
        setattr(fixture.gdn.out_proj, unsupported, object())
    with pytest.raises(ValueError, match="projection|reduction switch"):
        phases_module.make_gdn_phases(fixture.layer, fixture.context, core_op=fixture.core_op)


@pytest.mark.parametrize("layout", ["unpacked", "packed", "interleaved"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_shared_projection_partition_preserves_global_state_and_tensor_views(phases_module, layout, dtype):
    fixture = make_fixture(layout, 8, dtype)
    intervals = {"full": (0, 8), "left": (0, 3), "right": (3, 8)}

    def core(qkv, b, a, output, prefix, flag):
        start, stop = intervals[fixture.active[0]]
        assert prefix == fixture.gdn.prefix and not flag
        live = min(stop, 7) - start
        update = qkv[:live, -fixture.z_size :].reshape(live, fixture.nv // fixture.tp, fixture.dv)
        fixture.state[start : start + live].add_(update * (b[:live] + 1).unsqueeze(-1))
        output[:live].copy_(fixture.state[start : start + live] + a[:live].unsqueeze(-1))

    adapter = phases_module.make_gdn_phases(
        fixture.layer, fixture.context, core_op=core, split_layout=fixture.split_layout
    )
    whole = adapter.project(fixture.value, fixture.prior, "full")
    adapter.core(whole)
    expected, expected_residual = adapter.output(whole)
    expected_state = fixture.state.clone()
    fixture.state.copy_(fixture.initial)
    projected = adapter.project(fixture.value, fixture.prior, "full")
    children = adapter.partition(projected, boundaries=(0, 3, 8))
    for child in children:
        for name in ("mixed_qkv", "z", "b", "a", "core_output", "residual"):
            assert (
                getattr(child, name).untyped_storage().data_ptr()
                == getattr(projected, name).untyped_storage().data_ptr()
            )
    outputs, residuals = [], []
    for child in children:
        adapter.core(child)
        output, residual = adapter.output(child)
        outputs.append(output)
        residuals.append(residual)
    torch.testing.assert_close(torch.cat(outputs), expected, rtol=0.02, atol=0.02)
    torch.testing.assert_close(torch.cat(residuals), expected_residual, rtol=0, atol=0)
    torch.testing.assert_close(fixture.state, expected_state, rtol=0, atol=0)
    torch.testing.assert_close(fixture.state[-1], fixture.initial[-1], rtol=0, atol=0)
    with pytest.raises(ValueError, match="consumed"):
        adapter.core(projected)
    with pytest.raises(ValueError, match="consumed"):
        adapter.partition(projected, boundaries=(0, 3, 8))


@pytest.mark.parametrize(
    "names,boundaries",
    [
        (("left", "right"), (0, 0, 8)),
        (("left", "right"), (1, 3, 8)),
        (("left", "right"), (0, 3, 7)),
        (("left", "left"), (0, 3, 8)),
        (("left", "right"), (0, True, 8)),
    ],
)
def test_partition_rejects_incomplete_or_overlapping_metadata_lanes(phases_module, names, boundaries):
    fixture = make_fixture("packed", 8, torch.float32)
    adapter = phases_module.make_gdn_phases(fixture.layer, fixture.context, core_op=fixture.core_op)
    projected = adapter.project(fixture.value, fixture.prior, "full")
    with pytest.raises(ValueError):
        adapter.partition(projected, names=names, boundaries=boundaries)
    assert not projected.completed
