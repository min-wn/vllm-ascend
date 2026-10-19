# SPDX-License-Identifier: Apache-2.0

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


def load():
    path = Path(__file__).resolve().parents[4] / "benchmarks/aclgraph_full_core_join.py"
    spec = importlib.util.spec_from_file_location("full_core_join", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_legal_pair_uses_only_catalog_and_whole_requests():
    module = load()
    assert module.legal_pair(400, [192, 200, 208, 400]) == (200, 200)
    assert module.legal_pair(304, [144, 160, 304]) == (160, 144)
    with pytest.raises(ValueError, match="default Q4"):
        module.legal_pair(392, [196, 400])
    with pytest.raises(ValueError, match="default Q4"):
        module.legal_pair(400, [198, 202, 400])


def test_shape_screen_rejects_custom_graph_sizes(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[4] / "benchmarks"))
    from aclgraph_core_join_shape_probe import cases

    catalog = [64, 128, 152, 176, 200, 240, 304, 400]
    assert cases(catalog) == [(304,), (152, 152), (128, 176), (64, 240), (400,), (200, 200)]
    with pytest.raises(ValueError, match="default Q4"):
        cases([n for n in catalog if n != 176])


def test_only_disjoint_adjacent_fia_gdn_pairs_are_accepted():
    module = load()
    layers = [
        SimpleNamespace(layer_type=kind, layer_scale=False, use_attn_reduce_scatter_for_moe=False)
        for kind in ("full_attention", "linear_attention", "full_attention", "linear_attention")
    ]
    module.validate_boundaries(layers, [0, 2])
    for invalid in ([], [0, 0], [1], [3], [True]):
        with pytest.raises(ValueError):
            module.validate_boundaries(layers, invalid)
    layers[1].layer_scale = True
    with pytest.raises(ValueError, match="scaling"):
        module.validate_boundaries(layers, [0])


def test_installed_methods_restore_on_error_and_reject_skipped_consumer():
    module = load()
    schedule = object.__new__(module.CoreJoinSchedule)

    class Layer:
        def forward(self, value):
            return value

    down, layer = Layer(), Layer()
    schedule.pairs = [dict(down=down, layer=layer, pending=object())]
    with pytest.raises(RuntimeError, match="skipped"), schedule.installed():
        assert "forward" in down.__dict__ and "forward" in layer.__dict__
    assert "forward" not in down.__dict__ and "forward" not in layer.__dict__
    assert down.forward(4) == 4


@pytest.mark.parametrize("interleaved", [False, True])
def test_packed_join_preserves_projected_fields_and_zero_rows(monkeypatch, interleaved):
    import torch

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[4] / "benchmarks"))
    from multi_graph_tp_gdn_phases import GDNPhases

    def split_layout(qkvz, ba, *args):
        return qkvz[:, :6], qkvz[:, 6:].reshape(-1, 1, 2), ba[:, :1].contiguous(), ba[:, 1:].contiguous()

    gdn = SimpleNamespace(
        out_proj=SimpleNamespace(reduce_results=True),
        in_proj_qkvz=lambda x: (torch.cat([x, x], dim=-1), None),
        in_proj_ba=lambda x: (x[:, :2].clone(), None),
        gqa_interleaved_layout=interleaved,
        key_dim=2,
        value_dim=2,
        tp_size=1,
        head_v_dim=2,
        head_k_dim=2,
        num_v_heads=1,
        num_k_heads=1,
        _split_ba_for_tp=lambda ba: (ba[:, :1], ba[:, 1:]),
    )
    layer = SimpleNamespace(linear_attn=gdn, input_layernorm=lambda value, prior: (value + prior, value + prior))
    from contextlib import nullcontext

    phases = GDNPhases(layer, lambda name: nullcontext(), core_op=lambda *args: None, split_layout=split_layout)
    value, prior = torch.arange(24).reshape(6, 4).float(), torch.ones(6, 4)
    expected = phases.project(value, prior)
    module = load()
    lanes = [
        module.project_packed(phases, value[:2], prior[:2], "left"),
        module.project_packed(phases, value[2:], prior[2:], "right"),
    ]
    actual = module.join_packed(lanes)
    for field in ("mixed_qkv", "z", "b", "a", "core_output", "residual"):
        assert torch.equal(getattr(actual, field), getattr(expected, field))
    assert actual.owner is phases and actual.name == "full"
    assert torch.count_nonzero(actual.core_output) == 0
    with pytest.raises(ValueError, match="unconsumed"):
        module.join_packed(lanes)


def test_pair_rank_timing_rejects_swapped_or_duplicated_steps(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[4] / "benchmarks"))
    from aclgraph_full_core_join_analysis import pair_samples

    a = dict(trial=0, mode="padding", step=0, batch=98, capacity=400, elapsed_us=10, critical_rank_us=12, device_us=9)
    b = dict(a, elapsed_us=12, device_us=11)
    assert pair_samples([[a], [b]])[0]["device_us"] == 11
    with pytest.raises(ValueError, match="identities"):
        pair_samples([[a], [dict(b, step=1)]])
    with pytest.raises(ValueError, match="duplicate"):
        pair_samples([[a, a], [b, b]])


def test_full_trace_checks_attention_count_and_modified_boundaries(monkeypatch, tmp_path):
    import json

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[4] / "benchmarks"))
    from aclgraph_full_core_join_analysis import analyze_trace

    events, clock = [], 0

    def emit(name, kernel=True):
        nonlocal clock
        events.append(
            dict(name=name, ts=clock, dur=2, tid=1, ph="X", args={"Task Type": "KERNEL_AICORE"} if kernel else {})
        )
        clock += 2

    emit("hcom_allReduce", False)
    for layer in range(64):
        if layer == 4:
            stream_segment_boundary = clock
        for _ in range(1 if layer % 4 == 3 else (4 if layer == 4 else 2)):
            emit("MatMul")
        emit("FusedInferAttentionScore" if layer % 4 == 3 else "q4_tile_solve_1")
        if layer % 4 != 3:
            emit("q4_log_deferred_0")
        emit("MatMul")
        emit("hcom_allReduce", False)
        emit("MatMul")
        emit("MatMul")
        emit("hcom_allReduce", False)
        if layer == 3:
            emit("aclnnInplaceCopy_TensorMove")
            events[-1]["tid"] = 99
            emit("MatMul")
            emit("hcom_allReduce", False)
            emit("aclnnInplaceCopy_TensorMove")
            events[-1]["tid"] = 99
    for event in events:
        if event["ts"] >= stream_segment_boundary and event["tid"] == 1:
            event["tid"] = 2
    path = tmp_path / "trace.json"
    path.write_text(json.dumps(events))
    result = analyze_trace(path, "split_barrier", [3])
    assert result["counts"]["matmul"] == 307
    assert result["counts"]["all_reduce"] == 130
    assert result["boundaries"][0]["input_overlap_last_ar_us"] == 0
    assert result["boundaries"][0]["down_output_copy_us"] == 4
    events = [e for e in events if e["name"] != "FusedInferAttentionScore"]
    path.write_text(json.dumps(events))
    with pytest.raises(ValueError, match="inventory"):
        analyze_trace(path, "split_barrier", [3])


def test_profile_supplement_cannot_be_reported_as_qualified_timing(monkeypatch, tmp_path):
    import json

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[4] / "benchmarks"))
    import aclgraph_full_core_join_analysis as analysis

    (tmp_path / "results").mkdir()
    (tmp_path / "launch.json").write_text(json.dumps(dict(sources={})))
    modes = ["padding", "split_barrier", "split_early"]
    report = dict(
        status="profile_only_complete_not_qualified_or_timed",
        split_schedule=dict(modes=modes, boundaries=[3]),
        selection={},
        prefix=dict(completed_requests=128),
        capture=dict(fia_tasks_per_graph=16),
        profiles=[dict(mode=m, step=1, batch=73, capacity=304, path=m, sha256="test") for m in modes],
    )
    monkeypatch.setattr(analysis, "analyze_trace", lambda *args: dict(sha256="test"))

    def save():
        for rank in (0, 1):
            (tmp_path / "results" / f"diagnostic.rank{rank}.json").write_text(json.dumps(report))

    save()
    assert len(analysis.analyze_profile_supplement(tmp_path)["traces"]) == 6
    with pytest.raises(ValueError, match="incomplete full-model"):
        analysis.analyze(tmp_path)
    report["samples"] = [dict(device_us=1)]
    save()
    with pytest.raises(ValueError, match="without timing"):
        analysis.analyze_profile_supplement(tmp_path)
    del report["samples"]
    report["profiles"].pop()
    save()
    with pytest.raises(ValueError, match="control mode"):
        analysis.analyze_profile_supplement(tmp_path)
