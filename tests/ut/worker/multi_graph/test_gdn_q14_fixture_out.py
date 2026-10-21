# SPDX-License-Identifier: Apache-2.0
"""The fixture experiment must change only the gather destination contract."""

import ast
import hashlib
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / "benchmarks/multi_graph_gdn_q14_fixture_out.py"
FROZEN = ROOT / "benchmarks/multi_graph_gdn_q14_probe.py"


def module():
    assert SOURCE.exists(), "independent out fixture candidate is not implemented"
    sys.path.insert(0, str(SOURCE.parent))
    try:
        spec = importlib.util.spec_from_file_location("q14_fixture_out_tested", SOURCE)
        result = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(result)
        return result
    finally:
        sys.path.pop(0)


def prepare_node(path, class_name):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    return next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "prepare")


def test_out_prepare_changes_only_the_explicit_temporary_and_copy():
    module()
    original = prepare_node(FROZEN, "Trial")
    loop = next(n for n in original.body if isinstance(n, ast.For) and ast.unparse(n.target) == "(name, source)")
    assert ast.unparse(loop.body[0]) == "self.buffers[name][:count].copy_(source.index_select(0, select))"
    loop.body = ast.parse("torch.index_select(source, 0, select, out=self.buffers[name][:count])").body
    assert ast.dump(prepare_node(SOURCE, "OutTrial")) == ast.dump(original)


def test_frozen_probe_is_unmodified():
    assert hashlib.sha256(FROZEN.read_bytes().replace(b"\r\n", b"\n")).hexdigest() == (
        "bde7a1c1e4d9e401f9ce37528e2442f018cd9c067ecec3877494c557d7872b5a"
    )


def test_four_config_order_and_repeats_are_frozen():
    candidate = module()
    expected = ("legacy_forced_q4", "legacy_q14_ready", "out_forced_q4", "out_q14_ready")
    assert expected == candidate.CONFIGS
    assert candidate.REPEATS == 8
    assert [candidate.round_order(i) for i in range(3)] == [expected, expected[::-1], expected]


def test_primary_screen_compares_both_out_sides_and_keeps_all_samples():
    candidate = module()
    samples = []
    endpoints = (20.0, 22.0, 12.0, 11.0)
    for round_id in range(3):
        for config, cost in zip(candidate.CONFIGS, endpoints, strict=True):
            for sample in range(8):
                samples.append(dict(config=config, round=round_id, sample=sample, lifecycle_ms=cost))
    result = candidate.summarize(samples)
    assert result["primary_out_vs_out"]["stable"]
    assert result["primary_out_vs_out"]["pooled_improvement_percent"] == pytest.approx(100 / 12)
    assert not result["legacy_vs_legacy"]["stable"]
    assert result["sample_count"] == 96
    with pytest.raises(ValueError, match="96"):
        candidate.summarize(samples[:-1])


def test_duplicate_sample_is_rejected_instead_of_silently_reweighted():
    candidate = module()
    samples = [
        dict(config=c, round=r, sample=s, lifecycle_ms=10.0)
        for r in range(3)
        for c in candidate.CONFIGS
        for s in range(8)
    ]
    samples[-1] = samples[-2].copy()
    with pytest.raises(ValueError, match="identity"):
        candidate.summarize(samples)
