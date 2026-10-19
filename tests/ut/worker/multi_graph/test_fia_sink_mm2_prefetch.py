# SPDX-License-Identifier: Apache-2.0
"""Pinned-source, complete fallback and isolated build contracts."""

import hashlib
import importlib
import json
from pathlib import Path

import pytest


@pytest.fixture
def modules(monkeypatch):
    repo = Path(__file__).resolve().parents[4]
    monkeypatch.syspath_prepend(str(repo / "benchmarks"))
    return tuple(
        importlib.import_module("aclgraph_fia_sink_mm2_prefetch_" + name) for name in ("patch", "build", "contract")
    )


@pytest.fixture
def source():
    repo = Path(__file__).resolve().parents[4]
    return (
        repo
        / "benchmarks/results/multi_graph_structural/fia-sink-mm2-double-buffer-v1"
        / "raw/screen/cube-before.h.txt"
    ).read_bytes()


def test_drift_and_reapplication_fail(modules, source):
    patch, _, _ = modules
    assert hashlib.sha256(source).hexdigest() == patch.EXPECTED_SHA256
    for data in (b"", source + b"\n", patch.transformed(source)):
        with pytest.raises(ValueError, match="pinned, qualified"):
            patch.transformed(data)


def test_only_scheduling_edits_and_original_fallback_are_present(modules, source):
    patch, _, _ = modules
    before, after = source.decode(), patch.transformed(source).decode()
    begin = before.index(patch.SIGNATURE)
    marker = "#endif // FIA_BLOCK_CUBE_NONQUANT_GQA_SINK_H"
    original = before[begin : before.index(marker)]
    compute = after[after.index(patch.SIGNATURE) : after.index(marker)]
    for old, new in reversed(patch.compute_edits(original)):
        assert compute.count(new) == 1
        compute = compute.replace(new, old, 1)
    assert compute == original
    # Everything before ComputeMm2 is byte-for-byte identical after removing
    # the new declarations, gate and helper. Allocations and MM1 remain intact.
    prefix = after[: after.index(patch.SIGNATURE)].replace(patch.HELPER, "", 1)
    prefix = prefix.replace(patch.GUARD, "", 1).replace("\n" + patch.DECLARATION, "", 1)
    assert prefix == before[:begin]
    assert "WaitFlag<HardEvent::MTE1_MTE2>" in patch.HELPER
    assert "WaitFlag<HardEvent::MTE2_MTE1>" not in patch.HELPER
    assert "!isLastK" in patch.NEXT and "k.sizeAct - nextK" in patch.NEXT
    assert "reuseVBuf = false;" in patch.compute_edits(original)[2][1]
    assert (
        compute[compute.index("            SetFlag<HardEvent::MTE1_MTE2>(KP_EVENT0") :]
        == original[original.index("            SetFlag<HardEvent::MTE1_MTE2>(KP_EVENT0") :]
    )


def test_gate_excludes_multiple_m_tiles_and_unsupported_paths(modules):
    patch, _, _ = modules
    for predicate in ("!CFG::ENABLE_UNIFLAG", "CFG::PRELOAD_NUM == 2", "!CFG::S2_BASICSIZE_IS_1024"):
        assert predicate in patch.GUARD
    for predicate in (
        "info.actMBaseSize > 0U",
        "info.actMBaseSize <= D256_MM2_QUERY_ROWS",
        "info.actualSingleProcessSInnerSize > 0U",
        "info.actualSingleProcessSInnerSize <= 512U",
        "!info.isSinkTensor",
        "constInfo.gSize == 6U",
        "constInfo.kvCacheBlockSize == 128U",
    ):
        assert predicate in patch.GATE


def test_dry_run_and_evidence_protection(modules, source, tmp_path):
    patch, _, _ = modules
    path = tmp_path / patch.SOURCE
    path.parent.mkdir(parents=True)
    path.write_bytes(source)
    report = patch.patch_tree(tmp_path, check=True)
    assert path.read_bytes() == source
    assert report["buffer_allocations_unchanged"] and report["original_slot_release_position"]
    assert not report["graph_split_exclusive"] and not report["new_task_or_workspace_prefetch"]
    assert patch.patch_tree(tmp_path) == report
    assert json.loads((tmp_path / "mm2-prefetch-patch.json").read_bytes()) == report
    path.write_bytes(source)
    with pytest.raises(ValueError, match="refusing to overwrite"):
        patch.patch_tree(tmp_path)
    assert path.read_bytes() == source


@pytest.mark.parametrize("enabled", (0, 1))
def test_wrapper_restricts_keys_and_defines_one_macro(modules, enabled):
    patch, builder, _ = modules
    anchor = "    compile_op(src, origin_func_name, op_info, options, code_channel, '{}', {'valueDepend': {}})"
    result = builder.wrapper_options(anchor, enabled)
    assert result.count(f"-D{patch.MACRO}={enabled}") == 1
    assert builder.KEYS == ("104000000010222223", "104000000010322223")
    with pytest.raises(ValueError, match="already patched"):
        builder.wrapper_options(result, enabled)


@pytest.mark.parametrize("enabled", (True, -1, 2, "1"))
def test_invalid_macro_rejected(modules, enabled):
    _, builder, _ = modules
    with pytest.raises(ValueError, match="prefetch"):
        builder.wrapper_options("wrapper", enabled)


def test_cpu_program_uses_complete_actual_methods(modules, source):
    patch, _, contract = modules
    repo = Path(__file__).resolve().parents[4]
    axis = (
        repo / "benchmarks/results/multi_graph_structural/fia-sink-mm2-double-buffer-v1" / "raw/screen/axis.h.txt"
    ).read_text(encoding="utf-8")
    program = contract.program(source, axis)
    original = source.decode()
    original = original[
        original.index(patch.SIGNATURE) : original.index("#endif // FIA_BLOCK_CUBE_NONQUANT_GQA_SINK_H")
    ]
    assert original.replace("FiaBlockCubeNonQuantGqa", "Original") in program
    assert patch.HELPER in program
    assert "struct AxisSlices" in program and "void balanced()" in program
    assert "call<3" in program and "n<=512" in program
