# SPDX-License-Identifier: Apache-2.0
"""Isolation, complete-layout fallback and unchanged execution contracts."""

import hashlib
import importlib
import json
from pathlib import Path

import pytest


@pytest.fixture
def modules(monkeypatch):
    repo = Path(__file__).resolve().parents[4]
    monkeypatch.syspath_prepend(str(repo / "benchmarks"))
    return (
        importlib.import_module("aclgraph_fia_sink_mm2_double_buffer_patch"),
        importlib.import_module("aclgraph_fia_sink_mm2_double_buffer_build"),
    )


@pytest.fixture
def qualified_source():
    repo = Path(__file__).resolve().parents[4]
    return (
        repo / "benchmarks/results/multi_graph_structural/fia-sink-mm2-transfer-v1" / "raw/screen/cube-before.h.txt"
    ).read_bytes()


def test_source_drift_and_reapplication_fail(modules, qualified_source):
    patch, _ = modules
    assert hashlib.sha256(qualified_source).hexdigest() == patch.EXPECTED_SHA256
    for data in (b"", qualified_source + b"\n", patch.transformed(qualified_source)):
        with pytest.raises(ValueError, match="pinned, qualified"):
            patch.transformed(data)


def test_math_copy_lifetimes_and_event_ids_are_unchanged(modules, qualified_source):
    patch, _ = modules
    before = qualified_source.decode()
    after = patch.transformed(qualified_source).decode()
    begin = "__aicore__ inline void FiaBlockCubeNonQuantGqa<FIAT, Config>::CopyQToL1"
    end = "__aicore__ inline void FiaBlockCubeNonQuantGqa<FIAT, Config>::ComputeMm2"
    assert before[before.index(begin) : before.index(end)] == after[after.index(begin) : after.index(end)]
    marker = "    bool canFullLoadV = (KL1Slices.size() <= L1_V_BUFCNT);"
    assert before[before.index(marker) :] == after[after.index(marker) :]
    begin = (
        "template <typename FIAT, typename Config> __aicore__ inline void "
        "FiaBlockCubeNonQuantGqa<FIAT, Config>::AllocEventID"
    )
    end = "__aicore__ inline void FiaBlockCubeNonQuantGqa<FIAT, Config>::CopyQToL1"
    assert before[before.index(begin) : before.index(end)] == after[after.index(begin) : after.index(end)]
    assert "L1_V_BUFCNT = CFG::S2_BASICSIZE_IS_1024 ? 4 : 2;" in after
    assert "LocalTensor<Q_T> qL1Tensor[L1_Q_BUFCNT]" in after
    assert "LocalTensor<KV_T> vL1Tensor[L1_V_BUFCNT]" in after
    assert "const uint32_t K_BASE = K_SPLIT_SIZE;" in after


def test_geometry_selects_allocation_not_only_transfer_split(modules, qualified_source):
    patch, _ = modules
    for predicate in (
        "constInfo.headDim == 256U",
        "constInfo.headDimRope == 0U",
        "!constInfo.ropeSplitMode",
        "constInfo.keySinkNumber == 0U",
        "constInfo.gSize == 6U",
        "constInfo.kvHeadNum == 2U",
        "constInfo.qHeadNum == 12U",
        "constInfo.qSeqSize == 4U",
        "constInfo.headDimAlign == 256U",
        "constInfo.preLoadNum == 2U",
        "constInfo.kvCacheBlockSize == 128U",
        "constInfo.subBlockNum == 2U",
        "constInfo.sinkNumber == 0",
        "constInfo.s2BaseSize > 0U",
        "constInfo.s2BaseSize <= 512U",
    ):
        assert predicate in patch.Q_ALLOCATION
    assert "CFG::PRELOAD_NUM == 2" in patch.GUARD
    assert "!CFG::S2_BASICSIZE_IS_1024" in patch.GUARD
    after = patch.transformed(qualified_source).decode()
    assert after.index("thesisDoubleVEnabled = THESIS_DOUBLE_V_STATIC") < after.index("pipe->InitBuffer(L1_Q")
    assert "thesisDoubleVEnabled ? THESIS_DOUBLE_SLOT_ELEMENTS : L1_Q_SIZE" in patch.Q_ALLOCATION
    assert "thesisDoubleVEnabled ? THESIS_DOUBLE_SLOT_ELEMENTS : L1_SINK_V_SIZE" in patch.V_ALLOCATION
    assert "THESIS_DOUBLE_V_WIDE && thesisDoubleVEnabled && !info.isSinkTensor" in patch.SPLIT


def test_dry_run_and_evidence_overwrite_protection(modules, qualified_source, tmp_path):
    patch, _ = modules
    path = tmp_path / patch.SOURCE
    path.parent.mkdir(parents=True)
    path.write_bytes(qualified_source)
    report = patch.patch_tree(tmp_path, check=True)
    assert path.read_bytes() == qualified_source
    assert report["l1_total_bytes"] == 464 * 1024
    assert report["event_ids_and_slot_counts_unchanged"]
    assert not report["graph_split_exclusive"]
    assert patch.patch_tree(tmp_path) == report
    assert json.loads((tmp_path / "mm2-double-buffer-patch.json").read_text()) == report
    path.write_bytes(qualified_source)
    with pytest.raises(ValueError, match="refusing to overwrite"):
        patch.patch_tree(tmp_path)
    assert path.read_bytes() == qualified_source


@pytest.mark.parametrize("enabled", (0, 1))
def test_same_layout_compile_toggle_and_existing_keys(modules, enabled):
    patch, builder = modules
    call = "    compile_op(src, origin_func_name, op_info, options, code_channel, '{}', {'valueDepend': {}})"
    result = builder.wrapper_options(call, enabled)
    assert result.count("-DTHESIS_FIA_MM2_DOUBLE_V_LAYOUT=1") == 1
    assert result.count(f"-D{patch.MACRO}={enabled}") == 1
    assert builder.KEYS == ("104000000010222223", "104000000010322223")
    with pytest.raises(ValueError, match="already patched"):
        builder.wrapper_options(result, enabled)


@pytest.mark.parametrize("enabled", (True, -1, 2, "1"))
def test_invalid_macro_value_is_rejected(modules, enabled):
    _, builder = modules
    with pytest.raises(ValueError, match="wide-v"):
        builder.wrapper_options("generated wrapper", enabled)
