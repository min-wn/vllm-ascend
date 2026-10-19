# SPDX-License-Identifier: Apache-2.0
"""Source-pinned resident-O transformation and real C++ ownership regressions."""

import hashlib
import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.fixture
def patch_module():
    root = Path(__file__).resolve().parents[4]
    spec = importlib.util.spec_from_file_location(
        "fia_running_o_patch_cpu", root / "benchmarks/aclgraph_fia_sink_running_o_patch.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def pinned_source(patch_module):
    root = Path(__file__).resolve().parents[6]
    path = root / "outputs/thesis_incremental_plan/pinned_sink_source/source" / patch_module.SOURCE
    if not path.is_file():
        pytest.skip("pinned native source archive is unavailable in this checkout")
    return path.read_bytes()


def test_only_exact_unmodified_vector_source_is_accepted(patch_module, pinned_source):
    assert hashlib.sha256(pinned_source).hexdigest() == patch_module.EXPECTED_SHA256
    patched = patch_module.transformed(pinned_source)
    assert patch_module.STATE_HELPER.encode() in patched
    assert patch_module.METHODS.encode() in patched
    for invalid in (pinned_source + b"\n", pinned_source.replace(b"ComputeVec2", b"ComputeVecX", 1), patched):
        with pytest.raises(ValueError, match="pinned unmodified"):
            patch_module.transformed(invalid)


@pytest.mark.parametrize("invalid", [b"", b"untrusted source", b"\xff\xfe\x00\x00"])
def test_wrong_source_is_rejected_before_decoding(patch_module, invalid):
    with pytest.raises(ValueError, match="pinned unmodified"):
        patch_module.transformed(invalid)


def test_anchor_changes_fail_closed(patch_module):
    for text in ("missing", "anchor anchor"):
        with pytest.raises(ValueError, match="anchor changed"):
            patch_module.replace_once(text, "anchor", "replacement")
    assert patch_module.replace_once("before anchor after", "anchor", "x") == "before x after"


def test_dry_run_writes_nothing_and_real_patch_records_hashes(patch_module, pinned_source, tmp_path):
    source = tmp_path / "isolated"
    path = source / patch_module.SOURCE
    path.parent.mkdir(parents=True)
    path.write_bytes(pinned_source)
    before_files = sorted(source.rglob("*"))
    manifest = patch_module.patch_tree(source, check=True)
    assert path.read_bytes() == pinned_source
    assert sorted(source.rglob("*")) == before_files
    assert manifest["before_sha256"] == patch_module.EXPECTED_SHA256
    assert manifest["macro_enabled_by_patch_tool"] is False
    assert manifest["ub_bytes"]["original"] == manifest["ub_bytes"]["patched"] == 184 * 1024
    assert patch_module.patch_tree(source) == manifest
    assert json.loads((source / "running-o-ub-patch.json").read_text()) == manifest
    assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest["after_sha256"]
    diff = (source / "running-o-ub-source.patch").read_text(encoding="utf-8")
    assert "+    TBuf<> thesisRunningOBuff;" in diff
    with pytest.raises(ValueError, match="pinned unmodified"):
        patch_module.patch_tree(source)


def test_existing_evidence_is_not_overwritten(patch_module, pinned_source, tmp_path):
    path = tmp_path / patch_module.SOURCE
    path.parent.mkdir(parents=True)
    path.write_bytes(pinned_source)
    evidence = tmp_path / "running-o-ub-patch.json"
    evidence.write_text("previous attempt")
    with pytest.raises(ValueError, match="refusing to overwrite"):
        patch_module.patch_tree(tmp_path)
    assert path.read_bytes() == pinned_source
    assert evidence.read_text() == "previous attempt"


def test_symlink_cannot_escape_isolated_tree(patch_module, pinned_source, tmp_path):
    source = tmp_path / "isolated"
    path = source / patch_module.SOURCE
    path.parent.mkdir(parents=True)
    outside = tmp_path / "outside.h"
    outside.write_bytes(pinned_source)
    try:
        path.symlink_to(outside)
    except OSError:
        pytest.skip("file symlinks are unavailable for this user")
    with pytest.raises(ValueError, match="escapes"):
        patch_module.patch_tree(source)
    assert outside.read_bytes() == pinned_source


def test_geometry_uses_native_max_s1_and_actual_task_m(patch_module):
    methods = patch_module.METHODS
    for guard in (
        "constInfo.qSeqSize == 4U",
        "constInfo.gSize == 6U",
        "constInfo.headDim == 256U",
        "constInfo.headDimAlign == 256U",
        "constInfo.preLoadNum == 2U",
        "constInfo.s2BaseSize <= 512U",
        "constInfo.subBlockNum == 2U",
        "constInfo.sinkNumber == 0",
        "constInfo.keySinkNumber == 0",
        "!learnableSinkFlag",
        "info.actS1Size == 4U",
        "info.actMBaseSize == 24U",
        "!info.tndIsS2SplitCore",
        "!info.isSinkTensor",
        "startRow == 0U",
        "dealRowCount == mSplitInfo.vecDealM",
        "info.actualSingleProcessSInnerSizeAlign <= 512U",
    ):
        assert guard in methods
    assert "whole" not in methods and "upstream" not in methods
    assert "constInfo.mBaseSize <= 32" not in methods


def test_shrunken_queue_also_bounds_unsupported_fp32_staging(patch_module, pinned_source):
    text = patch_module.transformed(pinned_source).decode()
    assert "if (thesisRunningOEnabled && mSplitSize > 16U)" in text
    assert "mSplitSize = 16U;" in text
    allocation = patch_module.ALLOCATION
    assert allocation.count("BUFFER_SIZE_BYTE_16K") == 2
    assert allocation.count("BUFFER_SIZE_BYTE_32K") == 2
    assert 16 * 512 * 2 == 16384
    assert 16 * 256 * 4 == 16384
    assert 16 * 256 * 2 == 8192


def test_resident_read_requires_valid_same_owner_and_contiguous_s2(patch_module):
    helper = patch_module.STATE_HELPER
    for field in ("bIdx", "n2Idx", "gS1Idx", "actMBaseSize", "vecStartM", "elements"):
        assert f"{field} ==" in helper
    assert "return active && !isLastS2Loop &&" in helper
    assert "loop + 1U == info.loop" in helper
    assert "s2Idx + 1U == info.s2Idx" in helper
    assert "info.isFirstSInnerLoop || SameOwner" in helper
    assert "thesisRunningOState.active && (info.isFirstSInnerLoop || !use)" in patch_module.METHODS


def test_cold_spill_protects_saved_owner_address_and_both_pipelines(patch_module):
    methods = patch_module.METHODS
    spill = methods[methods.index("::ThesisSpillRunningO()") : methods.index("::ThesisPrepareRunningO(")]
    assert spill.index("if (!thesisRunningOState.active)") < spill.index("thesisRunningOBuff.Get")
    assert spill.index("HardEvent::V_MTE3") < spill.index("DataCopy(vec2ResGm[")
    assert "vec2ResGm[thesisRunningOState.workspaceOffset]" in spill
    assert "thesisRunningOState.elements" in spill
    assert spill.index("HardEvent::MTE3_MTE2") > spill.index("DataCopy(vec2ResGm[")
    assert spill.index("HardEvent::MTE3_V") > spill.index("DataCopy(vec2ResGm[")
    assert spill.rindex("thesisRunningOState.active = false;") > spill.index("WaitFlag<HardEvent::MTE3_V>")


def test_resident_store_preserves_mte3_cross_core_handoff(patch_module):
    store = patch_module.STORE_NEW
    resident = store[: store.index("        } else")]
    assert "vec2ResGm" not in resident
    assert resident.index("DataCopy(thesisRunningOBuff.Get") < resident.index("HardEvent::V_MTE3")
    assert resident.index("WaitFlag<HardEvent::V_MTE3>") < resident.index("thesisRunningOState.Remember")
    expected_fallback = "\n".join("    " + line if line else line for line in patch_module.STORE_OLD.splitlines())
    assert expected_fallback in store


def test_softmax_recurrence_final_format_and_cross_core_flags_are_preserved(patch_module, pinned_source):
    old = pinned_source.decode()
    new = patch_module.transformed(pinned_source).decode()
    for token in ("RowMuls<COMPUTE_T>", "RowDivs<COMPUTE_T>", "Add(bmm2ResUb, bmm2ResUb, bmm2ResPreUb"):
        assert old.count(token) == new.count(token)
    marker = "template <typename FIAT>\n__aicore__ inline void FiaBlockVecNonQuant<FIAT>::Bmm2ResCopyOut("
    assert old[old.index(marker) :] == new[new.index(marker) :]
    assert "thesisRunningOState.active = false;" in new[: new.index(marker)]
    assert new.count("defined(THESIS_FIA_RUNNING_O_UB)") == new.count(
        "defined(THESIS_FIA_RUNNING_O_UB) && THESIS_FIA_RUNNING_O_UB && (__CCE_AICORE__ == 220)"
    )


CPP_STUBS = r"""
#include <cassert>
#include <cstdint>
#include <vector>
#define __aicore__
#define THESIS_FIA_RUNNING_O_UB 1
#define __CCE_AICORE__ 220
struct RunInfo {
    bool isFirstSInnerLoop = false, isLastS2Loop = false;
    uint32_t bIdx = 0, n2Idx = 0, gS1Idx = 0, actMBaseSize = 24, loop = 0, s2Idx = 0;
};
"""

CPP_MAIN = r"""
int main() {
    for (uint32_t rows : {16U, 8U}) {
        ThesisRunningOState state;
        std::vector<float> acc(rows * 256), baseline(rows * 256), value(rows * 256);
        for (uint32_t head = 0, loop = 0; head < 12; ++head) {
            for (uint32_t s2 = 0; s2 < 9; ++s2, ++loop) {
                RunInfo info;
                info.bIdx = head / 2; info.n2Idx = head % 2; info.loop = loop; info.s2Idx = s2;
                info.isFirstSInnerLoop = s2 == 0; info.isLastS2Loop = s2 == 8;
                assert(state.CanUse(info, true, 16U - rows, rows * 256));
                for (uint32_t i = 0; i < acc.size(); ++i) {
                    const float partial = float((i + s2) % 17) / 8.0f;
                    const float scale = float((s2 % 3) + 1) / 4.0f;
                    value[i] = partial + (s2 ? acc[i] * scale : 0.0f);
                    baseline[i] = partial + (s2 ? baseline[i] * scale : 0.0f);
                    assert(value[i] == baseline[i]);
                }
                if (info.isLastS2Loop) {
                    state.active = false;
                } else {
                    acc = value;
                    state.Remember(info, 16U - rows, rows * 256, uint64_t(loop % 2) * 65536);
                }
            }
        }
    }
    ThesisRunningOState state;
    RunInfo first; first.isFirstSInnerLoop = true;
    assert(state.CanUse(first, true, 0, 4096));
    RunInfo next; next.loop = 1; next.s2Idx = 1;
    assert(!state.CanUse(next, true, 0, 4096)); // Never read uninitialized UB.
    state.Remember(first, 0, 4096, 123456);
    assert(state.isFirstSInnerLoop && !state.isLastS2Loop);
    assert(state.CanUse(next, true, 0, 4096));
    assert(state.workspaceOffset == 123456 && state.elements == 4096);
    assert(!state.CanUse(next, false, 0, 4096)); // FD/unsupported tasks go to GM.
    for (uint32_t field = 0; field < 8; ++field) {
        RunInfo changed = next;
        uint32_t vecStart = 0, count = 4096;
        switch (field) {
            case 0: ++changed.bIdx; break;
            case 1: ++changed.n2Idx; break;
            case 2: ++changed.gS1Idx; break;
            case 3: ++changed.actMBaseSize; break;
            case 4: ++changed.loop; break;
            case 5: ++changed.s2Idx; break;
            case 6: ++vecStart; break;
            default: ++count; break;
        }
        assert(!state.CanUse(changed, true, vecStart, count));
    }
    state.active = false;
    assert(!state.CanUse(next, true, 0, 4096)); // A final task invalidates the slot.
    first.bIdx = 100;
    assert(state.CanUse(first, true, 0, 4096)); // New owner starts without a UB read.
}
"""


def test_real_cpp_state_two_aiv_recurrence_and_invalid_ownership(patch_module, tmp_path):
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("no local C++ compiler; run this CPU harness on the native build host")
    source = tmp_path / "running_o.cpp"
    binary = tmp_path / "running_o"
    source.write_text(CPP_STUBS + patch_module.STATE_HELPER + CPP_MAIN, encoding="ascii")
    subprocess.run([compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror", str(source), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)
