# SPDX-License-Identifier: Apache-2.0
"""Source/version isolation and compute-order contracts for the MM2 candidate."""

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
        importlib.import_module("aclgraph_fia_sink_mm2_transfer_patch"),
        importlib.import_module("aclgraph_fia_sink_mm2_transfer_build"),
        importlib.import_module("aclgraph_fia_sink_stage2_patch"),
    )


@pytest.fixture
def qualified_source(modules):
    patch, _, stage2 = modules
    repo = Path(__file__).resolve().parents[4]
    snapshot = (
        repo / "benchmarks/results/multi_graph_structural/fia-sink-mm2-transfer-v1" / "raw/screen/cube-before.h.txt"
    )
    if snapshot.is_file():
        qualified = snapshot.read_bytes()
        assert hashlib.sha256(qualified).hexdigest() == patch.EXPECTED_SHA256
        return qualified
    workspace = Path(__file__).resolve().parents[6]
    path = workspace / "outputs/thesis_incremental_plan/pinned_sink_source/source" / patch.SOURCE
    if not path.is_file():
        pytest.skip("the pinned native source is unavailable in this checkout")
    original = path.read_bytes()
    assert hashlib.sha256(original).hexdigest() == stage2.HASHES[stage2.CUBE]
    return stage2.transformed(stage2.CUBE, original.decode("utf-8")).encode("utf-8")


def test_source_drift_and_reapplication_are_rejected(modules, qualified_source):
    patch, _, _ = modules
    assert hashlib.sha256(qualified_source).hexdigest() == patch.EXPECTED_SHA256
    for data in (b"", b"untrusted", qualified_source + b"\n", patch.transformed(qualified_source)):
        with pytest.raises(ValueError, match="pinned, qualified"):
            patch.transformed(data)


def test_only_allocation_and_l1_split_change(modules, qualified_source):
    patch, _, _ = modules
    old = qualified_source.decode()
    new = patch.transformed(qualified_source).decode()
    # The complete MM1 implementation and MM2 compute loop remain byte-identical.
    begin = "__aicore__ inline void FiaBlockCubeNonQuantGqa<FIAT, Config>::ComputeMm1(RunInfo &info)"
    end = "__aicore__ inline void FiaBlockCubeNonQuantGqa<FIAT, Config>::ComputeMm2(const RunInfo &info)"
    assert old[old.index(begin) : old.index(end)] == new[new.index(begin) : new.index(end)]
    marker = "    bool canFullLoadV = (KL1Slices.size() <= L1_V_BUFCNT);"
    assert old[old.index(marker) :] == new[new.index(marker) :]
    assert "const uint32_t K_BASE = K_SPLIT_SIZE;" in new
    assert "kL1.Split(K_BASE)" in new


def test_unsupported_geometry_keeps_safe_split(modules):
    patch, _, _ = modules
    for condition in (
        "IsSameType<Q_T, bfloat16_t>::value",
        "IsSameType<KV_T, bfloat16_t>::value",
        "PAGE_ATTENTION",
        "LAYOUT_T == FIA_LAYOUT::TND",
        "PA_LAYOUT == FIA_LAYOUT::BSH",
        "!CFG::S2_BASICSIZE_IS_1024",
    ):
        assert condition in patch.GUARD
    assert "constInfo.headDim == 256U" in patch.SPLIT
    assert "constInfo.headDimRope == 0U" in patch.SPLIT
    assert "!info.isSinkTensor" in patch.SPLIT
    assert "? 128U : K_SPLIT_SIZE" in patch.SPLIT


def test_dry_run_and_overwrite_protection(modules, qualified_source, tmp_path):
    patch, _, _ = modules
    source = tmp_path / "source"
    kernel = source / patch.SOURCE
    kernel.parent.mkdir(parents=True)
    kernel.write_bytes(qualified_source)
    before = sorted(source.rglob("*"))
    manifest = patch.patch_tree(source, check=True)
    assert sorted(source.rglob("*")) == before
    assert kernel.read_bytes() == qualified_source
    assert manifest["l1_v_bytes"]["total"] == 65536
    assert manifest["l1_total_bytes"] == 464 * 1024
    assert manifest["graph_split_exclusive"] is False
    assert patch.patch_tree(source) == manifest
    assert json.loads((source / "mm2-transfer-patch.json").read_text()) == manifest
    assert hashlib.sha256(kernel.read_bytes()).hexdigest() == manifest["after_sha256"]
    kernel.write_bytes(qualified_source)
    with pytest.raises(ValueError, match="refusing to overwrite"):
        patch.patch_tree(source)
    assert kernel.read_bytes() == qualified_source


@pytest.mark.parametrize("enabled", (0, 1))
def test_compile_definition_and_restricted_binary_scope(modules, enabled):
    patch, builder, _ = modules
    call = "    compile_op(src, origin_func_name, op_info, options, code_channel, '{}', {'valueDepend': {}})"
    changed = builder.wrapper_options(call, enabled)
    assert changed.count(f"-D{patch.MACRO}={enabled}") == 1
    assert "THESIS_FIA_RUNNING_O_UB" not in changed
    assert builder.KEYS == ("104000000010222223", "104000000010322223")
    with pytest.raises(ValueError, match="already patched"):
        builder.wrapper_options(changed, enabled)


@pytest.mark.parametrize("enabled", (True, -1, 2, "1"))
def test_invalid_macro_value_is_rejected(modules, enabled):
    _, builder, _ = modules
    with pytest.raises(ValueError, match="wide-v"):
        builder.wrapper_options("generated wrapper", enabled)
