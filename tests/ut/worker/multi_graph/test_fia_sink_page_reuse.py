# SPDX-License-Identifier: Apache-2.0
"""Source identity, patch isolation and bounded qualification tests."""

import importlib
from pathlib import Path

import pytest


@pytest.fixture
def patch(monkeypatch):
    repo = Path(__file__).resolve().parents[4]
    monkeypatch.syspath_prepend(str(repo / "benchmarks"))
    return importlib.import_module("aclgraph_fia_sink_page_reuse_patch")


@pytest.fixture
def source():
    return (
        Path(__file__).resolve().parents[4]
        / "benchmarks/results/multi_graph_structural/fia-sink-mm2-prefetch-v1/raw/screen/cube-before.h.txt"
    ).read_bytes()


def test_roundtrip_only_helper_call_and_local_state_changed(patch, source):
    text = patch.transformed(source).decode()
    for new, old in (
        (patch.HELPER, ""),
        (patch.NEW_COPY, patch.OLD_COPY),
        (patch.LOCALS, ""),
        (patch.GUARD, ""),
        ("\n" + patch.DECLARATION, ""),
    ):
        assert text.count(new) == 1
        text = text.replace(new, old, 1)
    assert text == source.decode()
    assert "constInfo.s2BaseSize == 512U" in patch.GATE
    assert "info.actMBaseSize <= D256_MM2_QUERY_ROWS" in patch.GATE
    assert "static" not in patch.LOCALS
    assert "WaitFlag" not in patch.HELPER and "SetFlag" not in patch.HELPER


def test_source_and_evidence_protection(patch, source, tmp_path):
    for data in (b"", source + b"\n", patch.transformed(source)):
        with pytest.raises(ValueError, match="pinned"):
            patch.transformed(data)
    path = tmp_path / patch.SOURCE
    path.parent.mkdir(parents=True)
    path.write_bytes(source)
    dry = patch.patch_tree(tmp_path, check=True)
    assert path.read_bytes() == source
    assert dry["kv_data_bytes_saved"] == 0 and not dry["graph_split_exclusive"]
    assert patch.patch_tree(tmp_path) == dry
    path.write_bytes(source)
    with pytest.raises(ValueError, match="overwrite"):
        patch.patch_tree(tmp_path)
    assert path.read_bytes() == source


@pytest.mark.parametrize("value", [True, -1, 2, "1"])
def test_invalid_build_macro(patch, value):
    build = importlib.import_module("aclgraph_fia_sink_page_reuse_build")
    with pytest.raises(ValueError, match="page_reuse"):
        build.wrapper_options("wrapper", value)


@pytest.mark.parametrize("value", [0, 1])
def test_build_keys_and_reapplication(patch, value):
    build = importlib.import_module("aclgraph_fia_sink_page_reuse_build")
    anchor = "    compile_op(src, origin_func_name, op_info, options, code_channel, '{}', {'valueDepend': {}})"
    changed = build.wrapper_options(anchor, value)
    assert changed.count(f"-D{patch.MACRO}={value}") == 1
    assert build.KEYS == ("104000000010222223", "104000000010322223")
    with pytest.raises(ValueError, match="patched"):
        build.wrapper_options(changed, value)
