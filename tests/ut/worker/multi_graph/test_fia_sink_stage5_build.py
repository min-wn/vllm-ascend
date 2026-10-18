# SPDX-License-Identifier: Apache-2.0
"""CPU-only relink regressions for the missing CMake object dependency."""

import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def builder():
    path = Path(__file__).resolve().parents[4] / "benchmarks/aclgraph_fia_sink_stage5_build.py"
    spec = importlib.util.spec_from_file_location("stage5_builder_cpu", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write(path, contents):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents)
    return path


@pytest.fixture
def configured(builder, tmp_path_factory, monkeypatch):
    # Keep fake build paths below Windows MAX_PATH; production paths retain
    # their pinned CMake layout in the helper.
    monkeypatch.setattr(builder, "SOURCE_RELATIVE", Path("csrc") / builder.META_NAME)
    monkeypatch.setattr(builder, "OBJECT_RELATIVE", Path(builder.META_NAME + ".o"))
    monkeypatch.setattr(builder, "FLAGS_RELATIVE", Path("flags.make"))
    root = tmp_path_factory.mktemp("p5") / "experiment"
    source = root / "stage5-no-fd/source"
    build = source / "csrc/build"
    cpp = write(source / builder.SOURCE_RELATIVE, "reviewed whole-row source")
    write(
        source / "stage5-no-fd-patch.json",
        json.dumps(dict(after_sha256=builder.file_record(cpp)["sha256"], macro=builder.MACRO)),
    )
    write(build / "CMakeCache.txt", "CMAKE_HOME_DIRECTORY:INTERNAL=" + str(source / "csrc") + "\n")
    write(build / builder.FLAGS_RELATIVE, "CXX_FLAGS = -O2 -D" + builder.MACRO + "\n")
    obj = write(build / builder.OBJECT_RELATIVE, "reviewed object")
    library = write(build / builder.LIBRARY_NAME, "old draft library")
    vendor = root / "opp/packages/vendors/baseline"
    write(vendor / "op_api/lib/libcust_opapi.so", "qualified unchanged opapi")
    monkeypatch.setattr(
        builder, "BASELINE_OPAPI_SHA256", builder.file_record(vendor / "op_api/lib/libcust_opapi.so")["sha256"]
    )
    write(
        vendor / "op_impl/cpu/config/cust_aicpu_kernel.json",
        json.dumps({"FusedInferAttentionScoreV2SinkMetadata": {"opInfo": {"kernelSo": "old.so"}}}),
    )
    (vendor / "op_impl/cpu/aicpu_kernel/impl").mkdir(parents=True, exist_ok=True)
    write(vendor / "op_impl/cpu/aicpu_kernel/impl/old.so", "original baseline kernel")
    os.utime(cpp, ns=(100, 100))
    os.utime(obj, ns=(200, 200))
    os.utime(library, ns=(150, 150))
    return SimpleNamespace(
        root=root,
        source=source,
        build=build,
        vendor=vendor,
        output=root / "stage5-no-fd/relinked-v3",
        cpp=cpp,
        obj=obj,
        library=library,
    )


def fake_link(builder, configured, *, text="new reviewed library", timestamp=300, log=True, returncode=0):
    def run(command, **kwargs):
        assert command == [
            "cmake",
            "--build",
            str(configured.build),
            "--target",
            "cust_aicpu_kernels",
            "--verbose",
            "-j4",
        ]
        assert kwargs["cwd"] == configured.source
        assert not configured.library.exists()
        assert (configured.output / "backup" / builder.LIBRARY_NAME).read_text() == "old draft library"
        if returncode == 0:
            configured.library.write_text(text)
            os.utime(configured.library, ns=(timestamp, timestamp))
        line = (
            "compiler -shared " + str(configured.obj) + " -o " + str(configured.library)
            if log
            else "target already built"
        )
        kwargs["stdout"].write((line + "\n").encode())
        return SimpleNamespace(returncode=returncode)

    return run


def test_missing_output_forces_relink_and_preserves_old_vendor(builder, configured):
    original_config = (configured.vendor / "op_impl/cpu/config/cust_aicpu_kernel.json").read_bytes()
    status = builder.relink(
        configured.root, configured.source, configured.vendor, configured.output, runner=fake_link(builder, configured)
    )
    assert status["state"] == "completed"
    assert status["original_output_absent"] is True
    assert status["npu_work_executed"] is False and status["system_install_performed"] is False
    assert status["backup"]["sha256"] == status["before"]["previous_library"]["sha256"]
    staged = Path(status["vendor"])
    assert (staged / "op_impl/cpu/aicpu_kernel/impl" / builder.UNIQUE_LIBRARY).read_text() == "new reviewed library"
    assert (configured.vendor / "op_impl/cpu/config/cust_aicpu_kernel.json").read_bytes() == original_config
    assert (configured.vendor / "op_impl/cpu/aicpu_kernel/impl/old.so").read_text() == "original baseline kernel"
    assert status["verified_link_commands"]


@pytest.mark.parametrize("mode", ["same_hash", "old_timestamp", "no_link_log", "compiler_failure"])
def test_unproven_relink_is_not_staged(builder, configured, mode):
    options = {
        "same_hash": dict(text="old draft library"),
        "old_timestamp": dict(timestamp=100),
        "no_link_log": dict(log=False),
        "compiler_failure": dict(returncode=2),
    }[mode]
    with pytest.raises(RuntimeError):
        builder.relink(
            configured.root,
            configured.source,
            configured.vendor,
            configured.output,
            runner=fake_link(builder, configured, **options),
        )
    status = json.loads((configured.output / builder.REPORT_NAME).read_text())
    assert status["state"] == "failed"
    assert not (configured.output / "opp").exists()
    assert (configured.output / "backup" / builder.LIBRARY_NAME).read_text() == "old draft library"


def test_wrong_macro_and_source_manifest_are_rejected_before_backup(builder, configured):
    write(configured.build / builder.FLAGS_RELATIVE, "CXX_FLAGS = -O2\n")
    with pytest.raises(ValueError, match="flags"):
        builder.preflight(configured.root, configured.source, configured.vendor, configured.output)
    assert configured.library.exists() and not configured.output.exists()
    write(configured.build / builder.FLAGS_RELATIVE, "CXX_FLAGS = -D" + builder.MACRO)
    configured.cpp.write_text("unexpected source edit")
    with pytest.raises(ValueError, match="manifest"):
        builder.preflight(configured.root, configured.source, configured.vendor, configured.output)


def test_source_and_new_output_boundaries_are_checked(builder, configured, tmp_path):
    for output in (tmp_path / "outside", configured.source / "new", configured.vendor / "new", configured.root):
        with pytest.raises(ValueError):
            builder.preflight(configured.root, configured.source, configured.vendor, output)
    configured.output.mkdir()
    with pytest.raises(ValueError, match="new output"):
        builder.preflight(configured.root, configured.source, configured.vendor, configured.output)
