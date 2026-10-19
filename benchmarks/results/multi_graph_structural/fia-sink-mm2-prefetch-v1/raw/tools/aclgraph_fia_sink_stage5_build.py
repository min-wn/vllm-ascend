# SPDX-License-Identifier: Apache-2.0
"""Relink a configured isolated stage-five AICPU target and stage a new vendor.

Example (after sourcing the existing Ascend profile, with no NPU process):
  python aclgraph_fia_sink_stage5_build.py --root /workspace/EXPERIMENT \
    --source /workspace/EXPERIMENT/stage5-no-fd/source \
    --baseline-vendor /workspace/EXPERIMENT/opp/packages/vendors/VENDOR \
    --output-dir /workspace/EXPERIMENT/stage5-no-fd/relinked-v3

The existing output library is moved intact into the fresh backup directory so
the configured target must relink it despite its missing object dependency.
Only the target and a new copied vendor tree are written; no system install,
dependency configuration, package rebuild, or NPU workload is performed.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
import traceback
from pathlib import Path

META_DIRECTORY = "attention/fused_infer_attention_score_v2_sink_metadata"
META_NAME = "fused_infer_attention_score_v2_sink_metadata_aicpu.cpp"
SOURCE_RELATIVE = Path("csrc") / META_DIRECTORY / "op_kernel_aicpu" / META_NAME
OBJECT_RELATIVE = (
    Path(META_DIRECTORY)
    / "CMakeFiles/fused_infer_attention_score_v2_sink_metadata_obj.dir/op_kernel_aicpu"
    / (META_NAME + ".o")
)
FLAGS_RELATIVE = Path(META_DIRECTORY) / "CMakeFiles/fused_infer_attention_score_v2_sink_metadata_obj.dir/flags.make"
LIBRARY_NAME = "libtransformer_aicpu_kernels.so"
UNIQUE_LIBRARY = "libthesis_stage5_whole_aicpu.so"
MACRO = "THESIS_NO_FD_WHOLE_ROW"
REPORT_NAME = "stage5-relink-status.json"
BASELINE_OPAPI_SHA256 = "2c1660f393e0e8cc9c8d935929420e3e3b61297163994ade48eec5ac14c9ad40"


def file_record(path):
    stat = path.stat()
    return dict(
        path=str(path),
        bytes=stat.st_size,
        mtime_ns=stat.st_mtime_ns,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )


def _inside(path, root, *, exists=True):
    resolved = Path(path).resolve(strict=exists)
    if not resolved.is_relative_to(root):
        raise ValueError(f"path escapes isolated root: {resolved}")
    return resolved


def preflight(root, source, baseline_vendor, output_dir):
    """Check the preconfigured source, exact macro and fresh output boundary."""
    root = Path(root).resolve(strict=True)
    source = _inside(source, root)
    vendor = _inside(baseline_vendor, root)
    output = _inside(output_dir, root, exists=False)
    if source != root / "stage5-no-fd/source" or not source.is_dir():
        raise ValueError("source must be the existing isolated stage5-no-fd/source tree")
    if output.exists() or output == root or source.is_relative_to(output) or vendor.is_relative_to(output):
        raise ValueError("choose a new output directory that does not contain the source or baseline vendor")
    if output.is_relative_to(source) or output.is_relative_to(vendor):
        raise ValueError("output may not be created inside the source or baseline vendor")
    build = source / "csrc/build"
    cache = (build / "CMakeCache.txt").read_text()
    home = next(
        (line.split("=", 1)[1] for line in cache.splitlines() if line.startswith("CMAKE_HOME_DIRECTORY:INTERNAL=")),
        None,
    )
    if home is None or Path(home).resolve(strict=True) != source / "csrc":
        raise ValueError("CMake cache belongs to a different source tree")
    flags = (build / FLAGS_RELATIVE).read_text()
    if not {"-D" + MACRO, "-D" + MACRO + "=1"}.intersection(flags.split()):
        raise ValueError("metadata object flags do not enable the reviewed whole-row policy")
    metadata_source = _inside(source / SOURCE_RELATIVE, root)
    manifest_path = source / "stage5-no-fd-patch.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("after_sha256") != file_record(metadata_source)["sha256"] or manifest.get("macro") != MACRO:
        raise ValueError("reviewed patch manifest does not match the current metadata source")
    for path in (build / LIBRARY_NAME, build / OBJECT_RELATIVE, vendor / "op_api/lib/libcust_opapi.so"):
        if not _inside(path, root).is_file():
            raise ValueError(f"required configured artifact is not a file: {path}")
    if file_record(vendor / "op_api/lib/libcust_opapi.so")["sha256"] != BASELINE_OPAPI_SHA256:
        raise ValueError("baseline OpAPI differs from the qualified unchanged stage-two library")
    config = vendor / "op_impl/cpu/config/cust_aicpu_kernel.json"
    json.loads(config.read_text())["FusedInferAttentionScoreV2SinkMetadata"]["opInfo"]["kernelSo"]
    for child in vendor.rglob("*"):
        if child.is_symlink():
            raise ValueError("baseline vendor symlinks must be resolved in a separate audited copy before relinking")
    return root, source, vendor, output


def relink(root, source, baseline_vendor, output_dir, *, jobs=4, timeout=300, runner=subprocess.run):
    """Relink only cust_aicpu_kernels, verify provenance, and stage a new copy."""
    root, source, vendor, output = preflight(root, source, baseline_vendor, output_dir)
    if type(jobs) is not int or not 1 <= jobs <= 8 or timeout < 1:
        raise ValueError("jobs must be in 1..8 and timeout must be positive")
    output.mkdir(parents=True)
    target = output / REPORT_NAME
    status = dict(state="running", started_unix=time.time(), npu_work_executed=False, system_install_performed=False)

    def save():
        target.write_text(json.dumps(status, indent=2) + "\n", encoding="ascii")

    build, library = source / "csrc/build", source / "csrc/build" / LIBRARY_NAME
    command = ["cmake", "--build", str(build), "--target", "cust_aicpu_kernels", "--verbose", "-j" + str(jobs)]
    status.update(
        command=command, cwd=str(source), source_patch=json.loads((source / "stage5-no-fd-patch.json").read_text())
    )
    save()
    try:
        inputs = {
            "metadata_source": file_record(source / SOURCE_RELATIVE),
            "metadata_object": file_record(build / OBJECT_RELATIVE),
            "previous_library": file_record(library),
            "compile_flags": file_record(build / FLAGS_RELATIVE),
            "cmake_cache": file_record(build / "CMakeCache.txt"),
        }
        status["before"] = inputs
        backup = output / "backup" / LIBRARY_NAME
        backup.parent.mkdir()
        library.rename(backup)
        status.update(backup=file_record(backup), original_output_absent=not library.exists())
        save()
        if status["backup"]["sha256"] != inputs["previous_library"]["sha256"]:
            raise RuntimeError("archived output library changed before the forced relink")
        log_path = output / "relink.log"
        with log_path.open("xb") as log:
            result = runner(command, cwd=source, stdout=log, stderr=subprocess.STDOUT, timeout=timeout, check=False)
        status.update(build_returncode=result.returncode, link_log=file_record(log_path))
        save()
        if result.returncode:
            raise RuntimeError(f"isolated metadata relink returned {result.returncode}")
        linked = file_record(library)
        obj, cpp = file_record(build / OBJECT_RELATIVE), file_record(source / SOURCE_RELATIVE)
        status.update(after=dict(library=linked, metadata_object=obj, metadata_source=cpp))
        save()
        text = log_path.read_text(errors="replace")
        link_lines = [
            line
            for line in text.splitlines()
            if "-shared" in line and "-o" in line and LIBRARY_NAME in line and META_NAME + ".o" in line
        ]
        if not link_lines:
            raise RuntimeError("verbose log contains no actual metadata-object shared-library link command")
        if linked["sha256"] == inputs["previous_library"]["sha256"]:
            raise RuntimeError("linked library hash is unchanged; refuse to stage an unproven rebuild")
        if obj["mtime_ns"] < cpp["mtime_ns"] or linked["mtime_ns"] < max(obj["mtime_ns"], cpp["mtime_ns"]):
            raise RuntimeError("linked library/object is older than the reviewed source or metadata object")
        if cpp["sha256"] != inputs["metadata_source"]["sha256"]:
            raise RuntimeError("metadata source changed during the bounded build")
        status["verified_link_commands"] = link_lines
        staged = output / "opp/packages/vendors" / vendor.name
        shutil.copytree(vendor, staged)
        staged_library = staged / "op_impl/cpu/aicpu_kernel/impl" / UNIQUE_LIBRARY
        shutil.copy2(library, staged_library)
        config = staged / "op_impl/cpu/config/cust_aicpu_kernel.json"
        config_data = json.loads(config.read_text())
        config_data["FusedInferAttentionScoreV2SinkMetadata"]["opInfo"]["kernelSo"] = UNIQUE_LIBRARY
        config.write_text(json.dumps(config_data, indent=2) + "\n", encoding="ascii")
        if file_record(staged_library)["sha256"] != linked["sha256"]:
            raise RuntimeError("staged unique AICPU library differs from the verified linked output")
        if file_record(staged / "op_api/lib/libcust_opapi.so")["sha256"] != BASELINE_OPAPI_SHA256:
            raise RuntimeError("staged unchanged OpAPI differs from the qualified baseline")
        status.update(
            state="completed",
            returncode=0,
            vendor=str(staged),
            staged_library=file_record(staged_library),
            artifacts={
                str(path.relative_to(staged)): file_record(path) for path in sorted(staged.rglob("*")) if path.is_file()
            },
            finished_unix=time.time(),
        )
        save()
    except BaseException:
        status.update(state="failed", returncode=1, error=traceback.format_exc(), finished_unix=time.time())
        save()
        raise
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "source", "baseline-vendor", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()
    if os.name != "posix" or args.root.resolve(strict=True).parent != Path("/workspace").resolve():
        parser.error("this helper runs only in an existing isolated /workspace experiment")
    status = relink(args.root, args.source, args.baseline_vendor, args.output_dir, jobs=args.jobs, timeout=args.timeout)
    print(json.dumps(dict(state=status["state"], vendor=status["vendor"], status=str(args.output_dir / REPORT_NAME))))


if __name__ == "__main__":
    main()
