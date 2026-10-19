# SPDX-License-Identifier: Apache-2.0
"""Compile one isolated BF16 running-output consumer without a system install.

The configured stage-two generated kernel tree and parameter file are copied
or read without mutation. Both qualified metadata producers receive the same
new consumer. This tool compiles a binary; it performs no NPU execution.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import aclgraph_fia_sink_running_o_patch as patch

BINARY = "FusedInferAttentionScoreV2Sink_cacc23924ba5ed25533f5fc5005e0673"
OP = "fused_infer_attention_score_v2_sink"
WRAPPER = "FusedInferAttentionScoreV2Sink.py"
OPAPI_HASH = "2c1660f393e0e8cc9c8d935929420e3e3b61297163994ade48eec5ac14c9ad40"
PARAM_HASH = "9384517184de3478d50eec0461cd78a5b1e1aeee2ccc86a6ae927b70cd474880"
POLICY_LIBRARIES = {
    "upstream": ("libfia_v2_sink_aicpu_kernels.so", "796983827bc6326a250009a2aacc810c44f65545ec2ae56dab517a5ec4e7111b"),
    "whole": ("libthesis_stage5_whole_aicpu.so", "1c4ab9fcf2b7d2935be3e60f2a215fb65dc85859bc6fe99fe5271b4c4784ea90"),
}
REPORT = "running-o-build.json"
MAX_TILING_KEY = (1 << 64) - 1


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inside(path, root, *, exists=True):
    path = Path(path).resolve(strict=exists)
    if not path.is_relative_to(root):
        raise ValueError("path escapes the isolated experiment")
    return path


def preflight(root, upstream, whole, output):
    root = Path(root).resolve(strict=True)
    output = inside(output, root, exists=False)
    if output.exists() or output == root:
        raise ValueError("output must be a fresh directory below the isolated root")
    build = inside(root / "source/csrc/build", root)
    vendors = {"upstream": inside(upstream, root), "whole": inside(whole, root)}
    for policy, vendor in vendors.items():
        if output.is_relative_to(vendor) or vendor.is_relative_to(output):
            raise ValueError("output overlaps an existing vendor")
        name, digest = POLICY_LIBRARIES[policy]
        config = json.loads((vendor / "op_impl/cpu/config/cust_aicpu_kernel.json").read_text())
        if config["FusedInferAttentionScoreV2SinkMetadata"]["opInfo"]["kernelSo"] != name:
            raise ValueError("metadata dispatcher selects another producer")
        if sha256(vendor / "op_impl/cpu/aicpu_kernel/impl" / name) != digest:
            raise ValueError("metadata producer differs from the qualified build")
        if sha256(vendor / "op_api/lib/libcust_opapi.so") != OPAPI_HASH:
            raise ValueError("OpAPI differs from the qualified common library")
    source = inside(build / "binary/ascend910b/src", root)
    if output.is_relative_to(source) or source.is_relative_to(output) or output.is_relative_to(build):
        raise ValueError("output overlaps the configured baseline build")
    param = inside(build / "binary/ascend910b/gen" / (BINARY + "_param.json"), root)
    data = json.loads(param.read_text())
    if data["op_list"][0]["bin_filename"] != BINARY or data["op_list"][0]["inputs"][0]["dtype"] != "bfloat16":
        raise ValueError("only the exact qualified BF16 parameter file is supported")
    if sha256(param) != PARAM_HASH:
        raise ValueError("BF16 compiler parameters differ from the qualified byte snapshot")
    for path in (source / OP / WRAPPER, source / OP / "op_kernel/fia_block_vec_nonquant_sink.h"):
        inside(path, root)
    return root, build, source, param, vendors, output


def wrapper_options(text, resident):
    anchor = "    compile_op(src, origin_func_name, op_info, options, code_channel, '{}', {'valueDepend': {}})"
    if text.count(anchor) != 1 or "THESIS_FIA_RUNNING_O_UB" in text:
        raise ValueError("generated wrapper compile anchor differs or was already patched")
    return text.replace(anchor, f'    options.append("-DTHESIS_FIA_RUNNING_O_UB={int(resident)}")\n' + anchor)


def canonical_key(value):
    if type(value) is int:
        value = str(value)
    if (
        type(value) is not str
        or not 1 <= len(value) <= 20
        or any(char < "0" or char > "9" for char in value)
        or (len(value) > 1 and value[0] == "0")
    ):
        raise ValueError("tiling keys must be canonical unsigned decimal uint64 values")
    if int(value) > MAX_TILING_KEY:
        raise ValueError("tiling key exceeds uint64")
    return value


def parse_tiling_keys(value):
    if type(value) is not str:
        raise ValueError("tiling keys must be a comma-separated decimal string")
    keys = tuple(canonical_key(key) for key in value.split(","))
    if len(set(keys)) != len(keys):
        raise ValueError("tiling keys must be unique")
    return tuple(sorted(keys, key=int))


def checked_tiling_keys(keys):
    if keys is None:
        return None
    if type(keys) is not tuple or any(type(key) is not str for key in keys):
        raise ValueError("tiling keys must be a canonical sorted tuple of decimal strings")
    normalized = parse_tiling_keys(",".join(keys))
    if keys != normalized:
        raise ValueError("tiling key tuple must be numerically sorted")
    return keys


def binary_qualification(metadata, selected_keys):
    selected_keys = checked_tiling_keys(selected_keys)
    kernels = metadata.get("kernelList") if isinstance(metadata, dict) else None
    if not isinstance(kernels, list) or not kernels:
        raise ValueError("compiled metadata must contain a nonempty kernelList")
    if any(not isinstance(kernel, dict) or "tilingKey" not in kernel for kernel in kernels):
        raise ValueError("every compiled kernel must identify its tilingKey")
    actual_keys = tuple(sorted({canonical_key(kernel["tilingKey"]) for kernel in kernels}, key=int))
    if selected_keys is not None and actual_keys != selected_keys:
        raise ValueError("compiled kernelList keys differ from the requested restricted tiling keys")
    return dict(
        scope="restricted_tiling_keys" if selected_keys is not None else "all_parameter_tiling_keys",
        requested_tiling_keys=list(selected_keys) if selected_keys is not None else None,
        actual_tiling_keys=list(actual_keys),
        kernel_entries=len(kernels),
        exact_selected_keys_verified=selected_keys is not None,
        arbitrary_shape_fallback_qualified=False,
        npu_runtime_qualified=False,
        limitation="Only listed tiling keys exist; shape, plan, lifetime and runtime qualification remain required",
    )


def compile_command(wrapper, param, output, tiling_keys=None):
    tiling_keys = checked_tiling_keys(tiling_keys)
    command = [
        "opc",
        str(wrapper),
        "--main_func=" + OP,
        "--input_param=" + str(param),
        "--soc_version=Ascend910B1",
        "--output=" + str(output),
        "--impl_mode=high_performance,optional",
        "--simplified_key_mode=0",
        "--op_mode=dynamic",
    ]
    if tiling_keys is not None:
        command.append("--tiling_key=" + ",".join(tiling_keys))
    return command


def build(
    args,
    *,
    runner=subprocess.run,
    source_patch=patch,
    wrapper_transform=wrapper_options,
    report_name=REPORT,
    feature_name="resident",
):
    root, configured, source, param, vendors, output = preflight(
        args.root, args.upstream_vendor, args.whole_vendor, args.output_dir
    )
    if type(args.resident) is not int or args.resident not in (0, 1) or args.timeout < 1:
        raise ValueError("resident must be 0/1 and timeout must be positive")
    tiling_keys = checked_tiling_keys(getattr(args, "tiling_keys", None))
    if Path(report_name).name != report_name or feature_name not in ("resident", "wide_v", "prefetch"):
        raise ValueError("unsupported experiment report or feature")
    output.mkdir(parents=True)
    report = output / report_name
    status = dict(
        state="running",
        started_unix=time.time(),
        npu_work_executed=False,
        system_install_performed=False,
        baseline_parameter=dict(path=str(param), sha256=sha256(param)),
        requested_tiling_keys=list(tiling_keys) if tiling_keys is not None else None,
        binary_scope="restricted_tiling_keys" if tiling_keys is not None else "all_parameter_tiling_keys",
        arbitrary_shape_fallback_qualified=False,
        source_scope="Copied generated AICore tree; original source/build/vendors remain unchanged",
    )
    status[feature_name] = bool(args.resident)

    def save():
        report.write_text(json.dumps(status, indent=2) + "\n", encoding="ascii")

    save()
    try:
        copied = output / "source/csrc/attention"
        shutil.copytree(source, copied, ignore=shutil.ignore_patterns("__pycache__", "*.done"))
        status["source_before"] = {
            str(path.relative_to(copied)): sha256(path) for path in sorted(copied.rglob("*")) if path.is_file()
        }
        save()
        command = [sys.executable, str(Path(source_patch.__file__).resolve()), "--source", str(output / "source")]
        status["patch_command"] = command
        save()
        with (output / "patch.log").open("xb") as log:
            result = runner(command, stdout=log, stderr=subprocess.STDOUT, timeout=60, check=False)
        if result.returncode:
            raise RuntimeError("isolated source patch failed")
        wrapper = copied / OP / WRAPPER
        wrapper.write_text(wrapper_transform(wrapper.read_text(), args.resident), encoding="utf-8")
        status["source_after"] = {
            str(path.relative_to(copied)): sha256(path) for path in sorted(copied.rglob("*")) if path.is_file()
        }
        changed = [
            name for name, digest in status["source_before"].items() if status["source_after"].get(name) != digest
        ]
        kernel_relative = source_patch.SOURCE.relative_to("csrc/attention").as_posix()
        expected = [OP + "/" + WRAPPER, kernel_relative]
        if sorted(changed) != sorted(expected):
            raise RuntimeError("unexpected copied source mutation")
        status["changed_source_files"] = changed
        binary_dir = output / "binary"
        binary_dir.mkdir()
        command = compile_command(wrapper, param, binary_dir, tiling_keys)
        env = dict(
            os.environ,
            # The old configure-time custom directory is no longer populated.
            # Use the qualified vendor's retained tiling registration library.
            ASCEND_CUSTOM_OPP_PATH=str(vendors["upstream"]),
            HI_PYTHON="python3",
            TILINGKEY_PAR_COMPILE="1",
            BIN_FILENAME_HASHED="1",
            ASCEND_GLOBAL_LOG_LEVEL="3",
            ASCEND_SLOG_PRINT_TO_STDOUT="1",
        )
        status.update(command=command, stage="compiling")
        save()
        with (output / "compile.log").open("xb") as log:
            result = runner(
                command, cwd=output, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout, check=False
            )
        status["compile_returncode"] = result.returncode
        save()
        if result.returncode:
            raise RuntimeError("isolated AICore compile failed")
        artifacts = [binary_dir / (BINARY + suffix) for suffix in (".o", ".json")]
        for path in artifacts:
            if not path.is_file() or path.stat().st_size == 0:
                raise RuntimeError("compiler produced no complete binary/json pair")
        metadata = json.loads(artifacts[1].read_text())
        status["binary_metadata"] = metadata
        status["binary_qualification"] = binary_qualification(metadata, tiling_keys)
        status["artifacts"] = {path.name: dict(bytes=path.stat().st_size, sha256=sha256(path)) for path in artifacts}
        status["vendors"] = {}
        for policy, original in vendors.items():
            destination = output / ("vendor-" + policy)
            shutil.copytree(original, destination)
            kernel_dir = destination / "op_impl/ai_core/tbe/kernel/ascend910b" / OP
            for path in artifacts:
                shutil.copyfile(path, kernel_dir / path.name)
            expected_changes = {"op_impl/ai_core/tbe/kernel/ascend910b/" + OP + "/" + path.name for path in artifacts}
            actual_changes = []
            for path in sorted(destination.rglob("*")):
                if path.is_file():
                    relative = path.relative_to(destination)
                    if sha256(path) != sha256(original / relative):
                        actual_changes.append(relative.as_posix())
            if not set(actual_changes).issubset(expected_changes):
                raise RuntimeError("staged vendor must change only the selected AICore binary/json pair")
            object_relative = "op_impl/ai_core/tbe/kernel/ascend910b/" + OP + "/" + BINARY + ".o"
            if args.resident and object_relative not in actual_changes:
                raise RuntimeError("resident consumer object is identical to the original binary")
            status["vendors"][policy] = dict(
                path=str(destination),
                changed_files=actual_changes,
                binary_scope=status["binary_qualification"]["scope"],
                available_tiling_keys=status["binary_qualification"]["actual_tiling_keys"],
                arbitrary_shape_fallback_qualified=False,
            )
        status.update(state="completed", stage="completed", returncode=0, finished_unix=time.time())
    except BaseException:
        status.update(state="failed", returncode=1, error=traceback.format_exc(), finished_unix=time.time())
        save()
        raise
    save()
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "upstream-vendor", "whole-vendor", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--resident", type=int, choices=(0, 1), default=1)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument(
        "--tiling-keys",
        type=parse_tiling_keys,
        default=None,
        help="Compile only these unique decimal uint64 keys (comma separated); default compiles the full parameter set",
    )
    args = parser.parse_args()
    if os.name != "posix" or args.root.resolve().parent != Path("/workspace").resolve():
        parser.error("use an isolated /workspace experiment after sourcing the existing Ascend profile")
    status = build(args)
    print(json.dumps(dict(state=status["state"], report=str(args.output_dir / REPORT))))


if __name__ == "__main__":
    main()
