# SPDX-License-Identifier: Apache-2.0
"""Build an isolated MM2 transfer candidate using the qualified build inputs."""

import argparse
import json
import os
from pathlib import Path

import aclgraph_fia_sink_mm2_transfer_patch as patch
import aclgraph_fia_sink_running_o_build as base

REPORT = "mm2-transfer-build.json"
KEYS = ("104000000010222223", "104000000010322223")


def wrapper_options(text, enabled):
    if type(enabled) is not int or enabled not in (0, 1):
        raise ValueError("wide-v must be 0 or 1")
    anchor = "    compile_op(src, origin_func_name, op_info, options, code_channel, '{}', {'valueDepend': {}})"
    if text.count(anchor) != 1 or patch.MACRO in text:
        raise ValueError("generated wrapper changed or was already patched")
    return text.replace(anchor, f'    options.append("-D{patch.MACRO}={enabled}")\n' + anchor)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "upstream-vendor", "whole-vendor", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--wide-v", type=int, choices=(0, 1), required=True)
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()
    if os.name != "posix" or args.root.resolve().parent != Path("/workspace").resolve():
        parser.error("use an isolated /workspace experiment after sourcing the existing Ascend profile")
    args.resident, args.tiling_keys = args.wide_v, KEYS
    status = base.build(
        args, source_patch=patch, wrapper_transform=wrapper_options, report_name=REPORT, feature_name="wide_v"
    )
    print(json.dumps(dict(state=status["state"], report=str(args.output_dir / REPORT))))


if __name__ == "__main__":
    main()
