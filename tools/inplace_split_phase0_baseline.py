#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional


def _run(cmd: list[str], cwd: Optional[Path] = None) -> Optional[str]:
    try:
        return subprocess.check_output(
            cmd,
            cwd=str(cwd) if cwd else None,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None


def _git_commit(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    return _run(["git", "rev-parse", "HEAD"], cwd=path)


def _package_path(module_name: str) -> Optional[Path]:
    try:
        module = __import__(module_name)
        module_file = getattr(module, "__file__", None)
        if module_file is None:
            return None
        return Path(module_file).resolve()
    except Exception:
        return None


def _torch_info() -> dict[str, Any]:
    info: dict[str, Any] = {}
    try:
        import torch

        info["torch"] = getattr(torch, "__version__", None)
        info["cann"] = getattr(getattr(torch, "version", None), "cann", None)
        npu = getattr(torch, "npu", None)
        if npu is not None and hasattr(npu, "is_available"):
            info["npu_available"] = bool(npu.is_available())
            if info["npu_available"] and hasattr(npu, "get_device_name"):
                info["device"] = npu.get_device_name(0)
            else:
                info["device"] = None
        else:
            info["npu_available"] = False
            info["device"] = None
    except Exception as exc:
        info["torch_error"] = str(exc)
    try:
        import torch_npu

        info["torch_npu"] = getattr(torch_npu, "__version__", None)
    except Exception as exc:
        info["torch_npu_error"] = str(exc)
    return info


def _selected_env() -> dict[str, Optional[str]]:
    keys = [
        "VLLM_USE_V1",
        "VLLM_WORKER_MULTIPROC_METHOD",
        "ASCEND_RT_VISIBLE_DEVICES",
        "VLLM_ASCEND_ACLGRAPH_DIAG",
        "VLLM_ASCEND_PERF_STATS_FILE",
        "VLLM_ASCEND_SPLIT_INPLACE_DEBUG",
        "VLLM_ASCEND_SPLIT_INPLACE_DEBUG_FILE",
    ]
    return {key: os.environ.get(key) for key in keys}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Capture phase-0 environment snapshot for inplace split.")
    parser.add_argument(
        "--output",
        default="/tmp/vllm_ascend_inplace_phase0/baseline_env.json",
        help="Path to write the JSON snapshot.")
    parser.add_argument("--model",
                        default=os.environ.get("MODEL"),
                        help="Model identifier used for the baseline run.")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    vllm_file = _package_path("vllm")
    vllm_repo = None
    if vllm_file is not None:
        for parent in [vllm_file.parent, *vllm_file.parents]:
            if (parent / ".git").exists():
                vllm_repo = parent
                break

    data: dict[str, Any] = {
        "vllm_ascend_git_commit": _git_commit(repo),
        "vllm_git_commit": _git_commit(vllm_repo) if vllm_repo else None,
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "model": args.model,
        "compilation_config": {},
        "split_batch_config": {},
        "env": _selected_env(),
    }
    data.update(_torch_info())

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print(str(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
