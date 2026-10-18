# SPDX-License-Identifier: Apache-2.0
"""Build/load the isolated Q4 block prototype without initializing an NPU."""

import argparse
import hashlib
import importlib.util
import os
import subprocess
from pathlib import Path


def build(directory):
    import torch

    source = Path(__file__).resolve().parent
    destination = Path(directory).resolve()
    fingerprint = hashlib.sha256()
    for path in sorted((*source.glob("*.cpp"), source / "CMakeLists.txt")):
        fingerprint.update(path.name.encode("ascii"))
        fingerprint.update(path.read_bytes())
    digest = fingerprint.hexdigest()
    artifact = destination / "gdn_q4_log.so"
    stamp = destination / "source.sha256"
    if artifact.exists() and stamp.exists() and stamp.read_text().strip() == digest:
        return artifact
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("use a fresh build directory when sources change")
    destination.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "cmake",
            "-S",
            str(source),
            "-B",
            str(destination),
            f"-DCMAKE_PREFIX_PATH={torch.utils.cmake_prefix_path}",
            f"-DPython_EXECUTABLE={os.sys.executable}",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DSOC_VERSION=Ascend910B3",
        ],
        check=True,
    )
    subprocess.run(["cmake", "--build", str(destination), "--parallel", "2"], check=True)
    stamp.write_text(digest + "\n", encoding="ascii")
    return artifact


def load(path):
    spec = importlib.util.spec_from_file_location("gdn_q4_log", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", required=True)
    args = parser.parse_args()
    print(build(args.build_dir), flush=True)
