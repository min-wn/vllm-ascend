"""Build the isolated Ascend C prototype; importing this file launches no NPU work."""

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
    for path in sorted((*source.glob("*.cpp"), *source.glob("*.h"), source / "CMakeLists.txt")):
        fingerprint.update(path.name.encode("ascii"))
        fingerprint.update(path.read_bytes())
    digest = fingerprint.hexdigest()
    artifact = destination / "gdn_compact_state.so"
    stamp = destination / "source.sha256"
    if artifact.exists() and stamp.exists() and stamp.read_text().strip() == digest:
        return artifact
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("use a fresh build directory when sources change; legacy CANN preprocessing is not reusable")
    destination.mkdir(parents=True, exist_ok=True)
    command = [
        "cmake",
        "-S",
        str(source),
        "-B",
        str(destination),
        f"-DCMAKE_PREFIX_PATH={torch.utils.cmake_prefix_path}",
        f"-DPython_EXECUTABLE={os.sys.executable}",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DSOC_VERSION=Ascend910B3",
    ]
    subprocess.run(command, check=True)
    subprocess.run(["cmake", "--build", str(destination), "--parallel", "2"], check=True)
    stamp.write_text(digest + "\n", encoding="ascii")
    return artifact


def load(path):
    spec = importlib.util.spec_from_file_location("gdn_compact_state", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", required=True)
    args = parser.parse_args()
    print(build(args.build_dir), flush=True)
