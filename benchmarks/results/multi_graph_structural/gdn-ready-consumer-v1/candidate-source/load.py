# SPDX-License-Identifier: Apache-2.0
"""Build isolated A2 mixed-core handoff; compilation initializes no NPU."""

import argparse
import hashlib
import importlib.util
import os
import shutil
import subprocess
from pathlib import Path


def build(directory):
    import torch

    source = Path(__file__).resolve().parent
    destination = Path(directory).resolve()
    fingerprint = hashlib.sha256()
    for path in sorted((*source.glob("*.cpp"), *source.glob("*.h"), source / "CMakeLists.txt", source / "load.py")):
        fingerprint.update(path.name.encode("ascii"))
        fingerprint.update(path.read_bytes())
    digest = fingerprint.hexdigest()
    artifact = destination / "gdn_q4_tile_pipeline.so"
    stamp = destination / "source.sha256"
    if artifact.exists() and stamp.exists() and stamp.read_text().strip() == digest:
        return artifact
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("use a fresh build directory when sources change")
    destination.mkdir(parents=True, exist_ok=True)
    staging = destination / "source"
    staging.mkdir()
    for path in source.iterdir():
        if path.is_file():
            shutil.copy2(path, staging / path.name)
    cann = Path(os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest"))
    generator = destination / "q4_products_tiling_generator"
    subprocess.run(
        [
            "g++",
            "-std=c++17",
            "-O2",
            str(source / "tiling_generator.cpp"),
            f"-I{cann / 'include'}",
            f"-I{cann / 'include/ascendc/host_api'}",
            f"-L{cann / 'lib64'}",
            f"-Wl,-rpath,{cann / 'lib64'}",
            "-ltiling_api",
            "-lregister",
            "-lplatform",
            "-lunified_dlog",
            "-lc_sec",
            "-lm",
            "-ldl",
            "-o",
            str(generator),
        ],
        check=True,
    )
    with (staging / "q4_products_tiling.h").open("w", encoding="ascii") as output:
        subprocess.run([str(generator)], stdout=output, check=True)
        subprocess.run([str(generator), "fp32"], stdout=output, check=True)
    subprocess.run(
        [
            "cmake",
            "-S",
            str(staging),
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
    spec = importlib.util.spec_from_file_location("gdn_q4_tile_pipeline", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", required=True)
    args = parser.parse_args()
    print(build(args.build_dir), flush=True)
