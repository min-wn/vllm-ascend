"""Run isolated multi-graph contracts with CPU PyTorch and without vLLM/NPU."""

import sys
import types
from pathlib import Path


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    tool_dir = str(Path(__file__).resolve().parent)
    sys.path[:] = [entry for entry in sys.path if entry != tool_dir]
    sys.path.insert(0, str(repo))
    import pytest

    sys.modules["vllm_ascend.logger"] = types.ModuleType("vllm_ascend.logger")
    test_root = repo / "tests" / "ut" / "worker" / "multi_graph"
    return pytest.main([f"--confcutdir={test_root}", str(test_root), "-q", *sys.argv[1:]])


if __name__ == "__main__":
    raise SystemExit(main())
