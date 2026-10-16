"""Run fixed-query NPU qualification independently of the serving test fixtures."""

import sys
from pathlib import Path


class ReplayOptions:
    def pytest_addoption(self, parser):
        parser.addoption("--replay-model", default="Qwen/Qwen2.5-0.5B-Instruct")


def main():
    repo = Path(__file__).resolve().parents[1]
    tool_dir = str(Path(__file__).resolve().parent)
    sys.path[:] = [entry for entry in sys.path if entry != tool_dir]
    sys.path.insert(0, str(repo))
    import pytest

    test = repo / "tests/e2e/pull_request/one_card/test_multi_graph_replay.py"
    return pytest.main([f"--confcutdir={test.parent}", str(test), "-q", *sys.argv[1:]], plugins=[ReplayOptions()])


if __name__ == "__main__":
    raise SystemExit(main())
