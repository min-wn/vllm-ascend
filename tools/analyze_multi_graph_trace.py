"""Analyze an Ascend Chrome trace without importing vLLM or torch-npu."""

import argparse
import importlib.util
import json
import sys
from pathlib import Path


def main():
    tool_dir = str(Path(__file__).resolve().parent)
    sys.path[:] = [entry for entry in sys.path if entry != tool_dir]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lane-streams", type=int, nargs=2, default=())
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[1] / "vllm_ascend/worker/multi_graph/trace_analysis.py"
    spec = importlib.util.spec_from_file_location("trace_analysis", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.analyze_trace(json.loads(args.trace.read_text(encoding="utf-8")), args.lane_streams)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Decode markers: {result['decode_markers']}; bounded windows: {len(result['steps'])}")


if __name__ == "__main__":
    main()
