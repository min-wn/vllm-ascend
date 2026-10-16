"""Select one seeded decode batch and compare replay at its default graph size.

The engine config and dispatcher select the padding capacity. The child benchmark
captures a controlled decoder graph, not an online serving engine. Query length
is one; split branches use their exact shapes. No batch resampling is performed.
"""

import argparse
import hashlib
import json
import os
import random
import statistics
import subprocess
import sys
from pathlib import Path


def select_case(args):
    from vllm.engine.arg_utils import EngineArgs
    from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher

    batch = random.Random(args.seed).randint(args.batch_min, args.batch_max)
    config = EngineArgs(
        model=args.model,
        dtype="bfloat16",
        max_num_seqs=args.max_num_seqs,
        max_model_len=max(2048, args.kv_length + 1),
    ).create_engine_config()
    dispatcher = CudagraphDispatcher(config)
    dispatcher.initialize_cudagraph_keys(config.compilation_config.cudagraph_mode)
    mode, descriptor = dispatcher.dispatch(batch, uniform_decode=True)
    if str(mode) != "FULL":
        raise ValueError(f"sampled batch {batch} does not dispatch to FULL: {mode}; do not resample")
    return dict(
        seed=args.seed,
        sample_range=[args.batch_min, args.batch_max],
        batch=batch,
        query=1,
        model=args.model,
        max_num_seqs=args.max_num_seqs,
        max_model_len=config.model_config.max_model_len,
        max_num_batched_tokens=config.scheduler_config.max_num_batched_tokens,
        capture_sizes=config.compilation_config.cudagraph_capture_sizes,
        capture_mode=str(config.compilation_config.cudagraph_mode),
        selected_mode=str(mode),
        selected_tokens=descriptor.num_tokens,
        selected_requests=descriptor.num_reqs,
        scope="default_capacity_controlled_decoder_replay_not_online_engine",
        split_shapes="exact_branches_not_default_capture_grid",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--seed", type=int, default=20261016)
    parser.add_argument("--batch-min", type=int, default=64)
    parser.add_argument("--batch-max", type=int, default=256)
    parser.add_argument("--max-num-seqs", type=int, default=256)
    parser.add_argument("--kv-length", type=int, default=512)
    parser.add_argument("--cores", nargs="+", type=int, default=[0, 10, 20, 10, 0])
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--select-only", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not 2 <= args.batch_min <= args.batch_max <= args.max_num_seqs:
        parser.error("require 2 <= batch-min <= batch-max <= max-num-seqs")
    if min(args.kv_length, args.rounds, args.repeats) < 1 or min(args.cores) < 0:
        parser.error("length/counts must be positive and core limits nonnegative")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any(args.output_dir.iterdir()):
        parser.error("output-dir must be empty to preserve previous experiment evidence")
    selection = select_case(args)
    script = Path(__file__).with_name("multi_graph_model_scaling.py")
    selection["benchmark_sha256_at_start"] = hashlib.sha256(script.read_bytes()).hexdigest()
    selection["cores_sequence"] = args.cores
    (args.output_dir / "selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    print(json.dumps(selection, indent=2), flush=True)
    if args.select_only:
        return
    summary = []
    for index, cores in enumerate(args.cores):
        destination = (args.output_dir / f"run-{index}-cores-{cores}.json").resolve()
        command = [
            sys.executable,
            str(script),
            "--model",
            args.model,
            "--batch",
            str(selection["batch"]),
            "--capacity",
            str(selection["selected_requests"]),
            "--query",
            "1",
            "--kv-length",
            str(args.kv_length),
            "--attention-cores",
            str(cores),
            "--rounds",
            str(args.rounds),
            "--repeats",
            str(args.repeats),
            "--output",
            str(destination),
        ]
        destination.with_suffix(".command.json").write_text(json.dumps(command, indent=2), encoding="utf-8")
        with destination.with_suffix(".log").open("w", encoding="utf-8") as log:
            subprocess.run(
                command,
                cwd=script.parent.parent,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=dict(os.environ, VLLM_BATCH_INVARIANT="0"),
                check=True,
                timeout=600,
            )
        if hashlib.sha256(script.read_bytes()).hexdigest() != selection["benchmark_sha256_at_start"]:
            raise RuntimeError("benchmark source changed during experiment")
        report = json.loads(destination.read_text(encoding="utf-8"))
        if report["status"] != "passed":
            raise RuntimeError(f"benchmark failed: {destination}")
        timing = {
            mode: statistics.median(
                row["wall_summary_us"]["p50"] for row in report["measurements"] if row["mode"] == mode
            )
            / 1000
            for mode in ("padding", "split_parallel")
        }
        summary.append(
            dict(
                file=destination.name,
                cores=cores,
                **timing,
                gain_percent=100 * (1 - timing["split_parallel"] / timing["padding"]),
            )
        )
        (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary[-1]), flush=True)


if __name__ == "__main__":
    main()
