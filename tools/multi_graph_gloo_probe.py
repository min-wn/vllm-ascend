"""Exercise two-rank routing agreement using CPU Gloo, without vLLM or NPU."""

import argparse
import json
import sys
import types
from datetime import timedelta
from pathlib import Path


def prepare_imports():
    root = Path(__file__).resolve().parents[1]
    sys.path[:] = [path for path in sys.path if path != str(root / "tools")]
    sys.path.insert(0, str(root))
    sys.modules["vllm_ascend.logger"] = types.ModuleType("vllm_ascend.logger")


def worker(rank, rendezvous, output):
    prepare_imports()
    import torch.distributed as distributed

    from vllm_ascend.worker.multi_graph.distributed import RankProposal, agree, torch_gloo_gather

    distributed.init_process_group(
        "gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=20)
    )
    try:
        accepted = agree(torch_gloo_gather(RankProposal(0, "tp2", "same-requests", (2, 2), True)), 2)
        fallback = agree(torch_gloo_gather(RankProposal(1, "tp2", "same-requests", (2, 2), rank == 0)), 2)
        divergent = False
        try:
            agree(torch_gloo_gather(RankProposal(2, "tp2", f"requests-{rank}", (2, 2), True)), 2)
        except RuntimeError:
            divergent = True
        assert accepted == (2, 2) and fallback is None and divergent
        Path(output, f"rank-{rank}.json").write_text(
            json.dumps({"rank": rank, "agreed": accepted, "fallback": fallback, "divergence_rejected": divergent}),
            encoding="utf-8",
        )
    finally:
        distributed.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    prepare_imports()
    import tempfile

    import torch.multiprocessing as multiprocessing

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as directory:
        rendezvous = (Path(directory) / "rendezvous").as_uri()
        multiprocessing.spawn(worker, args=(rendezvous, str(args.output_dir.resolve())), nprocs=2, join=True)


if __name__ == "__main__":
    main()
