# SPDX-License-Identifier: Apache-2.0
"""Save three frozen layer-4 recurrence sites; no quality or timing experiment."""

import argparse
import hashlib
from pathlib import Path

from multi_graph_gdn_q4_full_loader import run_full_model
from multi_graph_gdn_q4_full_probe import deterministic_token_ids
from multi_graph_gdn_q4_full_probe import experiment as base_experiment
from multi_graph_gdn_ready_chain_protocol import workload

SITES = {"original": (7, 31), "low": (7,)}


def extract(args, torch, dist, bundle, factory, registry, forward, vocabulary, report, *, checkpoint, schedule=None):
    from multi_graph_gdn_q4_log_backend import Q4LogBackend

    if schedule is not None:
        raise ValueError("no split schedule in this diagnostic")
    registry.mode = "deferred_tile_ready"
    torch.npu.matmul.allow_hf32 = False
    torch.set_float32_matmul_precision("highest")
    target = next(entry for entry in bundle.gdn_entries if entry.index == 4)
    backend = registry.backends[target.state.handle]
    original = Q4LogBackend.__call__
    current_step = None
    captured = []

    def intercept(self, query, key, value, initial_state, **kwargs):
        if self is backend and current_step in SITES[args.load]:
            torch.npu.synchronize()
            state = self.state
            payload = dict(
                query=query.cpu(), key=key.cpu(), value=value.cpu(),
                beta=kwargs["beta"].cpu(), gate=kwargs["g"].cpu(),
                base=state.base.cpu(), logs=state.logs.cpu(), valid=state.valid.cpu(),
                sequence=kwargs["actual_seq_lengths"].cpu(),
                ids=torch.index_select(state.device_owner_lookup, 0, kwargs["ssm_state_indices"].long()).cpu(),
                accepted=kwargs["num_accepted_tokens"].cpu(),
            )
            path = args.output.parent / f"{args.load}.step{current_step}.rank{dist.get_rank()}.pt"
            if path.exists():
                raise FileExistsError(path)
            torch.save(payload, path)
            captured.append(dict(
                step=current_step, layer=4, path=str(path),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                tensors={name: dict(shape=list(tensor.shape), dtype=str(tensor.dtype)) for name, tensor in payload.items()},
                state_source="independent compact continuation after coherent real-model prefill",
            ))
            print(f"saved {path}", flush=True)
        return original(self, query, key, value, initial_state, **kwargs)

    Q4LogBackend.__call__ = intercept
    try:
        for index, step in enumerate(bundle.trace.steps[:max(SITES[args.load]) + 1]):
            current_step = index
            capacity = args.selection["dispatched_tokens"][index]
            frame = factory.decode(step, capacity)
            cpu = torch.zeros(capacity, dtype=torch.int64)
            cpu[:step["batch"] * 4] = deterministic_token_ids(torch, step["batch"] * 4, vocabulary, step["input_seed"])
            registry.prepare_step(step["step"], step["request_ids"], step["accepted_from_previous"])
            forward(frame, cpu.to(bundle.device), registry=registry)
            torch.npu.synchronize()
            registry.finish_step(step["step"])
            print(f"rank={dist.get_rank()} extract load={args.load} step={index}", flush=True)
    finally:
        Q4LogBackend.__call__ = original
    report.update(status="saved_frozen_recurrence_sites_no_performance_claim", sites=captured)
    checkpoint()


def experiment(args, trace, runner, model, config):
    return base_experiment(args, trace, runner, model, config, replay_runner=extract)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/models/Qwen3.5-27B-original")
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--load", choices=tuple(SITES), required=True)
    parser.add_argument("--output", type=Path, required=True)
    for name in ("extension", "products-extension", "deferred-extension", "tile-extension"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".rank0.json").exists():
        parser.error("new output required")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.tensor_parallel_size, args.phase, args.prefix_limit, args.dense_nz = 2, "replay", None, True
    args.split_boundaries, args.packed_join = None, False
    trace, _, _ = workload(args.trace, args.load)
    run_full_model(args, trace, experiment=experiment)


if __name__ == "__main__":
    main()
