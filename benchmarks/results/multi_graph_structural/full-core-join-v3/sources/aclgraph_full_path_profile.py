# SPDX-License-Identifier: Apache-2.0
"""Collect isolated full-model replay timelines after unprofiled diagnostics."""

import hashlib
from contextlib import nullcontext


def profile_indices(step_count):
    if isinstance(step_count, bool) or not isinstance(step_count, int) or step_count < 2:
        raise ValueError("at least two continued steps required")
    return sorted({0, 1, step_count - 1})


def profile_replays(
    args,
    torch,
    dist,
    registry,
    steps,
    selection,
    entries,
    prepare,
    replay,
    *,
    restore,
    cpu_group,
    modes=("deferred_tile_ready",),
    select_mode=None,
    selected_indices=None,
):
    import torch_npu

    records = []
    selected = profile_indices(len(steps)) if selected_indices is None else selected_indices
    for mode in modes:
        restore()
        if select_mode is None:
            registry.mode = mode
        else:
            select_mode(mode)
        for index, step in enumerate(steps):
            capacity = selection["dispatched_tokens"][index]
            dist.barrier(group=cpu_group)
            context = (
                torch_npu.profiler.profile(
                    activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
                    record_shapes=True,
                )
                if index in selected
                else nullcontext()
            )
            with context as prof:
                with torch.profiler.record_function("full_path/prepare"):
                    registry.prepare_step(step["step"], step["request_ids"], step["accepted_from_previous"])
                    frame = prepare(step, capacity)
                with torch.profiler.record_function("full_path/replay_and_true_fia_update"):
                    replay(entries[mode, capacity], frame)
                with torch.profiler.record_function("full_path/join"):
                    torch.npu.synchronize()
                with torch.profiler.record_function("full_path/finish"):
                    registry.finish_step(step["step"])
            if index in selected:
                suffix = "" if modes == ("deferred_tile_ready",) else f".{mode}"
                path = args.output.with_name(f"{args.output.stem}{suffix}.step{index}.rank{dist.get_rank()}.trace.json")
                prof.export_chrome_trace(str(path))
                records.append(
                    dict(
                        mode=mode,
                        step=index,
                        batch=step["batch"],
                        capacity=capacity,
                        path=str(path),
                        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    )
                )
                print(
                    f"rank={dist.get_rank()} full-model profile mode={mode} step={index} tokens={capacity}", flush=True
                )
    return records
