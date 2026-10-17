"""Untimed-input causal probe for GateUp/HCCL competition, not decoder speed."""

import statistics


def probe_cases(rows, boundary):
    if not 0 < boundary < rows:
        raise ValueError("the split must have two nonempty partitions")
    return (
        ("full_alone", 0, rows, False),
        ("left_alone", 0, boundary, False),
        ("right_alone", boundary, rows, False),
        ("full_with_comm", 0, rows, True),
        ("left_with_comm", 0, boundary, True),
        ("right_with_comm", boundary, rows, True),
        ("comm_alone", 0, 0, True),
    )


def measure_contention(normalized, weight, local_down, boundary, group, cpu_group, repeats=12, trace_path=None):
    """Use actual selected-layer inputs and an actual TP reduction per probe.

    Each candidate's root and communication streams fork from one ready event.
    Both branches complete inside external replay timing. Captured timing events
    are deliberately avoided: they returned stale timestamps on this runtime.
    Captured restoration precedes that fork. No FIA is run or updated here.
    """
    import torch
    import torch.distributed as dist
    import torch.nn.functional as functional

    rows = normalized.shape[0]
    payload_source = local_down[:boundary].contiguous().clone()
    main = torch.npu.current_stream()
    entries, samples = {}, []
    for name, start, stop, communicate in probe_cases(rows, boundary):
        root, comm = torch.npu.Stream(), torch.npu.Stream()
        fork, done = torch.npu.Event(), torch.npu.Event()
        payload = torch.empty_like(payload_source)
        root.wait_stream(main)

        def operation(
            payload=payload,
            fork=fork,
            done=done,
            root=root,
            comm=comm,
            communicate=communicate,
            start=start,
            stop=stop,
        ):
            payload.copy_(payload_source)
            fork.record(root)
            if communicate:
                with torch.npu.stream(comm):
                    comm.wait_event(fork)
                    reduced = group.all_reduce(payload)
                    done.record(comm)
            else:
                reduced = None
            if stop > start:
                projected = functional.linear(normalized[start:stop], weight)
            else:
                projected = None
            if communicate:
                root.wait_event(done)
            return projected, reduced

        with torch.npu.stream(root):
            for _ in range(3):
                operation()
        torch.npu.synchronize()
        dist.barrier(group=cpu_group)
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, pool=torch.npu.graph_pool_handle(), stream=root):
            outputs = operation()
        with torch.npu.stream(root):
            graph.replay()
        torch.npu.synchronize()
        if stop > start:
            expected = functional.linear(normalized[start:stop], weight)
            torch.testing.assert_close(outputs[0], expected, rtol=0, atol=0)
        if communicate:
            expected = group.all_reduce(payload_source.clone())
            torch.testing.assert_close(outputs[1], expected, rtol=0, atol=0)
        entries[name] = dict(
            graph=graph,
            root=root,
            comm=comm,
            outputs=outputs,
            payload=payload,
            events=(fork, done),
            gemm=stop > start,
            communicate=communicate,
        )

    begin, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
    for round_id in range(3):
        names = tuple(entries) if round_id % 2 == 0 else tuple(reversed(entries))
        for name in names:
            entry = entries[name]
            root = entry["root"]
            for _ in range(3):
                with torch.npu.stream(root):
                    entry["graph"].replay()
                torch.npu.synchronize()
            durations = dict(total_us=[])
            for _ in range(repeats):
                dist.barrier(group=cpu_group)
                with torch.npu.stream(root):
                    begin.record(root)
                    entry["graph"].replay()
                    end.record(root)
                end.synchronize()
                durations["total_us"].append(begin.elapsed_time(end) * 1000)
            samples.append(dict(case=name, round=round_id, **durations))

    local = dict(rank=dist.get_rank(), samples=samples)
    ranks = [None] * dist.get_world_size()
    dist.all_gather_object(ranks, local, group=cpu_group)
    summaries = {}
    for name in entries:
        summary = {}
        for metric in ("total_us",):
            values = []
            for round_id in range(3):
                data = [next(s for s in r["samples"] if s["case"] == name and s["round"] == round_id) for r in ranks]
                if data[0][metric]:
                    values.append(statistics.median(max(row[metric][i] for row in data) for i in range(repeats)))
            summary[metric] = statistics.median(values) if values else None
            summary[f"rounds_{metric}"] = values
        summaries[name] = summary
    if trace_path is not None:
        import torch_npu

        with torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU]
        ) as profiler:
            for name, entry in entries.items():
                for _ in range(2):
                    dist.barrier(group=cpu_group)
                    with torch.profiler.record_function(f"contention.{name}"):
                        with torch.npu.stream(entry["root"]):
                            entry["graph"].replay()
                        torch.npu.synchronize()
        profiler.export_chrome_trace(str(trace_path))
    return dict(
        status="passed",
        scope="actual_gate_up_inputs_and_tp_collective_only_not_decoder_performance",
        payload_shape=list(payload_source.shape),
        payload_bytes=payload_source.numel() * payload_source.element_size(),
        real_all_reduces_per_replay={name: int(entry["communicate"]) for name, entry in entries.items()},
        restore_inside_total_timer=True,
        timing="external_events_complete_replay_only",
        internal_timing="profiler_separate_from_latency_samples" if trace_path else None,
        summaries=summaries,
        ranks=ranks,
    )
