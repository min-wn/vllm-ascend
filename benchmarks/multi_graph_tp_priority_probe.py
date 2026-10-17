"""Isolate request-branch collective priority; this is not decoder performance."""

import statistics
import time


def priority_orders(rows, pair):
    """Return fixed rank-consistent orders for two legal request partitions."""
    if len(pair) != 2 or min(pair) <= 0 or sum(pair) != rows:
        raise ValueError("pair must contain two positive capacities adding to the whole bucket")
    return {
        "whole_bucket": ("O", "Down"),
        "fifo": ("O_A", "O_B", "Down_A", "Down_B"),
        "depth_priority": ("O_A", "Down_A", "O_B", "Down_B"),
        "fifo_barrier": ("O_A", "O_B", "Down_A", "Down_B"),
    }


def measure_priority_probe(
    local_output,
    residual,
    post_norm,
    act_fn,
    gate_up_weight,
    down_weight,
    next_norm,
    group,
    cpu_group,
    *,
    pair=(200, 200),
    rounds=3,
    repeats=6,
):
    """Change only collective queue order or the next-Norm branch barrier.

    Inputs are actual per-rank local attention O projections, before their TP
    sum, and the corresponding residual. All O projections are already ready;
    this deliberately excludes attention producer readiness and FIA updates.
    ``next_norm`` must be the actual adjacent layer's input Norm, or an
    explicitly labelled proxy in the caller's enclosing report.

    Each split mode has identical half-shaped MLP arithmetic. A separate stream
    runs next Norm A as soon as Down A is reduced, overlapping the remaining B
    work. The barrier control delays that same Norm until both Down reductions
    finish. Root joins both Norm outputs and residuals before external timing
    ends. Source restoration, every reduction, waits and joins are timed.
    """
    import torch
    import torch.distributed as dist
    import torch.nn.functional as functional
    from multi_graph_numerical_assessment import error_metrics

    rows = local_output.shape[0]
    orders = priority_orders(rows, pair)
    if residual.shape != local_output.shape:
        raise ValueError("local O projection and residual must have matching shapes")
    if rounds < 1 or repeats < 1:
        raise ValueError("rounds and repeats must be positive")
    boundary = pair[0]
    main = torch.npu.current_stream()
    output_source, residual_source = local_output.clone(), residual.clone()
    entries, checks = {}, []

    class ProbeEntry:
        def __init__(self, name):
            self.name = name
            self.root, self.comm, self.norm_stream = (torch.npu.Stream() for _ in range(3))
            self.root.wait_stream(main)
            self.working = torch.empty_like(output_source)
            self.residual_working = torch.empty_like(residual_source)
            self.joined_hidden = torch.empty_like(output_source)
            self.joined_residual = torch.empty_like(residual_source)
            self.ready = [torch.npu.Event() for _ in range(4)]
            self.done = [torch.npu.Event() for _ in range(4)]
            self.norm_done = torch.npu.Event()
            self.retained = []

        def reduce(self, value, index):
            self.ready[index].record(self.root)
            with torch.npu.stream(self.comm):
                self.comm.wait_event(self.ready[index])
                reduced = group.all_reduce(value)
                self.done[index].record(self.comm)
            # Retain both input and returned storage: the communicator is not
            # assumed to return an alias of its input.
            self.retained.extend((value, reduced))
            return reduced

        def mlp(self, value, residual_value):
            normalized, next_residual = post_norm(value, residual_value)
            if isinstance(gate_up_weight, tuple):
                intermediate = torch.cat(
                    [act_fn(functional.linear(normalized, weight)) for weight in gate_up_weight], dim=-1
                )
            else:
                intermediate = act_fn(functional.linear(normalized, gate_up_weight))
            projected = functional.linear(intermediate, down_weight)
            self.retained.extend((normalized, next_residual, intermediate, projected))
            return projected, next_residual

        def operation(self):
            self.retained = []
            self.working.copy_(output_source)
            self.residual_working.copy_(residual_source)
            if self.name == "whole_bucket":
                reduced = self.reduce(self.working, 0)
                self.root.wait_event(self.done[0])
                projected, next_residual = self.mlp(reduced, self.residual_working)
                reduced_down = self.reduce(projected, 2)
                self.root.wait_event(self.done[2])
                hidden, final_residual = next_norm(reduced_down, next_residual)
                self.joined_hidden.copy_(hidden)
                self.joined_residual.copy_(final_residual)
                self.retained.extend((hidden, final_residual))
                return self.joined_hidden, self.joined_residual

            first = self.reduce(self.working[:boundary], 0)
            if self.name != "depth_priority":
                second = self.reduce(self.working[boundary:], 1)
            self.root.wait_event(self.done[0])
            first_down, first_residual = self.mlp(first, self.residual_working[:boundary])
            first_reduced = self.reduce(first_down, 2)
            if self.name == "depth_priority":
                second = self.reduce(self.working[boundary:], 1)
            self.root.wait_event(self.done[1])
            second_down, second_residual = self.mlp(second, self.residual_working[boundary:])
            second_reduced = self.reduce(second_down, 3)

            # This stream can advance A while root is still producing Down B.
            # The host submits it later, but captured replay follows the event
            # dependency rather than host submission time.
            with torch.npu.stream(self.norm_stream):
                wait_index = 3 if self.name == "fifo_barrier" else 2
                self.norm_stream.wait_event(self.done[wait_index])
                first_hidden, first_final_residual = next_norm(first_reduced, first_residual)
                self.norm_stream.wait_event(self.done[3])
                second_hidden, second_final_residual = next_norm(second_reduced, second_residual)
                self.norm_done.record(self.norm_stream)
            self.root.wait_event(self.norm_done)
            self.joined_hidden[:boundary].copy_(first_hidden)
            self.joined_hidden[boundary:].copy_(second_hidden)
            self.joined_residual[:boundary].copy_(first_final_residual)
            self.joined_residual[boundary:].copy_(second_final_residual)
            self.retained.extend((first_hidden, first_final_residual, second_hidden, second_final_residual))
            return self.joined_hidden, self.joined_residual

    for name in orders:
        entry = ProbeEntry(name)
        with torch.npu.stream(entry.root):
            for _ in range(3):
                entry.operation()
            eager = tuple(value.clone() for value in entry.operation())
        torch.npu.synchronize()
        dist.barrier(group=cpu_group)
        entry.graph = torch.npu.NPUGraph()
        with torch.npu.graph(entry.graph, pool=torch.npu.graph_pool_handle(), stream=entry.root):
            entry.outputs = entry.operation()
        with torch.npu.stream(entry.root):
            entry.graph.replay()
        torch.npu.synchronize()
        for field, actual, expected in zip(("hidden", "residual"), entry.outputs, eager, strict=True):
            # Also reject nonfinite values, even if a backend were to produce
            # matching nonfinite eager and captured results.
            error_metrics(actual, expected)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            checks.append(dict(case=name, field=field, eager_replay_bitwise=True, finite=True))
        entry.snapshot = tuple(value.clone() for value in entry.outputs)
        torch.npu.synchronize()
        entries[name] = entry

    same_shape_checks, whole_bucket_diagnostics = [], {}
    for name, entry in entries.items():
        if name == "whole_bucket":
            continue
        diagnostics = {}
        for field, actual, whole, fifo in zip(
            ("hidden", "residual"),
            entry.snapshot,
            entries["whole_bucket"].snapshot,
            entries["fifo"].snapshot,
            strict=True,
        ):
            diagnostics[field] = error_metrics(actual, whole)
            torch.testing.assert_close(actual, fifo, rtol=0, atol=0)
            same_shape_checks.append(dict(case=name, reference="fifo", field=field, bitwise=True))
        whole_bucket_diagnostics[name] = diagnostics

    begin, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
    samples = []
    for round_id in range(rounds):
        names = tuple(entries) if round_id % 2 == 0 else tuple(reversed(entries))
        for name in names:
            entry = entries[name]
            for _ in range(3):
                with torch.npu.stream(entry.root):
                    entry.graph.replay()
                torch.npu.synchronize()
            device_us, wall_us = [], []
            for _ in range(repeats):
                dist.barrier(group=cpu_group)
                started = time.perf_counter_ns()
                with torch.npu.stream(entry.root):
                    begin.record(entry.root)
                    entry.graph.replay()
                    end.record(entry.root)
                end.synchronize()
                wall_us.append((time.perf_counter_ns() - started) / 1000)
                device_us.append(begin.elapsed_time(end) * 1000)
            samples.append(dict(case=name, round=round_id, device_us=device_us, wall_us=wall_us))

    local = dict(
        rank=dist.get_rank(),
        samples=samples,
        checks=checks,
        same_shape_checks=same_shape_checks,
        whole_bucket_diagnostics=whole_bucket_diagnostics,
    )
    ranks = [None] * dist.get_world_size()
    dist.all_gather_object(ranks, local, group=cpu_group)
    summaries = {}
    for name in entries:
        summary = {}
        for metric in ("device_us", "wall_us"):
            round_medians = []
            for round_id in range(rounds):
                data = [
                    next(s for s in rank["samples"] if s["case"] == name and s["round"] == round_id) for rank in ranks
                ]
                round_medians.append(
                    statistics.median(max(row[metric][index] for row in data) for index in range(repeats))
                )
            summary[metric] = statistics.median(round_medians)
            summary[f"rounds_{metric}"] = round_medians
        summaries[name] = summary
    return dict(
        status="completed_with_numerical_diagnostics",
        scope="actual_local_O_through_MLP_and_next_input_norm_not_decoder_performance",
        local_O_already_ready=True,
        next_norm_source="caller_must_identify_adjacent_layer_or_proxy",
        pair=list(pair),
        input_shape=list(local_output.shape),
        collective_orders=orders,
        real_all_reduces_per_replay={name: len(order) for name, order in orders.items()},
        next_norm_policy={
            "whole_bucket": "after_whole_down",
            "fifo": "A_after_Down_A_and_B_after_Down_B",
            "depth_priority": "A_after_Down_A_and_B_after_Down_B",
            "fifo_barrier": "both_after_Down_B",
        },
        all_restoration_collectives_joins_and_output_copies_timed=True,
        timing="graph_external_device_events_and_host_wall_time_slowest_rank_per_sample",
        summaries=summaries,
        ranks=ranks,
    )
