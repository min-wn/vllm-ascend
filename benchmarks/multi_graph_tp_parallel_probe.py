"""Causal probes for request-branch parallelism; not decoder performance."""

import statistics
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class ProbeCase:
    name: str
    family: str
    boundary: int | None = None
    order: tuple[int, int] = (0, 1)
    overlap: bool = False


def probe_cases(rows, pairs=((200, 200), (272, 128))):
    """Plan only existing capture capacities, without changing request identity."""
    if rows <= 0:
        raise ValueError("rows must be positive")
    if not pairs:
        raise ValueError("at least one capacity pair is required")
    cases = [ProbeCase("down_full_bucket", "down"), ProbeCase("norm_full_bucket", "norm")]
    seen = set()
    for pair in pairs:
        if len(pair) != 2 or min(pair) <= 0 or sum(pair) != rows:
            raise ValueError("each pair must contain two positive capacities summing to rows")
        if tuple(pair) in seen:
            raise ValueError("capacity pairs must be unique")
        seen.add(tuple(pair))
        boundary = pair[0]
        prefix = f"{pair[0]}_{pair[1]}"
        orders = ((0, 1),) if pair[0] == pair[1] else ((0, 1), (1, 0))
        for order in orders:
            suffix = "forward" if order == (0, 1) else "reverse"
            for overlap in (False, True):
                mode = "overlap" if overlap else "barrier"
                cases.append(ProbeCase(f"down_{prefix}_{suffix}_{mode}", "down", boundary, order, overlap))
        for overlap in (False, True):
            mode = "overlap" if overlap else "barrier"
            cases.append(ProbeCase(f"norm_{prefix}_{mode}", "norm", boundary, (0, 1), overlap))
    return tuple(cases)


def branch_slices(rows, boundary, order=(0, 1)):
    if not 0 < boundary < rows or set(order) != {0, 1} or len(order) != 2:
        raise ValueError("two nonempty branches must each appear exactly once")
    branches = ((0, boundary), (boundary, rows))
    return tuple(branches[index] for index in order)


def paired_comparisons(cases):
    """Controls share arithmetic and row slices; only dependency/order changes."""
    comparisons = []
    names = {case.name for case in cases}
    for case in cases:
        if case.boundary is None:
            continue
        if case.overlap:
            comparisons.append((case.name.replace("overlap", "barrier"), case.name, "overlap"))
            comparisons.append((f"{case.family}_full_bucket", case.name, "versus_full_bucket"))
        if case.family == "down" and case.order == (1, 0) and case.overlap:
            forward = case.name.replace("reverse", "forward")
            if forward in names:
                comparisons.append((forward, case.name, "branch_order"))
    return tuple(comparisons)


def measure_parallel_probe(
    source_reduced,
    residual,
    local_output,
    post_norm,
    act_fn,
    gate_up_weight,
    down_weight,
    group,
    cpu_group,
    *,
    pairs=((200, 200), (272, 128)),
    rounds=3,
    repeats=6,
):
    """Use real selected-layer weights, O outputs and attention residual.

    Down family: post-Norm + whole GateUp/activation + branch Down/reductions.
    Norm family: branch O reductions + early post-Norm + whole MLP/reduction.
    The two families have different timed endpoints. All joins and output
    reductions are included. Source creation and attention are outside timing.
    The first and second branches preserve the same request rows when reversed.
    """
    import torch
    import torch.distributed as dist
    import torch.nn.functional as functional
    from multi_graph_numerical_assessment import error_metrics

    if rounds <= 0 or repeats <= 0:
        raise ValueError("rounds and repeats must be positive")
    if source_reduced.ndim != 2 or source_reduced.shape != residual.shape or local_output.shape != residual.shape:
        raise ValueError("source, residual and local output must have the same rank-two shape")
    if isinstance(gate_up_weight, tuple):
        raise ValueError("the probe requires one whole GateUp weight")
    if (
        gate_up_weight.ndim != 2
        or down_weight.ndim != 2
        or gate_up_weight.shape[1] != source_reduced.shape[1]
        or gate_up_weight.shape[0] != 2 * down_weight.shape[1]
        or down_weight.shape[0] != source_reduced.shape[1]
    ):
        raise ValueError("only a bias-free, ordinary GateUp/Down MLP is supported")
    rows = source_reduced.shape[0]
    cases = probe_cases(rows, pairs)
    entries, checks, diagnostics = {}, [], []
    main = torch.npu.current_stream()

    class Entry:
        def __init__(self, case):
            self.case = case
            self.root, self.comm = torch.npu.Stream(), torch.npu.Stream()
            self.root.wait_stream(main)
            self.ready = [torch.npu.Event(), torch.npu.Event()]
            self.done = [torch.npu.Event(), torch.npu.Event()]
            self.working = torch.empty_like(source_reduced)
            self.residual = torch.empty_like(residual)
            self.local = torch.empty_like(local_output)
            self.hidden = torch.empty_like(source_reduced)
            self.next_residual = torch.empty_like(residual)
            self.normalized = torch.empty_like(source_reduced)
            self.graph = torch.npu.NPUGraph()

        def reduce(self, value, index):
            self.ready[index].record(self.root)
            with torch.npu.stream(self.comm):
                self.comm.wait_event(self.ready[index])
                output = group.all_reduce(value)
                self.done[index].record(self.comm)
            return output

        def operation(self):
            self.working.copy_(source_reduced)
            self.residual.copy_(residual)
            case = self.case
            if case.family == "norm":
                self.local.copy_(local_output)
            if case.boundary is None:
                value = group.all_reduce(self.local) if case.family == "norm" else self.working
                normalized, next_residual = post_norm(value, self.residual)
                intermediate = act_fn(functional.linear(normalized, gate_up_weight))
                output = group.all_reduce(functional.linear(intermediate, down_weight))
                self.hidden.copy_(output)
                self.next_residual.copy_(next_residual)
                return self.hidden, self.next_residual
            slices = branch_slices(rows, case.boundary, case.order)
            if case.family == "down":
                normalized, next_residual = post_norm(self.working, self.residual)
                intermediate = act_fn(functional.linear(normalized, gate_up_weight))
                local = [None, None]
                reduced = [None, None]
                for index, (start, stop) in enumerate(slices):
                    local[index] = functional.linear(intermediate[start:stop], down_weight)
                    if case.overlap:
                        reduced[index] = self.reduce(local[index], index)
                if not case.overlap:
                    for index in range(2):
                        reduced[index] = self.reduce(local[index], index)
                for index, (start, stop) in enumerate(slices):
                    self.root.wait_event(self.done[index])
                    self.hidden[start:stop].copy_(reduced[index])
                self.next_residual.copy_(next_residual)
            else:
                reduced = [self.reduce(self.local[start:stop], index) for index, (start, stop) in enumerate(slices)]
                if not case.overlap:
                    self.root.wait_event(self.done[1])
                for index, (start, stop) in enumerate(slices):
                    self.root.wait_event(self.done[index])
                    normalized, next_residual = post_norm(reduced[index], self.residual[start:stop])
                    self.normalized[start:stop].copy_(normalized)
                    self.next_residual[start:stop].copy_(next_residual)
                intermediate = act_fn(functional.linear(self.normalized, gate_up_weight))
                output = group.all_reduce(functional.linear(intermediate, down_weight))
                self.hidden.copy_(output)
            return self.hidden, self.next_residual

    for case in cases:
        entry = Entry(case)
        with torch.npu.stream(entry.root):
            for _ in range(3):
                entry.operation()
            expected = tuple(output.clone() for output in entry.operation())
        torch.npu.synchronize()
        dist.barrier(group=cpu_group)
        with torch.npu.graph(entry.graph, pool=torch.npu.graph_pool_handle(), stream=entry.root):
            entry.outputs = entry.operation()
        with torch.npu.stream(entry.root):
            entry.graph.replay()
        torch.npu.synchronize()
        for field, actual, wanted in zip(("hidden", "residual"), entry.outputs, expected, strict=True):
            torch.testing.assert_close(actual, wanted, rtol=0, atol=0)
            checks.append(f"{case.name}/eager_graph/{field}")
        entries[case.name] = entry

    for control, candidate, mechanism in paired_comparisons(cases):
        reference, actual = entries[control], entries[candidate]
        for field, value, wanted in zip(("hidden", "residual"), actual.outputs, reference.outputs, strict=True):
            if mechanism != "versus_full_bucket":
                torch.testing.assert_close(value, wanted, rtol=0, atol=0)
                checks.append(f"{candidate}/{mechanism}/{field}")
            else:
                diagnostics.append(dict(case=candidate, reference=control, field=field, **error_metrics(value, wanted)))

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
                started = time.perf_counter()
                with torch.npu.stream(entry.root):
                    begin.record(entry.root)
                    entry.graph.replay()
                    end.record(entry.root)
                end.synchronize()
                wall_us.append((time.perf_counter() - started) * 1e6)
                device_us.append(begin.elapsed_time(end) * 1000)
            samples.append(dict(case=name, round=round_id, device_us=device_us, wall_us=wall_us))

    local = dict(rank=dist.get_rank(), checks=checks, numerical_diagnostics=diagnostics, samples=samples)
    ranks = [None] * dist.get_world_size()
    dist.all_gather_object(ranks, local, group=cpu_group)
    summaries = {}
    for name in entries:
        summary = {}
        for metric in ("device_us", "wall_us"):
            values = []
            for round_id in range(rounds):
                data = [
                    next(s for s in rank["samples"] if s["case"] == name and s["round"] == round_id) for rank in ranks
                ]
                values.append(statistics.median(max(row[metric][i] for row in data) for i in range(repeats)))
            summary[metric] = statistics.median(values)
            summary[f"rounds_{metric}"] = values
        summaries[name] = summary
    comparisons = []
    for control, candidate, mechanism in paired_comparisons(cases):
        metrics = {}
        for metric in ("device_us", "wall_us"):
            base, value = summaries[control][metric], summaries[candidate][metric]
            metrics[metric] = dict(saved_us=base - value, improvement_fraction=(base - value) / base)
        comparisons.append(dict(control=control, candidate=candidate, mechanism=mechanism, **metrics))
    return dict(
        status="completed_with_numerical_diagnostics",
        scope="real_post_attention_inputs_and_mlp_weights_only_not_decoder_performance",
        endpoints={
            "down": "post_norm_whole_gateup_activation_then_down_and_tp_reduce",
            "norm": "local_O_tp_reduce_post_norm_whole_gateup_activation_down_tp_reduce",
        },
        rows=rows,
        pairs=[list(pair) for pair in pairs],
        rounds=rounds,
        repeats=repeats,
        timing="external_replay_events_and_host_dispatch_through_complete_endpoint",
        all_output_waits_and_joins_timed=True,
        immutable_input_restoration_timed=True,
        whole_gate_up_and_activation_every_case=True,
        reverse_order_preserves_request_rows=True,
        finite_cross_shape_outputs_required=True,
        strict_same_schedule_and_same_shape_outputs=True,
        summaries=summaries,
        comparisons=comparisons,
        ranks=ranks,
    )
