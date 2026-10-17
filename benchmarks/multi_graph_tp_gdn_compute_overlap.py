"""Qualify TP GateUp/state-core overlap and a closed GDN tail endpoint."""

import hashlib
from pathlib import Path


def run_compute_pair(mode, *, run_core, run_gateup, run_activation, root, compute, ready, done, torch_module=None):
    """Both paths finish the state core before issuing Vector activation."""
    if mode not in ("serial", "overlap"):
        raise ValueError("compute pair mode must be serial or overlap")
    if torch_module is None:
        import torch as torch_module
    if mode == "serial":
        core = run_core()
        gate = run_gateup()
    else:
        ready.record(root)
        with torch_module.npu.stream(compute):
            compute.wait_event(ready)
            core = run_core()
            done.record(compute)
        gate = run_gateup()
        root.wait_event(done)
    return gate, run_activation(gate), core


def measure_compute_overlap(args, workload, runner, model, group, cpu_group):
    """Freeze the A prefix and B projection; time real operations after them."""
    import torch
    import torch.distributed as dist
    import torch.nn.functional as functional
    import torch_npu
    from multi_graph_numerical_assessment import error_metrics
    from multi_graph_tp_gdn import qualify_gdn
    from multi_graph_tp_gdn_ownership import time_entries
    from multi_graph_tp_gdn_phases import GDNPhase

    rank, world = dist.get_rank(), dist.get_world_size()
    if world != 2 or workload.capacity != 100 or workload.query != 4 or tuple(args.pair) != (200, 200):
        raise ValueError("compute overlap requires TP2, Q4 and default capacities 400/200/200")
    layer_index = args.loader_metadata["layer_index"] + 1
    layer = model.get_submodule(f"language_model.model.layers.{layer_index}")
    if layer.layer_scale or layer.use_attn_reduce_scatter_for_moe:
        raise ValueError("compute overlap requires an ordinary unscaled dense GDN layer")
    torch.manual_seed(741)
    source = tuple(torch.randn((400, 5120), dtype=torch.bfloat16, device="npu") * 0.05 for _ in range(2))
    adjacent = qualify_gdn(args, workload, runner, model, source, cpu_group, setup_only=True)
    projected = adjacent.phases.project(*source, "full")
    left, right = adjacent.phases.partition(projected)
    adjacent.phases.core(left, "left")
    local_a, _ = adjacent.phases.output(left, "left")
    local_a = local_a.clone()
    prior = projected.residual.clone()
    normalized_a, _ = layer.post_attention_layernorm(group.all_reduce(local_a.clone()), prior[:200])
    normalized_full = torch.cat((normalized_a, normalized_a), dim=0)
    gate_weight = torch_npu.npu_format_cast(layer.mlp.gate_up_proj.weight, 29)
    output_weight = torch_npu.npu_format_cast(layer.linear_attn.out_proj.weight, 29)
    if any(torch_npu.get_npu_format(value) != 29 for value in (gate_weight, output_weight)):
        raise ValueError("expected NZ TP weights")
    states = tuple(layer.linear_attn.kv_cache)
    initial = tuple(value.clone() for value in states)
    frozen = dict(
        mixed_qkv=right.mixed_qkv.clone(),
        z=right.z.clone(),
        b=right.b.clone(),
        a=right.a.clone(),
        local_a=local_a.clone(),
        prior=prior.clone(),
        normalized_full=normalized_full.clone(),
    )
    buffers = dict(
        mixed_qkv=right.mixed_qkv,
        z=right.z,
        b=right.b,
        a=right.a,
        local_a=local_a,
        prior=prior,
        normalized_full=normalized_full,
    )
    addresses = {name: value.data_ptr() for name, value in buffers.items()}

    def refresh(live, variant=0, accepted=1):
        adjacent.refresh(live, accepted)
        for name, value in buffers.items():
            value.copy_(frozen[name] + variant * 0.0001)
            if name in ("prior", "normalized_full"):
                value[live * 4 :].zero_()
            elif name not in ("local_a",):
                value[max(0, live - 50) * 4 :].zero_()
        if {name: value.data_ptr() for name, value in buffers.items()} != addresses:
            raise ValueError("frozen compute input addresses changed")
        torch.npu.synchronize()

    def reset(previous=initial):
        for value, wanted in zip(states, previous, strict=True):
            value.copy_(wanted)
        torch.npu.synchronize()

    def core():
        # Fresh guards/output storage; the projection views and metadata stay
        # fixed across capture. The registered core owns all actual state work.
        phase = GDNPhase(
            right.owner,
            "right",
            right.mixed_qkv,
            right.z,
            right.b,
            right.a,
            torch.zeros_like(right.core_output),
            right.residual,
        )
        return adjacent.phases.core(phase, "right")

    main = torch.npu.current_stream()

    class Entry:
        def __init__(self, name):
            self.name = name
            self.root, self.compute, self.comm = (torch.npu.Stream() for _ in range(3))
            self.root.wait_stream(main)
            self.ready, self.done = torch.npu.Event(), torch.npu.Event()
            self.reduce_ready = [torch.npu.Event(), torch.npu.Event()]
            self.reduce_done = [torch.npu.Event(), torch.npu.Event()]
            self.graph = torch.npu.NPUGraph()

        def fork_core(self):
            self.ready.record(self.root)
            with torch.npu.stream(self.compute):
                self.compute.wait_event(self.ready)
                result = core()
                self.done.record(self.compute)
            return result

        def reduce(self, value, index):
            self.reduce_ready[index].record(self.root)
            with torch.npu.stream(self.comm):
                self.comm.wait_event(self.reduce_ready[index])
                reduced = group.all_reduce(value)
                self.reduce_done[index].record(self.comm)
            return reduced

        def operation(self):
            name = self.name
            if name == "core_only":
                return (core(),)
            if name == "gateup_half":
                gate = functional.linear(normalized_full[:200], gate_weight)
                return gate, layer.mlp.act_fn(gate)
            if name == "gateup_full":
                return (functional.linear(normalized_full, gate_weight),)
            if name == "gateup_split":
                return (torch.cat(tuple(functional.linear(part, gate_weight) for part in normalized_full.chunk(2))),)
            if name in ("pair_serial", "pair_overlap"):
                return run_compute_pair(
                    name.removeprefix("pair_"),
                    run_core=core,
                    run_gateup=lambda: functional.linear(normalized_full[:200], gate_weight),
                    run_activation=layer.mlp.act_fn,
                    root=self.root,
                    compute=self.compute,
                    ready=self.ready,
                    done=self.done,
                )

            # Both windows restore the same frozen real O_A before its in-place
            # reduction. Its copy and every wait/Norm/GEMM/join are timed.
            reduced_a = self.reduce(local_a.clone(), 0)
            if name == "window_full":
                core_b = self.fork_core()
                self.root.wait_event(self.done)
                projected_b = functional.linear(core_b, output_weight)
                reduced_b = self.reduce(projected_b, 1)
                self.root.wait_event(self.reduce_done[0])
                self.root.wait_event(self.reduce_done[1])
                joined = torch.cat((reduced_a, reduced_b))
                normalized, residual = layer.post_attention_layernorm(joined, prior)
                intermediate = layer.mlp.act_fn(functional.linear(normalized, gate_weight))
                return intermediate, residual, core_b
            if name == "window_early_core":
                core_b = self.fork_core()
                self.root.wait_event(self.reduce_done[0])
                normalized, residual_a = layer.post_attention_layernorm(reduced_a, prior[:200])
                gate_a = functional.linear(normalized, gate_weight)
                self.root.wait_event(self.done)
            else:
                self.root.wait_event(self.reduce_done[0])
                normalized, residual_a = layer.post_attention_layernorm(reduced_a, prior[:200])
                mode = "overlap" if name == "window_overlap" else "serial"
                gate_a, _, core_b = run_compute_pair(
                    mode,
                    run_core=core,
                    run_gateup=lambda: functional.linear(normalized, gate_weight),
                    # Activation remains deferred until B O has published its AR.
                    run_activation=lambda gate: gate,
                    root=self.root,
                    compute=self.compute,
                    ready=self.ready,
                    done=self.done,
                )
            projected_b = functional.linear(core_b, output_weight)
            reduced_b = self.reduce(projected_b, 1)
            activation_a = layer.mlp.act_fn(gate_a)
            self.root.wait_event(self.reduce_done[1])
            normalized_b, residual_b = layer.post_attention_layernorm(reduced_b, prior[200:])
            activation_b = layer.mlp.act_fn(functional.linear(normalized_b, gate_weight))
            return torch.cat((activation_a, activation_b)), torch.cat((residual_a, residual_b)), core_b

    names = (
        "core_only",
        "gateup_half",
        "gateup_full",
        "gateup_split",
        "pair_serial",
        "pair_overlap",
        "window_full",
        "window_split_serial",
        "window_overlap",
        "window_early_core",
    )
    refresh(workload.batch)
    entries = {name: Entry(name) for name in names}
    for entry in entries.values():
        for _ in range(2):
            reset()
            with torch.npu.stream(entry.root):
                entry.operation()
            torch.npu.synchronize()
        reset()
        dist.barrier(group=cpu_group)
        with torch.npu.graph(entry.graph, pool=torch.npu.graph_pool_handle(), stream=entry.root):
            entry.outputs = entry.operation()
        torch.npu.synchronize()

    checks, diagnostics = [], []
    trajectories = {name: tuple(value.clone() for value in initial) for name in names}
    core_modes = tuple(name for name in names if name not in ("gateup_half", "gateup_full", "gateup_split"))
    for step, (live, accepted) in enumerate(zip((98, 97, 99, 98), (1, 2, 1, 3), strict=True)):
        refresh(live, step + 1, accepted)
        for name, entry in entries.items():
            previous = trajectories[name]
            reset(previous)
            entry.root.wait_stream(torch.npu.current_stream())
            with torch.npu.stream(entry.root):
                expected = tuple(value.clone() for value in entry.operation())
            torch.npu.synchronize()
            expected_states = tuple(value.clone() for value in states)
            reset(previous)
            entry.root.wait_stream(torch.npu.current_stream())
            with torch.npu.stream(entry.root):
                entry.graph.replay()
            torch.npu.synchronize()
            for actual, wanted in zip(entry.outputs, expected, strict=True):
                torch.testing.assert_close(actual, wanted, rtol=0, atol=0)
                if not torch.isfinite(actual).all().item():
                    raise ValueError(f"nonfinite compute output: {name}")
            for actual, wanted, before in zip(states, expected_states, previous, strict=True):
                torch.testing.assert_close(actual, wanted, rtol=0, atol=0)
                torch.testing.assert_close(actual[:201], before[:201], rtol=0, atol=0)
                torch.testing.assert_close(actual[1 + live * 4 :], before[1 + live * 4 :], rtol=0, atol=0)
                if name not in core_modes:
                    torch.testing.assert_close(actual, before, rtol=0, atol=0)
            trajectories[name] = tuple(value.clone() for value in states)
            checks.append(f"{name}/step{step}/strict_outputs_full_state_left_null_inactive")
        for name in core_modes:
            for observed, wanted in zip(trajectories[name], trajectories["core_only"], strict=True):
                torch.testing.assert_close(observed, wanted, rtol=0, atol=0)
        for pair in (("pair_serial", "pair_overlap"), ("window_split_serial", "window_overlap")):
            for observed, wanted in zip(entries[pair[0]].outputs, entries[pair[1]].outputs, strict=True):
                torch.testing.assert_close(observed, wanted, rtol=0, atol=0)
            checks.append(f"{pair[0]}/{pair[1]}/step{step}/same_arithmetic_exact")
        for actual, wanted in zip(
            entries["window_early_core"].outputs, entries["window_split_serial"].outputs, strict=True
        ):
            torch.testing.assert_close(actual, wanted, rtol=0, atol=0)
        checks.append(f"window_early_core/window_split_serial/step{step}/same_arithmetic_exact")
        for candidate, reference in (("gateup_split", "gateup_full"), ("window_overlap", "window_full")):
            for field, (observed, wanted) in enumerate(zip(entries[candidate].outputs, entries[reference].outputs)):
                diagnostics.append(
                    dict(
                        candidate=candidate,
                        reference=reference,
                        step=step,
                        field=field,
                        **error_metrics(observed, wanted),
                    )
                )
    refresh(workload.batch)
    reset()
    timed = {name: dict(graph=entry.graph, root=entry.root, prepare_timing=reset) for name, entry in entries.items()}
    timing = time_entries(timed, cpu_group, args.rounds, args.repeats)
    if args.trace:
        with torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU]
        ) as profiler:
            for name, entry in entries.items():
                reset()
                for _ in range(2):
                    dist.barrier(group=cpu_group)
                    with torch.profiler.record_function(f"gdn_compute_overlap.{name}"):
                        with torch.npu.stream(entry.root):
                            entry.graph.replay()
                        torch.npu.synchronize()
        profiler.export_chrome_trace(str(args.output.with_suffix(f".rank{rank}.trace.json")))
    ranks = [None] * world
    dist.all_gather_object(ranks, dict(rank=rank, checks=checks, diagnostics=diagnostics), group=cpu_group)
    return dict(
        status="qualified_actual_compute_pair_and_closed_tail_not_full_gdn",
        scope="frozen_real_A_output_and_B_projection_to_full_activation_residual_and_B_state",
        excluded=["A_state_core", "A_O_projection", "both_input_projections", "MLP_Down_and_reductions", "full_model"],
        pair_scope="frozen_normalized_A_GateUp_plus_official_B_state_core_then_deferred_activation",
        capture_tokens=400,
        branch_tokens=[200, 200],
        tp_size=world,
        local_heads=layer.linear_attn.num_v_heads // world,
        gateup_weight_shape=list(gate_weight.shape),
        weight_format="NZ",
        replicated_weights=False,
        collective_order=["O_A", "O_B"],
        state_contract="all_four_speculative_snapshots; A/null/inactive unchanged",
        dynamic_steps=4,
        ranks=ranks,
        timing=timing,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
