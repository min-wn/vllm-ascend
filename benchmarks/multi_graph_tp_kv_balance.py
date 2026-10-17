"""FIA work balancing across fixed request branches, with permutation costs.

This isolates frozen projected Q/K/V from one real attention layer. Prefix KV
comes from the caller. It is not a coherent full-model or generation benchmark.
"""

import random
import statistics
import time


def length_profiles(capacity, batch, query, maximum, seed=20261017):
    """Random lengths and a clustered control with the identical histogram."""
    if not 1 <= batch <= capacity or not 1 <= query <= maximum:
        raise ValueError("live batch and query must fit the allocated capacity")
    minimum = max(query, maximum // 8)
    generator = random.Random(seed)
    live = [generator.randint(minimum, maximum) for _ in range(batch)]
    return {
        "random": live + [0] * (capacity - batch),
        "clustered": sorted(live, reverse=True) + [0] * (capacity - batch),
    }


def assign_requests(lengths, capacities, balanced):
    """Quota-constrained longest-job-first placement, preserving global IDs."""
    if len(capacities) != 2 or min(capacities) <= 0 or sum(capacities) != len(lengths):
        raise ValueError("two positive branch capacities must cover every request")
    if any(length < 0 for length in lengths):
        raise ValueError("request lengths cannot be negative")
    if not balanced:
        return (tuple(range(capacities[0])), tuple(range(capacities[0], len(lengths))))
    lanes, loads = [[], []], [0, 0]
    for request in sorted(range(len(lengths)), key=lambda index: (-lengths[index], index)):
        available = [lane for lane in range(2) if len(lanes[lane]) < capacities[lane]]
        lane = min(available, key=lambda item: (loads[item], len(lanes[item]), item))
        lanes[lane].append(request)
        loads[lane] += lengths[request]
    return tuple(tuple(lane) for lane in lanes)


def token_indices(requests, query):
    return [request * query + offset for request in requests for offset in range(query)]


def mapped_metadata(workload, lengths, requests):
    """Map logical request order to its existing physical pages and slots."""
    if len(lengths) != workload.capacity or len(set(requests)) != len(requests):
        raise ValueError("lengths must cover capacity and requests must be unique")
    if any(request < 0 or request >= workload.capacity for request in requests):
        raise ValueError("request identity is outside the allocated capacity")
    if any(length and not workload.query <= length <= workload.kv_length for length in lengths):
        raise ValueError("live lengths must fit the query and allocated KV")
    width, block = workload.pages_per_request, workload.block_size
    return dict(
        lengths=[lengths[request] for request in requests],
        boundaries=[index * workload.query for index in range(len(requests) + 1)],
        pages=[
            list(range(request * width, (request + 1) * width)) if lengths[request] else [0] * width
            for request in requests
        ],
        positions=[
            position
            for request in requests
            for position in (
                range(lengths[request] - workload.query, lengths[request]) if lengths[request] else [0] * workload.query
            )
        ],
        slots=[
            request * width * block + position if lengths[request] else -1
            for request in requests
            for position in (
                range(lengths[request] - workload.query, lengths[request]) if lengths[request] else [0] * workload.query
            )
        ],
    )


def measure_kv_balance(
    workload,
    projection,
    impl,
    backend,
    caches,
    mask,
    context_factory,
    restore_cache,
    cpu_group,
    *,
    pair=(200, 200),
    rounds=3,
    repeats=6,
):
    """Time actual FIA+gate, all branch joins, and balanced GPU gather/scatter.

    context_factory(mapping, token_count) must return the selected layer's
    forward context. projection is canonical global-request Q,K,V,gate. All
    modes consume exactly those tensors; they do not reproject or change KV
    history. Metadata planning/upload is reported separately from graph replay.
    """
    import torch
    import torch.distributed as dist
    import torch_npu
    from multi_graph_numerical_assessment import error_metrics
    from vllm.forward_context import get_forward_context

    from vllm_ascend.attention.attention_v1 import AscendAttentionState, AscendMetadata
    from vllm_ascend.compilation.updatable_graph import UpdatableGraph

    if rounds < 1 or repeats < 1 or len(projection) != 4:
        raise ValueError("positive samples and Q,K,V,gate are required")
    rows = workload.capacity * workload.query
    if pair not in ((200, 200), (272, 128)) or sum(pair) != rows or any(size % workload.query for size in pair):
        raise ValueError("only existing 200+200 or 272+128 capacities covering the bucket are accepted")
    if any(value.shape[0] != rows for value in projection):
        raise ValueError("every projected tensor must have canonical full-bucket rows")
    capacities = tuple(size // workload.query for size in pair)
    main = torch.npu.current_stream()
    layer_name = backend.layer_name
    profiles = length_profiles(workload.capacity, workload.batch, workload.query, workload.kv_length)
    rank, world = dist.get_rank(), dist.get_world_size()
    cases = ("whole", "contiguous_serial", "contiguous_parallel", "balanced_serial", "balanced_parallel")
    checks, samples, planning, uploads = [], [], [], []
    restore_cache()
    initial_cache = tuple(value.clone() for value in caches)
    torch.npu.synchronize()

    class LaneGraph(UpdatableGraph):
        def __init__(self):
            super().__init__()
            self.lane = 0
            self.task_lanes = []

        def get_capture_resource(self, key, factory, use_max_workspace=False):
            return super().get_capture_resource((self.lane, key), factory, use_max_workspace)

        def register_task(self, operation, kwargs, provider):
            super().register_task(operation, kwargs, provider)
            self.task_lanes.append(self.lane)

    class Source:
        def __init__(self, entry):
            self.entry = entry

        def get(self, provider):
            return tuple(
                provider.resolve(self.entry.mappings[lane])
                for task, lane in zip(self.entry.graph.tasks, self.entry.graph.task_lanes, strict=True)
                if task.provider == provider
            )

    class Entry:
        def __init__(self, case, lengths):
            self.case = case
            self.balanced = case.startswith("balanced")
            self.parallel = case.endswith("parallel")
            self.requests = (
                (tuple(range(workload.capacity)),)
                if case == "whole"
                else assign_requests(lengths, capacities, self.balanced)
            )
            self.root, self.other, self.update_stream = torch.npu.Stream(), torch.npu.Stream(), torch.npu.Stream()
            self.prepared, self.updated, self.fork, self.join = (torch.npu.Event() for _ in range(4))
            self.graph, self.pool = LaneGraph(), torch.npu.graph_pool_handle()
            self.output = torch.empty(
                (rows, projection[3].shape[-1]), dtype=projection[0].dtype, device=projection[0].device
            )
            self.indices, self.mappings = [], []
            for requests in self.requests:
                values = mapped_metadata(workload, lengths, requests)
                host_lengths = torch.tensor(values["lengths"], dtype=torch.int32)
                boundaries = torch.tensor(values["boundaries"], dtype=torch.int32)
                metadata = AscendMetadata(
                    attn_mask=mask,
                    attn_state=AscendAttentionState.DecodeOnly
                    if workload.query == 1
                    else AscendAttentionState.ChunkedPrefill,
                    num_actual_tokens=len(requests) * workload.query,
                    num_decode_tokens=len(requests) * workload.query,
                    num_decodes=len(requests),
                    seq_lens=host_lengths,
                    seq_lens_cpu=host_lengths,
                    seq_lens_list=values["lengths"],
                    seq_lens_gpu=host_lengths.npu(),
                    query_start_loc=boundaries,
                    query_start_loc_gpu=boundaries.npu(),
                    actual_seq_lengths_q=values["boundaries"][1:],
                    max_query_len=workload.query,
                    block_tables=torch.tensor(values["pages"], dtype=torch.int32, device="npu"),
                    slot_mapping=torch.tensor(values["slots"], dtype=torch.int64, device="npu"),
                    model_runner_type="generate",
                )
                self.mappings.append({layer_name: metadata})
                self.indices.append(
                    torch.tensor(token_indices(requests, workload.query), dtype=torch.int64, device="npu")
                )
            self.source = Source(self)

        def lane(self, lane):
            requests = self.requests[lane]
            if self.balanced:
                q, k, v, gate = (value.index_select(0, self.indices[lane]) for value in projection)
            else:
                start = requests[0] * workload.query
                stop = (requests[-1] + 1) * workload.query
                q, k, v, gate = (value[start:stop] for value in projection)
            output = torch.empty_like(q)
            capturing = get_forward_context().capturing
            self.graph.lane = lane
            with context_factory(self.mappings[lane], len(requests) * workload.query):
                get_forward_context().capturing = capturing
                impl.forward(backend, q, k, v, caches, self.mappings[lane][layer_name], output=output)
            value = output.reshape(-1, gate.shape[-1]) * torch.sigmoid(gate)
            if self.balanced:
                self.output.index_copy_(0, self.indices[lane], value)
            else:
                self.output[start:stop].copy_(value)

        def operation(self):
            if self.parallel:
                self.fork.record(self.root)
            self.lane(0)
            if len(self.requests) == 2:
                if self.parallel:
                    with torch.npu.stream(self.other):
                        self.other.wait_event(self.fork)
                        self.lane(1)
                        self.join.record(self.other)
                    self.root.wait_event(self.join)
                else:
                    self.lane(1)
            return self.output

        def capture(self):
            self.root.wait_stream(main)
            self.other.wait_stream(main)
            restore_cache()
            with torch.npu.stream(self.root), context_factory(self.mappings[0], len(self.requests[0]) * workload.query):
                for _ in range(3):
                    self.operation()
            torch.npu.synchronize()
            dist.barrier(group=cpu_group)
            with context_factory(self.mappings[0], len(self.requests[0]) * workload.query):
                get_forward_context().capturing = True
                with torch.npu.graph(self.graph, pool=self.pool, stream=self.root):
                    self.operation()
            torch.npu.synchronize()
            if len(self.graph.tasks) != len(self.requests) or any(
                task.operation != torch_npu.npu_fused_infer_attention_score.out for task in self.graph.tasks
            ):
                raise ValueError("every captured FIA branch must retain its real update task")
            workspaces = [value.data_ptr() for value in self.graph.capture_resources.values()]
            if len(set(workspaces)) != len(self.requests):
                raise ValueError("concurrent FIA branches must use independent workspaces")

        def replay(self):
            tasks = self.graph.resolve_tasks(self.source)
            self.prepared.record(main)
            self.root.wait_event(self.prepared)
            self.update_stream.wait_event(self.prepared)
            with torch.npu.stream(self.root):
                self.graph.replay()
            self.graph.update(self.update_stream, tasks)
            self.updated.record(self.update_stream)
            self.root.wait_event(self.updated)

        def refresh(self, lengths):
            requests = (
                (tuple(range(workload.capacity)),)
                if self.case == "whole"
                else assign_requests(lengths, capacities, self.balanced)
            )
            for lane, ids in enumerate(requests):
                values = mapped_metadata(workload, lengths, ids)
                metadata = self.mappings[lane][layer_name]
                metadata.seq_lens_list[:] = values["lengths"]
                metadata.seq_lens.copy_(torch.tensor(values["lengths"], dtype=torch.int32))
                metadata.seq_lens_gpu.copy_(metadata.seq_lens)
                metadata.block_tables.copy_(torch.tensor(values["pages"], dtype=torch.int32))
                metadata.slot_mapping.copy_(torch.tensor(values["slots"], dtype=torch.int64))
                self.indices[lane].copy_(torch.tensor(token_indices(ids, workload.query), dtype=torch.int64))
            # Balanced computation uses runtime device indices; contiguous slices
            # are invariant. This assignment is only a host correctness record.
            self.requests = requests

    reports = {}
    for profile, lengths in profiles.items():
        entries = {case: Entry(case, lengths) for case in cases}
        for case, entry in entries.items():
            print(f"rank={rank} FIA balance capture {profile}/{case}", flush=True)
            entry.capture()
        for variant, live in enumerate((workload.batch, workload.batch - 1, workload.capacity - 1, workload.batch)):
            changed = length_profiles(
                workload.capacity, live, workload.query, workload.kv_length, seed=20261017 + variant
            )[profile]
            for entry in entries.values():
                entry.refresh(changed)
            restore_cache()
            entries["whole"].replay()
            torch.npu.synchronize()
            expected = entries["whole"].output.clone()
            expected_caches = tuple(value.clone() for value in caches)
            for case, entry in entries.items():
                restore_cache()
                with (
                    torch.npu.stream(entry.root),
                    context_factory(entry.mappings[0], len(entry.requests[0]) * workload.query),
                ):
                    eager = entry.operation().clone()
                torch.npu.synchronize()
                eager_caches = tuple(value.clone() for value in caches)
                restore_cache()
                entry.replay()
                torch.npu.synchronize()
                torch.testing.assert_close(
                    entry.output[: live * workload.query], eager[: live * workload.query], rtol=0, atol=0
                )
                for actual, wanted, whole, initial in zip(
                    caches, eager_caches, expected_caches, initial_cache, strict=True
                ):
                    torch.testing.assert_close(actual, wanted, rtol=0, atol=0)
                    torch.testing.assert_close(actual, whole, rtol=0, atol=0)
                    page = workload.pages_per_request
                    torch.testing.assert_close(actual[live * page :], initial[live * page :], rtol=0, atol=0)
                torch.testing.assert_close(
                    entry.output[: live * workload.query],
                    expected[: live * workload.query],
                    rtol=0.02,
                    atol=0.02,
                )
                metrics = error_metrics(entry.output[: live * workload.query], expected[: live * workload.query])
                checks.append(
                    dict(
                        profile=profile,
                        case=case,
                        variant=variant,
                        live=live,
                        same_schedule_bitwise=True,
                        identical_global_kv=True,
                        **metrics,
                    )
                )
        for case, entry in entries.items():
            elapsed = []
            for _ in range(max(6, repeats)):
                start = time.perf_counter()
                assigned = (
                    (tuple(range(workload.capacity)),)
                    if case == "whole"
                    else assign_requests(lengths, capacities, entry.balanced)
                )
                values = [mapped_metadata(workload, lengths, ids) for ids in assigned]
                indices = [token_indices(ids, workload.query) for ids in assigned]
                elapsed.append((time.perf_counter() - start) * 1e6)
            planning.append(
                dict(
                    profile=profile,
                    case=case,
                    median_us=statistics.median(elapsed),
                    includes="CPU_assignment_metadata_lists_and_indices",
                )
            )
            del values, indices
            start = time.perf_counter()
            entry.refresh(lengths)
            torch.npu.synchronize()
            uploads.append(dict(profile=profile, case=case, host_and_upload_us=(time.perf_counter() - start) * 1e6))
        for round_id in range(rounds):
            order = cases if round_id % 2 == 0 else tuple(reversed(cases))
            for case in order:
                entry = entries[case]
                restore_cache()
                entry.replay()
                torch.npu.synchronize()
                device_us, wall_us = [], []
                begin, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
                for _ in range(repeats):
                    dist.barrier(group=cpu_group)
                    start = time.perf_counter()
                    with torch.npu.stream(entry.root):
                        begin.record(entry.root)
                        entry.replay()
                        end.record(entry.root)
                    end.synchronize()
                    wall_us.append((time.perf_counter() - start) * 1e6)
                    device_us.append(begin.elapsed_time(end) * 1000)
                samples.append(dict(profile=profile, case=case, round=round_id, device_us=device_us, wall_us=wall_us))
        reports[profile] = dict(
            lengths=lengths,
            total_kv_tokens=sum(lengths),
            assignments={case: [list(ids) for ids in entry.requests] for case, entry in entries.items()},
            branch_kv_tokens={
                case: [sum(lengths[index] for index in ids) for ids in entry.requests]
                for case, entry in entries.items()
            },
        )
        del entries
    local = dict(rank=rank, checks=checks, samples=samples, planning=planning, uploads=uploads)
    ranks = [None] * world
    dist.all_gather_object(ranks, local, group=cpu_group)
    for profile, report in reports.items():
        report["summaries"] = {}
        for case in cases:
            summary = {}
            for metric in ("device_us", "wall_us"):
                medians = []
                for round_id in range(rounds):
                    data = [
                        next(
                            row
                            for row in item["samples"]
                            if row["profile"] == profile and row["case"] == case and row["round"] == round_id
                        )
                        for item in ranks
                    ]
                    medians.append(
                        statistics.median(max(row[metric][sample] for row in data) for sample in range(repeats))
                    )
                summary[metric] = statistics.median(medians)
                summary[f"rounds_{metric}"] = medians
            report["summaries"][case] = summary
    return dict(
        status="completed_with_numerical_diagnostics",
        scope="frozen_projected_QKV_FIA_gate_and_output_order_only_not_decoder",
        pair=list(pair),
        rounds=rounds,
        repeats=repeats,
        gpu_q_k_v_gate_gather_and_inverse_scatter_timed=True,
        metadata_cpu_planning_and_upload="separately_reported_not_in_replay",
        all_real_fia_updates_after_replay_submission=True,
        graph_and_metadata_reused_across_batch_and_assignment_changes=True,
        original_global_kv_page_and_slot_identity_preserved=True,
        clustered_profile_is_diagnostic_not_random_workload_claim=True,
        profiles=reports,
        ranks=ranks,
    )
