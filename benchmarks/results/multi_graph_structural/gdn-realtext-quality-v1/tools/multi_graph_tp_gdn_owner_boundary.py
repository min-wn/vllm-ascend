"""Compare complete request-owner input and output layout boundaries."""

import hashlib
from pathlib import Path


def measure_owner_boundary(args, group, cpu_group, capacity, query):
    """Time both hidden/residual endpoints, including packed-exit preparation."""
    import torch
    import torch.distributed as dist
    from multi_graph_tp_gdn_ownership import (
        canonical_gather_indices,
        capture_entry,
        owner_token_rows,
        time_entries,
    )

    rank, world = dist.get_rank(), dist.get_world_size()
    if world != 2 or capacity != 100 or query != 4 or group.world_size != world:
        raise ValueError("owner boundary qualification requires TP2, 100 capacity requests and Q4")
    width, tokens = 5120, capacity * query
    local_tokens = tokens // world
    device = torch.device("npu", torch.npu.current_device())
    root = torch.npu.Stream()
    root.wait_stream(torch.npu.current_stream())
    indices = owner_token_rows(capacity, query, rank, world)
    # Derive a separate request-ID oracle rather than validating the permutation
    # solely by applying its own inverse.
    oracle_indices = tuple(token for token in range(tokens) if (token // query) % world == rank)
    if indices != oracle_indices:
        raise ValueError("owner token selection disagrees with global request identity")
    inverse_values = canonical_gather_indices(capacity, query, world)
    rank_major = tuple(token for owner in range(world) for token in range(tokens) if (token // query) % world == owner)
    if tuple(rank_major[index] for index in inverse_values) != tuple(range(tokens)):
        raise ValueError("canonical owner permutation disagrees with independent token order")
    owner_rows = torch.tensor(indices, dtype=torch.int64, device=device)
    inverse = torch.tensor(inverse_values, dtype=torch.int64, device=device)
    hidden = torch.empty((tokens, width), dtype=torch.bfloat16, device=device)
    residual = torch.empty_like(hidden)
    local_hidden = torch.empty((local_tokens, width), dtype=hidden.dtype, device=device)
    local_residual = torch.empty_like(local_hidden)
    inputs = (hidden, residual, local_hidden, local_residual, owner_rows, inverse)
    addresses = tuple(value.data_ptr() for value in inputs)

    def entry_select():
        return hidden.index_select(0, owner_rows), residual.index_select(0, owner_rows)

    def exit_separate():
        gathered_hidden = group.all_gather(local_hidden, dim=0)
        gathered_residual = group.all_gather(local_residual, dim=0)
        return gathered_hidden.index_select(0, inverse), gathered_residual.index_select(0, inverse)

    def exit_packed():
        packed = torch.cat((local_hidden, local_residual), dim=-1)
        gathered = group.all_gather(packed, dim=0)
        canonical = gathered.index_select(0, inverse)
        return canonical[:, :width], canonical[:, width:]

    def exit_packed_contiguous():
        return tuple(value.contiguous() for value in exit_packed())

    def roundtrip_separate():
        selected_hidden, selected_residual = entry_select()
        gathered_hidden = group.all_gather(selected_hidden, dim=0)
        gathered_residual = group.all_gather(selected_residual, dim=0)
        return gathered_hidden.index_select(0, inverse), gathered_residual.index_select(0, inverse)

    def roundtrip_packed():
        selected_hidden, selected_residual = entry_select()
        packed = torch.cat((selected_hidden, selected_residual), dim=-1)
        gathered = group.all_gather(packed, dim=0)
        canonical = gathered.index_select(0, inverse)
        return canonical[:, :width], canonical[:, width:]

    def roundtrip_packed_contiguous():
        return tuple(value.contiguous() for value in roundtrip_packed())

    def refresh(live, variant):
        rows = torch.arange(tokens, dtype=torch.int32).unsqueeze(1)
        columns = torch.arange(width, dtype=torch.int32).unsqueeze(0)
        # Small integers are exactly representable in BF16, independently
        # distinguish tokens, channels, variants, and the two residual outputs.
        expected_hidden = ((rows * 17 + columns * 3 + variant * 11) % 97 - 48).to(torch.bfloat16)
        expected_residual = (-(rows * 13 + columns * 5 + variant * 7) % 89 - 44).to(torch.bfloat16)
        expected_hidden[live * query :].zero_()
        expected_residual[live * query :].zero_()
        reference = (expected_hidden, expected_residual)
        cpu_owner_rows = torch.tensor(oracle_indices, dtype=torch.int64)
        owner_reference = tuple(value.index_select(0, cpu_owner_rows) for value in reference)
        hidden.copy_(expected_hidden)
        residual.copy_(expected_residual)
        local_hidden.copy_(owner_reference[0])
        local_residual.copy_(owner_reference[1])
        torch.npu.synchronize()
        if tuple(value.data_ptr() for value in inputs) != addresses:
            raise ValueError("fixed owner boundary input address changed")
        return reference, owner_reference

    refresh(98, 0)
    operations = {
        "owner_entry_select": entry_select,
        "owner_exit_two_ag_reorder": exit_separate,
        "owner_exit_pack_one_ag_reorder_views": exit_packed,
        "owner_exit_pack_one_ag_reorder_contiguous": exit_packed_contiguous,
        "owner_roundtrip_two_ag_reorder": roundtrip_separate,
        "owner_roundtrip_pack_one_ag_reorder_views": roundtrip_packed,
        "owner_roundtrip_pack_one_ag_reorder_contiguous": roundtrip_packed_contiguous,
    }
    entries = {name: capture_entry(operation, root) for name, operation in operations.items()}
    output_addresses = {name: tuple(value.data_ptr() for value in entry["outputs"]) for name, entry in entries.items()}
    checks = []
    for variant, live in enumerate((98, 97, 99, 98), start=1):
        reference, owner_reference = refresh(live, variant)
        for name, entry in entries.items():
            with torch.npu.stream(root):
                expected = tuple(value.clone() for value in entry["operation"]())
                entry["graph"].replay()
            torch.npu.synchronize()
            if tuple(value.data_ptr() for value in entry["outputs"]) != output_addresses[name]:
                raise ValueError(f"captured output address changed: {name}")
            oracle = owner_reference if name == "owner_entry_select" else reference
            for field, actual, eager, wanted in zip(
                ("hidden", "residual"), entry["outputs"], expected, oracle, strict=True
            ):
                torch.testing.assert_close(actual, eager, rtol=0, atol=0)
                torch.testing.assert_close(actual.cpu(), wanted, rtol=0, atol=0)
                if live < capacity:
                    live_mask = (
                        torch.tensor(oracle_indices) < live * query
                        if name == "owner_entry_select"
                        else torch.arange(tokens) < live * query
                    )
                    torch.testing.assert_close(actual.cpu()[~live_mask], wanted[~live_mask], rtol=0, atol=0)
                checks.append(
                    dict(mode=name, live=live, variant=variant, field=field, eager_and_independent_oracle=True)
                )
            # AllGather must not consume or mutate the owned residual buffers.
            for actual, wanted in zip((local_hidden, local_residual), owner_reference, strict=True):
                torch.testing.assert_close(actual.cpu(), wanted, rtol=0, atol=0)
            for actual, wanted in zip((hidden, residual), reference, strict=True):
                torch.testing.assert_close(actual.cpu(), wanted, rtol=0, atol=0)
    refresh(98, 0)
    timing = time_entries(entries, cpu_group, args.rounds, args.repeats)
    rank_checks = [None] * world
    dist.all_gather_object(rank_checks, dict(rank=rank, checks=checks), group=cpu_group)
    layouts = {
        name: [
            dict(shape=list(value.shape), stride=list(value.stride()), contiguous=value.is_contiguous())
            for value in entry["outputs"]
        ]
        for name, entry in entries.items()
    }
    local_bytes = local_tokens * width * hidden.element_size()
    return dict(
        status="qualified_owner_boundaries_not_gdn_latency",
        scope=(
            "graph replay of owner row selection and both hidden/residual output reconstruction; "
            "no recurrence or dense computation"
        ),
        ownership="global_request_id modulo two, fixed owner capacity50 requests; dynamic live batches98/97/99/98",
        capacity_requests=capacity,
        query=query,
        capture_tokens=tokens,
        owner_capture_tokens=local_tokens,
        hidden_width=width,
        timing=timing,
        rank_checks=rank_checks,
        layouts=layouts,
        packed_contract=(
            "Cat hidden/residual to [200,10240] inside replay; one dim0 AllGather; one canonical row index_select; "
            "return two [400,5120] views with row stride10240. Last-dimension stride1; outputs are not contiguous."
        ),
        packed_consumer_limit=(
            "Views variants omit downstream consumers and are layout diagnostics. Contiguous variants include "
            "both output contiguous copies inside replay and match the separate-AllGather output layouts."
        ),
        timed_preparation=(
            "Input selection is included in roundtrip variants; "
            "packed Cat and canonical reorder are included in every packed exit"
        ),
        network_bytes_model=dict(
            separate_collective_calls=2,
            packed_collective_calls=1,
            local_hidden_bytes=local_bytes,
            local_hidden_and_residual_bytes=2 * local_bytes,
            final_replicated_hidden_and_residual_bytes=world * 2 * local_bytes,
            note=(
                "Packing changes startup count, not endpoint tensor payload bytes; "
                "HCCL internal traffic is not measured"
            ),
        ),
        rounds=args.rounds,
        repeats=args.repeats,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
