"""CPU checks for component-aware TP reconstruction and stable request owners."""

import ast
import importlib.util
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as functional


@pytest.fixture
def ownership(monkeypatch):
    path = Path(__file__).resolve().parents[4] / "benchmarks" / "multi_graph_tp_gdn_ownership.py"
    spec = importlib.util.spec_from_file_location("gdn_ownership_cpu_test", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def owner_core(monkeypatch):
    path = Path(__file__).resolve().parents[4] / "benchmarks" / "multi_graph_tp_gdn_owner_core.py"
    spec = importlib.util.spec_from_file_location("gdn_owner_core_cpu_test", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def owner_mlp(monkeypatch):
    path = Path(__file__).resolve().parents[4] / "benchmarks" / "multi_graph_tp_gdn_owner_mlp.py"
    spec = importlib.util.spec_from_file_location("gdn_owner_mlp_cpu_test", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("parts", [1, 2, 4])
def test_owner_mlp_partition_preserves_packed_projection_activation_and_down(owner_mlp, parts):
    torch.manual_seed(712)
    rows, hidden, width = 5, 7, 12
    inputs = torch.randn(rows, hidden, dtype=torch.float64)
    weight = torch.randn(hidden, width * 2, dtype=torch.float64).T
    down = torch.randn(hidden, width, dtype=torch.float64)
    assert not weight.is_contiguous()
    groups = owner_mlp.partition_gate_up(weight, parts)
    assert len(groups) == parts
    assert all(group.is_contiguous() and group.shape == (width * 2 // parts, hidden) for group in groups)
    full_gate, full_up = functional.linear(inputs, weight).chunk(2, dim=-1)
    gate_chunks, up_chunks, activated_chunks = [], [], []
    for index, group in enumerate(groups):
        start, stop = index * width // parts, (index + 1) * width // parts
        torch.testing.assert_close(group[: width // parts], weight[start:stop], rtol=0, atol=0)
        torch.testing.assert_close(group[width // parts :], weight[width + start : width + stop], rtol=0, atol=0)
        gate, up = functional.linear(inputs, group).chunk(2, dim=-1)
        gate_chunks.append(gate)
        up_chunks.append(up)
        activated_chunks.append(functional.silu(gate) * up)
    reconstructed = torch.cat((*gate_chunks, *up_chunks), dim=-1)
    torch.testing.assert_close(reconstructed, torch.cat((full_gate, full_up), dim=-1), rtol=1e-12, atol=1e-12)
    activated = torch.cat(activated_chunks, dim=-1)
    torch.testing.assert_close(activated, functional.silu(full_gate) * full_up, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(
        functional.linear(activated, down),
        functional.linear(functional.silu(full_gate) * full_up, down),
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.parametrize(
    "shape,parts",
    [
        ((0, 3), 1),
        ((6, 0), 1),
        ((5, 3), 1),
        ((6,), 1),
        ((2, 3, 4), 1),
        ((8, 3), 0),
        ((12, 3), 3),
        ((8, 3), True),
        ((12, 3), 4),
    ],
)
def test_owner_mlp_partition_rejects_invalid_weight_or_groups(owner_mlp, shape, parts):
    with pytest.raises(ValueError):
        owner_mlp.partition_gate_up(torch.ones(shape), parts)


@pytest.mark.parametrize("widths", [(2, 3, 6, 6), (3, 3), (5, 5)])
@pytest.mark.parametrize("world", [1, 2, 3])
@pytest.mark.parametrize("dim", [0, 1])
def test_merge_reconstructs_components_without_rank_interleaving(ownership, widths, world, dim):
    # Unique component/rank/channel sentinels expose a whole-shard concatenation.
    shards, expected_components = [], [[] for _ in widths]
    for rank in range(world):
        pieces = []
        for component, width in enumerate(widths):
            channels = torch.arange(width, dtype=torch.float64) + component * 1000 + rank * 100
            piece = channels[:, None] + torch.arange(4, dtype=torch.float64)[None, :] / 10
            if dim == 1:
                piece = piece.T
            pieces.append(piece)
            expected_components[component].append(piece)
        shards.append(torch.cat(pieces, dim=dim))
    expected = torch.cat([torch.cat(pieces, dim=dim) for pieces in expected_components], dim=dim)
    actual = ownership.merge_component_shards(shards, widths, dim=dim)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    if world > 1:
        assert not torch.equal(actual, torch.cat(shards, dim=dim))


@pytest.mark.parametrize("world", [1, 2, 3])
def test_row_sharding_selects_each_components_rank_slice(ownership, world):
    local_sizes, hidden = (2, 4, 6, 6), 5
    full_sizes = tuple(size * world for size in local_sizes)
    weight = torch.arange(sum(full_sizes) * hidden, dtype=torch.float64).reshape(-1, hidden)
    components = weight.split(full_sizes)
    for rank in range(world):
        expected = torch.cat(
            [component[rank * width : (rank + 1) * width] for component, width in zip(components, local_sizes)]
        )
        actual = ownership.shard_component_rows(weight, full_sizes, rank, world)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("local_sizes", [(2, 2, 6, 6), (6, 6)])
def test_merged_qkvz_and_ba_projection_preserves_each_head_component(ownership, local_sizes):
    torch.manual_seed(532)
    rows, hidden, world = 7, 5, 2
    values = torch.randn(rows, hidden, dtype=torch.float64)
    shards = [torch.randn(sum(local_sizes), hidden, dtype=torch.float64) for _ in range(world)]
    full = ownership.merge_component_shards(shards, local_sizes)
    actual = functional.linear(values, full).split(tuple(width * world for width in local_sizes), dim=-1)
    projected_shards = [functional.linear(values, shard).split(local_sizes, dim=-1) for shard in shards]
    for component, output in enumerate(actual):
        expected = torch.cat([projected[component] for projected in projected_shards], dim=-1)
        torch.testing.assert_close(output, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("world", [1, 2, 3])
def test_owner_mlp_matches_sum_of_tp_shards(ownership, world):
    torch.manual_seed(992)
    requests, query, hidden, local_width = 6, 4, 5, 3
    capacity = requests * query
    width = local_width * world
    values = torch.randn(capacity, hidden, dtype=torch.float64)
    gate_full, up_full = (torch.randn(width, hidden, dtype=torch.float64) for _ in range(2))
    down_full = torch.randn(hidden, width, dtype=torch.float64)
    gate_up_shards, down_shards, partials = [], [], []
    for rank in range(world):
        start, stop = rank * local_width, (rank + 1) * local_width
        gate_up = torch.cat([gate_full[start:stop], up_full[start:stop]])
        down = down_full[:, start:stop]
        gate, up = functional.linear(values, gate_up).chunk(2, dim=-1)
        partials.append(functional.linear(functional.silu(gate) * up, down))
        gate_up_shards.append(gate_up)
        down_shards.append(down)
    gate_up = ownership.merge_component_shards(gate_up_shards, (local_width, local_width))
    down = ownership.merge_component_shards(down_shards, (local_width,), dim=1)
    owners = []
    for rank in range(world):
        indices = torch.tensor(ownership.owner_token_rows(requests, query, rank, world))
        gate, up = functional.linear(values[indices], gate_up).chunk(2, dim=-1)
        owners.append(functional.linear(functional.silu(gate) * up, down))
    inverse = torch.tensor(ownership.canonical_gather_indices(requests, query, world))
    actual = torch.cat(owners)[inverse]
    expected = torch.stack(partials).sum(0)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("dim", [0, 1])
def test_merge_accepts_noncontiguous_projection_shards(ownership, dim):
    shards = [torch.arange(48, dtype=torch.float64).reshape(8, 6).T + rank * 100 for rank in (0, 1)]
    assert all(not shard.is_contiguous() for shard in shards)
    widths = (2, 4) if dim == 0 else (3, 5)
    expected = torch.cat(
        [torch.cat([shard.split(widths, dim=dim)[part] for shard in shards], dim=dim) for part in (0, 1)],
        dim=dim,
    )
    torch.testing.assert_close(ownership.merge_component_shards(shards, widths, dim=dim), expected)


@pytest.mark.parametrize("capacity,query,world", [(100, 4, 2), (8, 1, 4), (6, 4, 3), (1, 4, 1)])
def test_owner_mapping_covers_tokens_and_gather_restores_hidden_and_residual(ownership, capacity, query, world):
    all_owners = [ownership.owner_token_rows(capacity, query, rank, world) for rank in range(world)]
    flat = [token for owner in all_owners for token in owner]
    assert sorted(flat) == list(range(capacity * query))
    assert len(set(flat)) == len(flat)
    for rank, owner in enumerate(all_owners):
        assert len(owner) == capacity * query // world
        assert all(token // query % world == rank for token in owner)
        assert all(
            owner[index : index + query] == tuple(range(owner[index], owner[index] + query))
            for index in range(0, len(owner), query)
        )
    inverse = ownership.canonical_gather_indices(capacity, query, world)
    assert tuple(flat[index] for index in inverse) == tuple(range(capacity * query))
    hidden = torch.arange(capacity * query * 3).reshape(-1, 3)
    residual = hidden * -7 + 19
    for value in (hidden, residual):
        gathered = torch.cat([value[list(owner)] for owner in all_owners])
        torch.testing.assert_close(gathered[list(inverse)], value, rtol=0, atol=0)


def test_q4_owners_remain_stable_as_live_batch_shrinks_and_grows(ownership):
    capacity, query, world = 100, 4, 2
    owners = [ownership.owner_token_rows(capacity, query, rank, world) for rank in range(world)]
    assert [len(owner) for owner in owners] == [200, 200]
    expected_live_requests = [(49, 49), (49, 48), (50, 49), (49, 49)]
    previous = {}
    for live, expected in zip((98, 97, 99, 98), expected_live_requests):
        counts, current = [], {}
        for rank, owner in enumerate(owners):
            active = [token for token in owner if token // query < live]
            counts.append(len(active) // query)
            for token in active:
                current[token] = rank
                assert token not in previous or previous[token] == rank
        assert tuple(counts) == expected
        assert sorted(current) == list(range(live * query))
        previous = current


@pytest.mark.parametrize(
    "shards,widths,dim",
    [
        ([], (2, 2), 0),
        ([torch.ones(4, 3)], (), 0),
        ([torch.ones(4, 3)], (0, 4), 0),
        ([torch.ones(4, 3)], (-1, 5), 0),
        ([torch.ones(4, 3)], (2, 1), 0),
        ([torch.ones(4, 3), torch.ones(4, 2)], (2, 2), 0),
        ([torch.ones(4, 3), torch.ones(3, 3)], (2, 2), 0),
        ([torch.ones(4, 3)], (2, 2), 2),
        ([torch.ones(4, 3)], (2, 2), -1),
    ],
)
def test_merge_rejects_invalid_component_contract(ownership, shards, widths, dim):
    with pytest.raises(ValueError):
        ownership.merge_component_shards(shards, widths, dim=dim)


@pytest.mark.parametrize(
    "widths,rank,world",
    [((2, 2), 0, 0), ((2, 2), -1, 2), ((2, 2), 2, 2), ((3, 1), 0, 2), ((2, 1), 0, 1), ((), 0, 1)],
)
def test_row_sharding_rejects_invalid_component_or_rank(ownership, widths, rank, world):
    with pytest.raises(ValueError):
        ownership.shard_component_rows(torch.ones(4, 3), widths, rank, world)


@pytest.mark.parametrize(
    "capacity,query,rank,world",
    [(100, 4, 0, 0), (99, 4, 0, 2), (100, 0, 0, 2), (100, -1, 0, 2), (100, 4, -1, 2), (100, 4, 2, 2)],
)
def test_owner_mapping_rejects_unsupported_shape_or_rank(ownership, capacity, query, rank, world):
    with pytest.raises(ValueError):
        ownership.owner_token_rows(capacity, query, rank, world)


@pytest.mark.parametrize("capacity,query,world", [(100, 4, 0), (100, 4, -2), (99, 4, 2), (100, 0, 2)])
def test_canonical_gather_rejects_invalid_owner_plan(ownership, capacity, query, world):
    with pytest.raises(ValueError):
        ownership.canonical_gather_indices(capacity, query, world)


@pytest.mark.parametrize("misordered", [False, True])
def test_timing_aggregates_slowest_rank_per_sample_before_round_medians(ownership, monkeypatch, misordered):
    import torch.distributed as dist

    local_device = (10, 100, 20, 200, 30, 300, 40, 400)
    remote_device = (100, 10, 200, 20, 300, 30, 400, 40)
    event_count, replay_counts = [0], {"tp": 0, "owner": 0}

    class Event:
        def __init__(self, enable_timing):
            assert enable_timing
            self.index = event_count[0]
            event_count[0] += 1

        def record(self):
            pass

        def synchronize(self):
            pass

        def elapsed_time(self, other):
            assert other.index == self.index + 1
            return local_device[self.index // 2] / 1000

    def gather(gathered, samples, group):
        remote = [dict(sample, device_us=remote_device[index]) for index, sample in enumerate(samples)]
        if misordered:
            remote[0]["name"] = "wrong_mode"
        gathered[:] = [samples, remote]

    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(stream=lambda root: nullcontext(), synchronize=lambda: None, Event=Event),
        raising=False,
    )
    monkeypatch.setattr(dist, "barrier", lambda group: None)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(dist, "all_gather_object", gather)
    counter = iter(range(100))
    monkeypatch.setattr(ownership.time, "perf_counter", lambda: next(counter) / 1000)

    def replay(name):
        replay_counts[name] += 1

    entries = {
        name: dict(root=object(), graph=SimpleNamespace(replay=lambda name=name: replay(name)))
        for name in ("tp", "owner")
    }
    if misordered:
        with pytest.raises(ValueError, match="ordering"):
            ownership.time_entries(entries, object(), rounds=2, repeats=2)
        return
    report = ownership.time_entries(entries, object(), rounds=2, repeats=2)
    assert [sample["device_us"] for sample in report["slowest_rank_samples"]] == [
        100,
        100,
        200,
        200,
        300,
        300,
        400,
        400,
    ]
    assert report["summary"]["tp"]["device_us"] == 250
    assert report["summary"]["owner"]["device_us"] == 250
    assert replay_counts == {"tp": 10, "owner": 10}


@pytest.mark.parametrize("dim", [-1, 0, 1])
def test_owner_core_reconstructs_qkv_channels_for_state_and_conv(owner_core, dim):
    widths, hidden, world = (2, 3, 6), 4, 2
    shards, references = [], [[] for _ in widths]
    for rank in range(world):
        pieces = []
        for component, width in enumerate(widths):
            piece = torch.arange(width * hidden).reshape(width, hidden) + rank * 1000 + component * 100
            if dim != 0:
                piece = piece.T
            pieces.append(piece)
            references[component].append(piece)
        shards.append(torch.cat(pieces, dim=dim))
    expected = torch.cat([torch.cat(component, dim=dim) for component in references], dim=dim)
    actual = owner_core.merge_qkv_shards(shards, widths, dim=dim)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert not torch.equal(actual, torch.cat(shards, dim=dim))


@pytest.mark.parametrize("capacity", [2, 6, 100])
def test_owner_state_rows_cover_global_speculative_pages_once_except_null(ownership, owner_core, capacity):
    query, world = 4, 2
    row_sets = []
    for rank in range(world):
        requests = owner_core.owner_request_ids(capacity, rank, world)
        assert requests == tuple(range(rank, capacity, world))
        rows = owner_core.state_row_ids(requests, query)
        assert rows[0] == 0
        assert len(rows) == 1 + capacity * query // world
        assert tuple(row - 1 for row in rows[1:]) == ownership.owner_token_rows(capacity, query, rank, world)
        row_sets.append(set(rows))
    assert row_sets[0] & row_sets[1] == {0}
    assert row_sets[0] | row_sets[1] == set(range(1 + capacity * query))


def test_owner_state_rows_preserve_null_and_noncontiguous_request_ids(owner_core):
    assert owner_core.state_row_ids((), 4) == (0,)
    assert owner_core.state_row_ids((0, 2), 4) == (0, 1, 2, 3, 4, 9, 10, 11, 12)


@pytest.mark.parametrize(
    "shapes,widths,dim",
    [
        ([], (2, 2, 2), 0),
        ([(6, 3)], (2, 2, 2), 0),
        ([(6, 3), (6, 3)], (2, 4), 0),
        ([(6, 3), (6, 3)], (0, 2, 4), 0),
        ([(6, 3), (6, 3)], (2, -1, 5), 0),
        ([(6, 3), (6, 3)], (2, 2, 1), 0),
        ([(6, 3), (6, 4)], (2, 2, 2), 0),
        ([(6, 3), (6, 3)], (2, 2, 2), 2),
        ([(6, 3), (6, 3)], (2, 2, 2), -3),
    ],
)
def test_owner_core_merge_rejects_invalid_qkv_contract(owner_core, shapes, widths, dim):
    with pytest.raises(ValueError):
        owner_core.merge_qkv_shards([torch.ones(shape) for shape in shapes], widths, dim=dim)


@pytest.mark.parametrize("capacity,rank,world", [(0, 0, 2), (99, 0, 2), (100, -1, 2), (100, 2, 2), (100, 0, 1)])
def test_owner_core_rejects_unsupported_owner_plan(owner_core, capacity, rank, world):
    with pytest.raises(ValueError):
        owner_core.owner_request_ids(capacity, rank, world)


@pytest.mark.parametrize("requests,query", [((1, 1), 4), ((-1,), 4), ((0.5,), 4), ((0,), 1), ((0,), 0)])
def test_owner_core_rejects_state_aliasing_and_wrong_query(owner_core, requests, query):
    with pytest.raises(ValueError):
        owner_core.state_row_ids(requests, query)


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize(
    "mode",
    [
        "entry_select",
        "exit_separate",
        "exit_packed",
        "exit_packed_contiguous",
        "roundtrip_separate",
        "roundtrip_packed",
        "roundtrip_packed_contiguous",
    ],
)
def test_owner_boundary_operations_against_independent_request_oracle(ownership, rank, mode):
    path = Path(__file__).resolve().parents[4] / "benchmarks" / "multi_graph_tp_gdn_owner_boundary.py"
    definition = next(node for node in ast.parse(path.read_text()).body if isinstance(node, ast.FunctionDef))
    functions = [node for node in definition.body if isinstance(node, ast.FunctionDef) and node.name != "refresh"]
    capacity, query, width, world = 100, 4, 7, 2
    tokens = capacity * query
    owner_order = [[token for token in range(tokens) if token // query % world == owner] for owner in range(world)]
    rank_major = [token for owner in owner_order for token in owner]
    inverse_oracle = sorted(range(tokens), key=lambda index: rank_major[index])
    torch.testing.assert_close(
        torch.tensor(ownership.canonical_gather_indices(capacity, query, world)), torch.tensor(inverse_oracle)
    )
    owner_rows = torch.tensor(ownership.owner_token_rows(capacity, query, rank, world))
    hidden = torch.empty(tokens, width)
    residual = torch.empty_like(hidden)
    local_hidden, local_residual = torch.empty(tokens // world, width), torch.empty(tokens // world, width)
    references, calls = [], []

    def gather(selected, dim):
        assert dim == 0
        packed = selected.shape[1] == width * 2
        if packed:
            fields = references
        else:
            # Each operation calls hidden first, residual second. This oracle
            # never applies the implementation's inverse to produce answers.
            fields = [references[len(calls)]]
        expected_local = torch.cat([field[owner_order[rank]] for field in fields], dim=-1)
        torch.testing.assert_close(selected, expected_local, rtol=0, atol=0)
        calls.append(selected.shape)
        return torch.cat([torch.cat([field[owner] for field in fields], dim=-1) for owner in owner_order], dim=0)

    namespace = dict(
        torch=torch,
        hidden=hidden,
        residual=residual,
        local_hidden=local_hidden,
        local_residual=local_residual,
        owner_rows=owner_rows,
        inverse=torch.tensor(inverse_oracle),
        group=SimpleNamespace(all_gather=gather),
        width=width,
    )
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
    pointers = [value.data_ptr() for value in (hidden, residual, local_hidden, local_residual)]
    for variant, live in enumerate((98, 97, 99, 98), start=1):
        rows, columns = torch.arange(tokens)[:, None], torch.arange(width)[None, :]
        reference_hidden = (rows * 103 + columns * 7 + variant * 3).float()
        reference_residual = (-rows * 211 - columns * 13 - variant * 5).float()
        reference_hidden[live * query :] = 0
        reference_residual[live * query :] = 0
        references[:] = [reference_hidden, reference_residual]
        hidden.copy_(reference_hidden)
        residual.copy_(reference_residual)
        local_hidden.copy_(reference_hidden[owner_order[rank]])
        local_residual.copy_(reference_residual[owner_order[rank]])
        before = [value.clone() for value in (hidden, residual, local_hidden, local_residual)]
        calls.clear()
        outputs = namespace[mode]()
        assert len(outputs) == 2
        if mode == "entry_select":
            expected = [value[owner_order[rank]] for value in references]
            assert calls == []
        else:
            expected = references
            assert len(calls) == (1 if "packed" in mode else 2)
        for actual, wanted in zip(outputs, expected, strict=True):
            torch.testing.assert_close(actual, wanted.float(), rtol=0, atol=0)
            assert actual.stride(-1) == 1
            if mode in ("exit_packed", "roundtrip_packed"):
                assert not actual.is_contiguous()
                assert actual.stride(0) == width * 2
            else:
                assert actual.is_contiguous()
                assert actual.stride(0) == width
            active = (
                torch.tensor(owner_order[rank]) < live * query
                if mode == "entry_select"
                else torch.arange(tokens) < live * query
            )
            assert not actual[~active].count_nonzero()
        if mode in ("exit_packed", "roundtrip_packed"):
            assert outputs[0].untyped_storage().data_ptr() == outputs[1].untyped_storage().data_ptr()
            assert outputs[1].storage_offset() - outputs[0].storage_offset() == width
        for actual, saved, address in zip(
            (hidden, residual, local_hidden, local_residual), before, pointers, strict=True
        ):
            torch.testing.assert_close(actual, saved, rtol=0, atol=0)
            assert actual.data_ptr() == address
