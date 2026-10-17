import ast
import importlib.util
import inspect
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as functional

from vllm_ascend.worker.multi_graph.replay_benchmark import ReplayWorkload


@dataclass(frozen=True)
class MetadataProvider:
    layer_name: str

    def resolve(self, mapping):
        return mapping[self.layer_name]


@pytest.fixture
def benchmark(monkeypatch):
    directory = Path(__file__).resolve().parents[4] / "benchmarks"
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location("tp_attention", directory / "multi_graph_tp_attention.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_contention_probe_pairs_identical_shapes_and_payload(benchmark):
    from multi_graph_tp_contention import probe_cases

    cases = {name: (start, stop, comm) for name, start, stop, comm in probe_cases(400, 200)}
    for shape in ("full", "left", "right"):
        assert cases[f"{shape}_alone"][:2] == cases[f"{shape}_with_comm"][:2]
        assert not cases[f"{shape}_alone"][2] and cases[f"{shape}_with_comm"][2]
    assert cases["comm_alone"] == (0, 0, True)
    for boundary in (0, 400, 401):
        with pytest.raises(ValueError, match="nonempty"):
            probe_cases(400, boundary)


@pytest.mark.parametrize("live", [97, 98, 99, 100])
def test_gdn_padding_and_split_preserve_global_state_identity(benchmark, live):
    from multi_graph_tp_gdn import request_state_values

    whole = request_state_values(0, 100, live)
    left, right = request_state_values(0, 50, live), request_state_values(50, 100, live)
    for key in ("blocks", "drafts", "lengths"):
        assert left[key] + right[key] == whole[key]
    assert whole["boundaries"][-1] == 4 * live
    assert whole["boundaries"][live:] == [4 * live] * (101 - live)
    assert whole["drafts"][live:] == [-1] * (100 - live)
    assert whole["blocks"][live:] == [[0] * 4] * (100 - live)
    ids = [block for blocks in whole["blocks"][:live] for block in blocks]
    assert len(set(ids)) == live * 4 and min(ids) == 1
    assert right["blocks"][0] == [201, 202, 203, 204]


@pytest.mark.parametrize(
    "flags",
    [
        ["--contention-probe"],
        ["--gdn-probe"],
        ["--gdn-probe", "--include-mlp", "--shared-qkv", "--query", "1"],
        ["--gdn-probe", "--contention-probe", "--include-mlp", "--shared-qkv"],
        ["--gdn-pipeline"],
        ["--gdn-shared"],
        ["--gdn-shared-mlp"],
        ["--gdn-shared-dense"],
        ["--gdn-pipeline", "--gdn-probe", "--include-mlp", "--shared-qkv"],
    ],
)
def test_probe_cli_rejects_incompatible_modes(benchmark, monkeypatch, tmp_path, flags):
    monkeypatch.setattr("sys.argv", ["benchmark", *flags, "--output", str(tmp_path / "out.json")])
    with pytest.raises(SystemExit) as error:
        benchmark.main()
    assert error.value.code == 2


def test_parallel_fia_keeps_all_dense_shapes_and_collectives(benchmark):
    result = benchmark.execution_layout("split_attention_parallel", 400, (200, 200), True, True)
    assert result["attention_tokens"] == [200, 200]
    assert result["attention_streams"] == ["root", "comm"]
    assert result["input_and_qkv_tokens"] == result["projection_tokens"] == [400]
    assert result["mlp"]["projection_tokens"] == [400]
    assert result["collective_order"] == ["input", "output", "down"]
    with pytest.raises(ValueError, match="shared QKV"):
        benchmark.execution_layout("split_attention_parallel", 400, (200, 200), True, False)


def test_gdn_runtime_refresh_keeps_capture_extent_when_live_batch_grows(benchmark):
    from multi_graph_tp_gdn import qualify_gdn

    function = next(
        node
        for node in ast.walk(ast.parse(inspect.getsource(qualify_gdn)))
        if isinstance(node, ast.FunctionDef) and node.name == "refresh"
    )
    intervals = dict(full=(0, 100), left=(0, 50), right=(50, 100))
    data = {}

    def metadata(live, name):
        start, stop = intervals[name]
        return SimpleNamespace(num_actual_tokens=max(0, min(live, stop) - start) * 4, live=live, pointer=name)

    namespace = dict(
        intervals=intervals,
        data_by_name=data,
        metadata=metadata,
        addresses={name: name for name in intervals},
        identity=lambda value: value.pointer,
    )
    exec(compile(ast.Module(body=[function], type_ignores=[]), "gdn_refresh", "exec"), namespace)
    for live in (98, 97, 99, 98):
        namespace["refresh"](live)
        assert {name: value.num_actual_tokens for name, value in data.items()} == dict(full=400, left=200, right=200)
        assert all(value.live == live for value in data.values())


@pytest.mark.parametrize("query", [1, 4])
@pytest.mark.parametrize("batch,variant", [(98, 0), (97, 1), (99, 2), (98, 3)])
def test_split_metadata_preserves_request_identity_and_physical_kv(benchmark, query, batch, variant):
    workload = ReplayWorkload(98, 100, query, 512, (68, 30))
    full = benchmark.request_values(workload, 0, 100, batch, variant)
    left = benchmark.request_values(workload, 0, 68, batch, variant)
    right = benchmark.request_values(workload, 68, 100, batch, variant)
    for field in ("lengths", "pages", "positions", "slots"):
        assert left[field] + right[field] == full[field]
    assert left["boundaries"] == [index * query for index in range(69)]
    assert right["boundaries"] == [index * query for index in range(33)]
    assert full["lengths"][batch:] == [0] * (100 - batch)
    assert full["slots"][batch * query :] == [-1] * ((100 - batch) * query)
    assert len(set(full["slots"][: batch * query])) == batch * query
    for request in range(batch):
        length = full["lengths"][request]
        for token in range(query):
            index = request * query + token
            position = length - query + token
            page = full["pages"][request][position // workload.block_size]
            assert full["positions"][index] == position
            assert full["slots"][index] == page * workload.block_size + position % workload.block_size


def test_changed_lengths_preserve_capacity_and_leave_prefix(benchmark):
    workload = ReplayWorkload(98, 100, 4, 512, (68, 30))
    initial = benchmark.request_values(workload, 68, 100, 98, 0)
    changed = benchmark.request_values(workload, 68, 100, 99, 1)
    assert initial["lengths"] != changed["lengths"]
    assert initial["positions"] != changed["positions"]
    assert all(length >= workload.query + 1 for length in changed["lengths"] if length)
    assert all(len(initial[field]) == len(changed[field]) for field in initial)
    assert initial["boundaries"] == changed["boundaries"]


@pytest.mark.parametrize("start,stop,batch", [(-1, 100, 98), (0, 101, 98), (68, 68, 98), (0, 100, 101), (0, 100, 0)])
def test_invalid_live_batch_or_slice_is_rejected(benchmark, start, stop, batch):
    workload = ReplayWorkload(98, 100, 4, 512, (68, 30))
    with pytest.raises(ValueError, match="captured capacity"):
        benchmark.request_values(workload, start, stop, batch, 0)


def test_short_prefix_is_rejected(benchmark):
    workload = ReplayWorkload(98, 100, 4, 64, (68, 30))
    with pytest.raises(ValueError, match="64 prefix tokens"):
        benchmark.request_values(workload, 0, 100, 98, 0)


def test_branch_source_filters_provider_and_preserves_capture_order(benchmark):
    attention = MetadataProvider("attention")
    other = MetadataProvider("other")
    full, left, right = object(), object(), object()
    input_data = [
        SimpleNamespace(mapping={"other": full}),
        SimpleNamespace(mapping={"attention": left}),
        SimpleNamespace(mapping={"attention": right}),
    ]
    graph = SimpleNamespace(
        tasks=[
            SimpleNamespace(provider=attention),
            SimpleNamespace(provider=other),
            SimpleNamespace(provider=MetadataProvider("attention")),
        ],
        task_lanes=[2, 0, 1],
    )
    source = benchmark.BranchSource(graph, input_data)
    assert source.get(attention) == (right, left)
    assert source.get(other) == (full,)
    assert source.get(MetadataProvider("missing")) == ()


def test_branch_source_binds_right_lane_local_query_lengths_and_current_kv(benchmark):
    workload = ReplayWorkload(98, 100, 4, 512, (68, 30))
    provider = MetadataProvider("attention")
    intervals = ((0, 100), (0, 68), (68, 100))
    input_data = [SimpleNamespace(mapping={}) for _ in intervals]
    graph = SimpleNamespace(tasks=[SimpleNamespace(provider=provider)] * 2, task_lanes=[1, 2])
    source = benchmark.BranchSource(graph, input_data)

    for batch, variant in ((98, 0), (99, 1)):
        for data, (start, stop) in zip(input_data, intervals, strict=True):
            values = benchmark.request_values(workload, start, stop, batch, variant)
            data.mapping["attention"] = {
                "actual_seq_lengths": values["boundaries"][1:],
                "actual_seq_lengths_kv": values["lengths"],
                "block_table": values["pages"],
            }
        left, right = source.get(provider)
        full = input_data[0].mapping["attention"]
        assert left is input_data[1].mapping["attention"]
        assert right is input_data[2].mapping["attention"]
        assert right["actual_seq_lengths"] == list(range(4, 129, 4))
        assert right["actual_seq_lengths_kv"] == full["actual_seq_lengths_kv"][68:]
        assert right["block_table"] == full["block_table"][68:]
        assert right["actual_seq_lengths_kv"][batch - 68 :] == [0] * (100 - batch)
        assert right["block_table"][0] == list(range(272, 276))


@pytest.mark.parametrize("inplace_residual", [False, True])
def test_local_mlp_preserves_returned_residual_and_matches_dense_tp_sum(benchmark, inplace_residual):
    torch.manual_seed(19)
    hidden, intermediate, tokens = 8, 6, 7
    value, residual = (torch.randn(tokens, hidden, dtype=torch.float64) for _ in range(2))
    gate, up = (torch.randn(intermediate, hidden, dtype=torch.float64) for _ in range(2))
    down = torch.randn(hidden, intermediate, dtype=torch.float64)
    norm_weight = torch.randn(hidden, dtype=torch.float64)
    expected_residual = value + residual
    normalized = expected_residual * torch.rsqrt(expected_residual.square().mean(-1, keepdim=True) + 1e-6)
    normalized *= norm_weight
    expected = (functional.silu(normalized @ gate.T) * (normalized @ up.T)) @ down.T
    returned_residuals = []

    def post_norm(part, prior):
        returned = prior.add_(part) if inplace_residual else prior + part
        returned_residuals.append(returned)
        normed = returned * torch.rsqrt(returned.square().mean(-1, keepdim=True) + 1e-6)
        return normed * norm_weight, returned

    def act(packed):
        gate_part, up_part = packed.chunk(2, dim=-1)
        return functional.silu(gate_part) * up_part

    # Independent row and TP partitions must compose into the same dense MLP.
    rank_outputs = []
    for rank_start in (0, 3):
        rank_outputs.append([])
        gate_up = torch.cat((gate[rank_start : rank_start + 3], up[rank_start : rank_start + 3]))
        for start, stop in ((0, 4), (4, tokens)):
            result, next_residual = benchmark.project_mlp(
                value[start:stop],
                residual[start:stop].clone(),
                post_norm,
                act,
                gate_up,
                down[:, rank_start : rank_start + 3],
            )
            assert next_residual is returned_residuals[-1]
            torch.testing.assert_close(next_residual, expected_residual[start:stop])
            rank_outputs[-1].append(result)
    observed = torch.cat(rank_outputs[0]) + torch.cat(rank_outputs[1])
    torch.testing.assert_close(observed, expected)
    assert benchmark.experiment_modes(True, False) == ("joined", "split_output_overlap", "split_mlp_overlap")
    assert benchmark.experiment_modes(False, False) == ("joined", "split_serial", "split_overlap")


@pytest.mark.parametrize("shared_qkv", [False, True])
def test_actual_split_mlp_schedule_waits_for_and_joins_returned_collective_storage(benchmark, shared_qkv):
    # Execute the production scheduling methods with CPU tensors and streams
    # that log dependencies. This checks ordering, not asynchronous NPU math.
    syntax = ast.parse(inspect.getsource(benchmark.experiment))
    definition = next(
        node for node in syntax.body[0].body if isinstance(node, ast.ClassDef) and node.name == "CapturedAttention"
    )
    log, returned = [], []
    current_stream = "root"
    input_names = ("input",) if shared_qkv else ("input_A", "input_B")
    collective_names = iter(input_names + ("output_A", "output_B", "down_A", "down_B"))

    class Stream:
        def __init__(self, name):
            self.name = name

        def wait_event(self, event):
            assert event.recorded
            log.append(("wait", event.name, self.name))

    class Event:
        def __init__(self, name):
            self.name, self.recorded = name, False

        def record(self, stream):
            self.recorded = True
            log.append(("record", self.name, stream.name))

    @contextmanager
    def stream(selected):
        nonlocal current_stream
        previous, current_stream = current_stream, selected.name
        try:
            yield
        finally:
            current_stream = previous

    def all_reduce(value):
        name = next(collective_names)
        log.append(("reduce", name, current_stream))
        result = value * 2
        assert result.data_ptr() != value.data_ptr()
        returned.append(result)
        return result

    torch.manual_seed(29)
    rows, boundary, hidden, intermediate = 7, 4, 8, 6
    down, residual = (torch.randn(rows, hidden, dtype=torch.float64) for _ in range(2))
    o_weight = torch.randn(hidden, hidden, dtype=torch.float64)
    gate_up_weight = torch.randn(intermediate * 2, hidden, dtype=torch.float64)
    down_weight = torch.randn(hidden, intermediate, dtype=torch.float64)
    qkv_weight = torch.randn(hidden, hidden, dtype=torch.float64)
    norm_results, transformed_slices = [], []
    forward_context = SimpleNamespace(capturing=True)
    intervals = ((0, rows), (0, boundary), (boundary, rows))
    input_data = [
        SimpleNamespace(start=start, stop=stop, positions=torch.arange(start, stop), forward_context=nullcontext)
        for start, stop in intervals
    ]

    def post_norm(value, prior):
        updated = prior + value
        return updated * torch.rsqrt(updated.square().mean(-1, keepdim=True) + 1e-6), updated

    def activation(packed):
        gate, up = packed.chunk(2, -1)
        return functional.silu(gate) * up

    def local_mlp(value, prior, gate_up, projection):
        log.append(("mlp", value.shape[0], current_stream))
        return benchmark.project_mlp(value, prior, post_norm, activation, gate_up, projection)

    def input_norm(value, prior):
        log.append(("input_norm", value.shape[0], current_stream))
        result = post_norm(value, prior)
        norm_results.append(result)
        return result

    def transform_qkv(projection, positions, cos_sin):
        assert cos_sin is None
        transformed_slices.append((projection, positions))
        return projection + positions[:, None] * 0.125

    def attend_projected(projection, data):
        assert forward_context.capturing
        assert projection.shape[0] == data.stop - data.start
        return projection

    namespace = dict(
        adjacent=None,
        torch=SimpleNamespace(npu=SimpleNamespace(stream=stream)),
        functional=functional,
        args=SimpleNamespace(include_mlp=True, shared_rope=False, shared_qkv=shared_qkv),
        group=SimpleNamespace(all_reduce=all_reduce),
        rows=rows,
        boundary=boundary,
        down=down,
        residual=residual,
        output_weight_nz=o_weight,
        gate_up_weight_timed=gate_up_weight,
        down_weight_nz=down_weight,
        local_mlp=local_mlp,
        weight_nz=qkv_weight,
        layer=SimpleNamespace(input_layernorm=input_norm),
        input_data=input_data,
        workload=SimpleNamespace(query=1),
        get_forward_context=lambda: forward_context,
        transform_qkv=transform_qkv,
        attend_projected=attend_projected,
    )
    exec(compile(ast.Module(body=[definition], type_ignores=[]), benchmark.__file__, "exec"), namespace)
    cls = namespace["CapturedAttention"]
    entry = cls.__new__(cls)
    entry.mode = "split_mlp_overlap"
    entry.graph, entry.workspaces = SimpleNamespace(lane=0), ()
    entry.root, entry.comm = Stream("root"), Stream("comm")
    entry.events = [Event(name) for name in ("fork", "input_A", "input_B")]
    entry.output_events = [Event(name) for name in ("OA_ready", "OA_done", "OB_ready", "OB_done")]
    entry.mlp_events = [Event(name) for name in ("DA_ready", "DA_done", "DB_ready", "DB_done")]
    for field in (
        "working",
        "residual_working",
        "attention_output",
        "attention_residual",
        "joined_output",
        "joined_residual",
    ):
        setattr(entry, field, torch.empty_like(down))

    def branch(lane, value):
        start, stop = (0, boundary) if lane == 1 else (boundary, rows)
        entry.attention_output[start:stop].copy_(value + 5)
        entry.attention_residual[start:stop].copy_(value + entry.residual_working[start:stop])

    if not shared_qkv:
        entry.branch = branch
    result, final_residual = entry.operation()
    if shared_qkv:
        expected_input_residual = 2 * down + residual
        normalized_input = expected_input_residual * torch.rsqrt(
            expected_input_residual.square().mean(-1, keepdim=True) + 1e-6
        )
        expected_qkv = normalized_input @ qkv_weight.T
        torch.testing.assert_close(entry.shared_qkv_output, expected_qkv)
        assert len(norm_results) == 1
        torch.testing.assert_close(norm_results[0][1], expected_input_residual)
        assert entry.attention_residual.data_ptr() != norm_results[0][1].data_ptr()
        torch.testing.assert_close(entry.attention_residual, expected_input_residual)
        assert len(transformed_slices) == 2
        for (projection, positions), (start, stop) in zip(transformed_slices, intervals[1:], strict=True):
            assert (
                projection.data_ptr() == entry.shared_qkv_output.data_ptr() + start * hidden * projection.element_size()
            )
            torch.testing.assert_close(projection, expected_qkv[start:stop])
            torch.testing.assert_close(positions, torch.arange(start, stop))
        attended = expected_qkv + torch.arange(rows)[:, None] * 0.125
        assert ("input_norm", rows, "root") in log
        entry.output = result, final_residual
        assert entry.identity()[-1] == entry.shared_qkv_output.data_ptr()
    else:
        attended = 2 * down + 5
    attention_output = functional.linear(attended, o_weight) * 2
    expected_residual = 2 * down + residual + attention_output
    normalized = expected_residual * torch.rsqrt(expected_residual.square().mean(-1, keepdim=True) + 1e-6)
    packed = normalized @ gate_up_weight.T
    gate, up = packed.chunk(2, -1)
    expected = ((functional.silu(gate) * up) @ down_weight.T) * 2
    torch.testing.assert_close(result, expected)
    torch.testing.assert_close(final_residual, expected_residual)
    assert result is entry.joined_output and final_residual is entry.joined_residual
    assert entry.output_partitions[0][1] is returned[-4]
    assert entry.output_partitions[1][1] is returned[-3]
    assert entry.mlp_partitions[0][1] is returned[-2]
    assert entry.mlp_partitions[1][1] is returned[-1]
    expected_collectives = [("reduce", name, "root" if shared_qkv else "comm") for name in input_names]
    expected_collectives += [("reduce", name, "comm") for name in ("output_A", "output_B", "down_A", "down_B")]
    assert [item for item in log if item[0] == "reduce"] == expected_collectives
    expected_order = [
        ("reduce", "output_B", "comm"),
        ("wait", "OA_done", "root"),
        ("mlp", boundary, "root"),
        ("reduce", "down_A", "comm"),
        ("wait", "OB_done", "root"),
        ("mlp", rows - boundary, "root"),
        ("reduce", "down_B", "comm"),
        ("wait", "DA_done", "root"),
        ("wait", "DB_done", "root"),
    ]
    indices = [log.index(item) for item in expected_order]
    assert indices == sorted(indices)


@pytest.mark.parametrize("parts", [1, 2])
def test_paired_gate_up_channels_preserve_activation_and_down(benchmark, parts):
    torch.manual_seed(73)
    x = torch.randn(7, 8, dtype=torch.float64)
    residual = torch.randn_like(x)
    packed = torch.randn(12, 8, dtype=torch.float64)
    down = torch.randn(8, 6, dtype=torch.float64)

    def norm(value, prior):
        updated = value + prior
        return updated * torch.rsqrt(updated.square().mean(-1, keepdim=True) + 1e-6), updated

    def act(value):
        gate, up = value.chunk(2, dim=-1)
        return functional.silu(gate) * up

    expected = benchmark.project_mlp(x, residual, norm, act, packed, down)
    weights = benchmark.partition_gate_up(packed, parts)
    normalized, _ = norm(x, residual)
    intermediate = torch.cat([act(functional.linear(normalized, w)) for w in weights], dim=-1)
    torch.testing.assert_close(intermediate, act(functional.linear(normalized, packed)))
    observed = benchmark.project_mlp(x, residual, norm, act, weights if parts > 1 else weights[0], down)
    torch.testing.assert_close(observed, expected)
    split = [benchmark.project_mlp(x[a:b], residual[a:b], norm, act, weights, down) for a, b in ((0, 4), (4, 7))]
    for i in (0, 1):
        torch.testing.assert_close(torch.cat([p[i] for p in split]), expected[i])
    with pytest.raises(ValueError, match="equally divisible"):
        benchmark.partition_gate_up(packed, 4)


@pytest.mark.parametrize("overlap", [False, True])
def test_adjacent_pipeline_orders_collectives_and_drains_both_lanes(benchmark, overlap):
    syntax = ast.parse(inspect.getsource(benchmark.experiment))
    definition = next(
        node for node in syntax.body[0].body if isinstance(node, ast.ClassDef) and node.name == "CapturedAttention"
    )
    log = []

    def run(value, prior, name, local):
        assert local
        log.append(f"prefix_{name}")
        return value + 1, prior + 2

    def mlp(value, prior):
        return value + 3, prior + 4

    namespace = dict(adjacent=SimpleNamespace(run=run, local_mlp=mlp), boundary=2)
    exec(compile(ast.Module(body=[definition], type_ignores=[]), benchmark.__file__, "exec"), namespace)
    cls = namespace["CapturedAttention"]
    entry = cls.__new__(cls)
    entry.gdn_overlap = overlap
    entry.root = SimpleNamespace(wait_event=lambda event: log.append(event))
    entry.mlp_events = ("DA_ready", "DA_done", "DB_ready", "DB_done")
    entry.gdn_events = tuple(f"gdn_{i}" for i in range(8))

    def reduce(value, index):
        log.append(f"reduce_{index}")
        return value * 2

    entry.gdn_reduce = reduce
    entry.joined_output = torch.empty(4, 3)
    entry.joined_residual = torch.empty(4, 3)
    first, second = torch.full((2, 3), 2.0), torch.full((2, 3), 5.0)
    prior = torch.ones(2, 3)
    output, residual = entry.finish_adjacent((None, first, prior), (None, second, prior))
    torch.testing.assert_close(output, torch.cat((first, second)) * 4 + 10)
    torch.testing.assert_close(residual, torch.full((4, 3), 7.0))
    assert [item for item in log if item.startswith("reduce")] == [f"reduce_{i}" for i in range(4)]
    assert log.index("DB_done") < log.index("reduce_0")
    assert (log.index("prefix_left") < log.index("DB_done")) == overlap
    assert log.index("gdn_1") < log.index("reduce_2")
    assert log.index("gdn_3") < log.index("reduce_3")
    assert log[-2:] == ["gdn_5", "gdn_7"]


@pytest.mark.parametrize("weight_format", ["ND", "NZ"])
@pytest.mark.parametrize("lowercase", [False, True])
def test_gate_up_cli_normalizes_format_and_preserves_common_controls(
    benchmark, monkeypatch, tmp_path, weight_format, lowercase
):
    weight_format = weight_format.lower() if lowercase else weight_format
    seen = []
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setattr(
        "sys.argv",
        [
            "benchmark",
            "--include-mlp",
            "--shared-qkv",
            "--gate-up-format",
            weight_format,
            "--gate-up-partitions",
            "2",
            "--output",
            str(tmp_path / "result.json"),
        ],
    )
    monkeypatch.setattr(
        benchmark,
        "select_workload",
        lambda args: dict(
            pair_tokens=[200, 200],
            pair_requests=[50, 50],
            batch=98,
            capacity_requests=100,
        ),
    )
    monkeypatch.setattr(benchmark, "run_qwen35", lambda args, workload, **kwargs: seen.append((args, workload)))
    benchmark.main()
    args, workload = seen[0]
    assert args.gate_up_format == weight_format.upper()
    assert args.gate_up_partitions == 2 and args.shared_qkv and args.include_mlp
    assert benchmark.experiment_modes(args.include_mlp, args.split_output_overlap) == (
        "joined",
        "split_output_overlap",
        "split_mlp_overlap",
    )
    assert workload.capacity == 100
