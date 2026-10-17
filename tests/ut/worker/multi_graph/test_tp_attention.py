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


@pytest.mark.parametrize("serial, parallel", [(False, False), (True, False), (False, True)])
@pytest.mark.parametrize("inplace_residual", [False, True])
def test_gdn_core_pipeline_keeps_dense_shapes_and_joins_returned_reductions(
    benchmark, serial, parallel, inplace_residual
):
    syntax = ast.parse(inspect.getsource(benchmark.experiment))
    definition = next(
        node for node in syntax.body[0].body if isinstance(node, ast.ClassDef) and node.name == "CapturedAttention"
    )
    active, log, reductions = "root", [], []

    class Stream:
        def __init__(self, name):
            self.name = name

        def wait_event(self, event):
            assert event.recorded, f"{self.name} consumes unrecorded {event.name}"
            log.append(("wait", event.name, self.name))

    class Event:
        def __init__(self, name):
            self.name, self.recorded = name, False

        def record(self, selected):
            assert selected.name == active
            self.recorded = True
            log.append(("record", self.name, active))

    @contextmanager
    def stream(selected):
        nonlocal active
        previous, active = active, selected.name
        try:
            yield
        finally:
            active = previous

    torch.manual_seed(881)
    rows, boundary, hidden, width = 7, 3, 4, 6
    value, prior = (torch.randn(rows, hidden, dtype=torch.float64) for _ in range(2))
    gate_up = torch.randn(width * 2, hidden, dtype=torch.float64)
    down = torch.randn(hidden, width, dtype=torch.float64)
    expected_residual = (value + 9) * 2 + prior + 2
    normalized = expected_residual * torch.rsqrt(expected_residual.square().mean(-1, keepdim=True) + 1e-6)
    gate, up = functional.linear(normalized, gate_up).chunk(2, dim=-1)
    expected = functional.linear(functional.silu(gate) * up, down) * 2
    global_ids = torch.arange(1, rows + 1)
    metadata = tuple(
        SimpleNamespace(global_ids=global_ids[start:stop]) for start, stop in ((0, boundary), (boundary, rows))
    )
    metadata_before = tuple(item.global_ids.clone() for item in metadata)
    state = torch.arange(rows + 3, dtype=torch.float64)
    expected_state = state.clone()
    expected_state[global_ids] += (value + 1).sum(-1)

    def project(selected, residual, name):
        assert active == "root" and name == "full"
        assert selected is value and residual is prior
        log.append(("project", rows, active))
        return SimpleNamespace(value=selected + 1, residual=residual + 2, name=name)

    def partition(projected, boundaries):
        assert active == "root" and boundaries == (0, boundary, rows)
        assert projected.name == "full"
        log.append(("partition", boundaries, active))
        return tuple(
            SimpleNamespace(value=projected.value[start:stop], metadata=metadata[lane], lane=lane, normalized=None)
            for lane, (start, stop) in enumerate(zip(boundaries, boundaries[1:]))
        )

    def core(phase):
        assert active == (f"core_{phase.lane}" if parallel else "root")
        assert phase.metadata is metadata[phase.lane]
        assert phase.normalized is None
        state.index_add_(0, phase.metadata.global_ids, phase.value.sum(-1))
        phase.normalized = phase.value + 3
        log.append(("core", phase.lane, active))

    def output(phase):
        assert active == "root" and phase.normalized is not None
        log.append(("output", phase.lane, active))
        return phase.normalized + 5, None

    def all_reduce(selected):
        assert active == "comm"
        index = len(reductions)
        result = selected * 2
        assert result.data_ptr() != selected.data_ptr()
        reductions.append(result)
        log.append(("reduce", index, active))
        return result

    def prepare_mlp(selected, residual):
        assert active == "root" and selected.shape == (rows, hidden)
        torch.testing.assert_close(selected, (value + 9) * 2)
        log.append(("GateUp", rows, active))
        updated = residual.add_(selected) if inplace_residual else residual + selected
        normalized = updated * torch.rsqrt(updated.square().mean(-1, keepdim=True) + 1e-6)
        gate, up = functional.linear(normalized, gate_up).chunk(2, dim=-1)
        return functional.silu(gate) * up, updated

    def project_down(selected):
        assert active == "root"
        lane = int(selected.shape[0] != boundary)
        log.append(("Down", lane, active))
        return functional.linear(selected, down)

    namespace = dict(
        torch=SimpleNamespace(npu=SimpleNamespace(stream=stream)),
        adjacent=SimpleNamespace(
            phases=SimpleNamespace(project=project, partition=partition, core=core, output=output),
            prepare_mlp=prepare_mlp,
            project_down=project_down,
        ),
        group=SimpleNamespace(all_reduce=all_reduce),
        boundary=boundary,
        rows=rows,
    )
    exec(compile(ast.Module(body=[definition], type_ignores=[]), benchmark.__file__, "exec"), namespace)
    entry = namespace["CapturedAttention"].__new__(namespace["CapturedAttention"])
    entry.root, entry.comm = Stream("root"), Stream("comm")
    entry.gdn_core_mode = "split_gdn_core_serial" if serial else "split_gdn_core_pipeline"
    if parallel:
        entry.gdn_parallel_core = True
        entry.gdn_core_streams = [Stream(f"core_{lane}") for lane in range(2)]
        entry.gdn_core_events = [Event(f"core_event_{index}") for index in range(3)]
    entry.gdn_events = [Event(f"gdn_{index}") for index in range(8)]
    entry.gdn_joined = torch.empty_like(value)
    entry.joined_output, entry.joined_residual = torch.empty_like(value), torch.empty_like(prior)
    actual, residual = entry.finish_core_pipeline(value, prior)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(residual, expected_residual)
    torch.testing.assert_close(state, expected_state, rtol=0, atol=0)
    for item, before in zip(metadata, metadata_before):
        torch.testing.assert_close(item.global_ids, before, rtol=0, atol=0)
    assert actual is entry.joined_output and residual is entry.joined_residual
    assert len(reductions) == 4 and all(event.recorded for event in entry.gdn_events)
    assert all(entry.gdn_partitions[index] is reductions[index] for index in range(2))
    assert all(entry.gdn_partitions[index + 4] is reductions[index + 2] for index in range(2))
    assert [item for item in log if item[0] == "project"] == [("project", rows, "root")]
    assert [item for item in log if item[0] == "GateUp"] == [("GateUp", rows, "root")]
    assert [item[1] for item in log if item[0] == "reduce"] == [0, 1, 2, 3]
    assert [item[1] for item in log if item[0] == "Down"] == [0, 1]
    for reduction, next_work in (
        (0, ("core", 1, "core_1" if parallel else "root")),
        (2, ("Down", 1, "root")),
    ):
        first_wait = log.index(("wait", f"gdn_{2 * reduction + 1}", "root"))
        assert (first_wait < log.index(next_work)) == serial
    assert log.index(("wait", "gdn_3", "root")) < log.index(("GateUp", rows, "root"))
    assert log[-1] == ("wait", "gdn_7", "root")
    if parallel:
        assert all(event.recorded for event in entry.gdn_core_events)
        assert log.index(("record", "core_event_0", "root")) < log.index(("core", 0, "core_0"))
        for lane in range(2):
            order = [
                ("wait", "core_event_0", f"core_{lane}"),
                ("core", lane, f"core_{lane}"),
                ("record", f"core_event_{lane + 1}", f"core_{lane}"),
                ("wait", f"core_event_{lane + 1}", "root"),
                ("output", lane, "root"),
                ("reduce", lane, "comm"),
            ]
            assert [log.index(item) for item in order] == sorted(log.index(item) for item in order)
        assert log.index(("core", 1, "core_1")) < log.index(("output", 0, "root"))
        assert log.index(("reduce", 0, "comm")) < log.index(("wait", "core_event_2", "root"))


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
        ["--gdn-wavefront"],
        ["--gdn-wavefront", "--gdn-pipeline", "--gdn-shared", "--include-mlp", "--shared-qkv"],
        ["--gdn-phases"],
        ["--gdn-phases", "--gdn-pipeline", "--gdn-shared", "--include-mlp", "--shared-qkv"],
        ["--gdn-phases", "--gdn-pipeline", "--gdn-wavefront", "--include-mlp", "--shared-qkv"],
        ["--gdn-core-pipeline"],
        ["--gdn-core-concurrency"],
        ["--gdn-core-pipeline", "--gdn-pipeline", "--include-mlp", "--shared-qkv"],
        ["--gdn-core-pipeline", "--gdn-shared", "--include-mlp", "--shared-qkv"],
        [
            "--gdn-core-pipeline",
            "--gdn-pipeline",
            "--gdn-shared",
            "--gdn-shared-mlp",
            "--include-mlp",
            "--shared-qkv",
        ],
        [
            "--gdn-core-pipeline",
            "--gdn-pipeline",
            "--gdn-shared",
            "--gate-up-partitions",
            "2",
            "--include-mlp",
            "--shared-qkv",
        ],
        ["--down-pipeline"],
        ["--down-pipeline", "--gdn-pipeline", "--include-mlp", "--shared-qkv"],
        ["--down-pipeline", "--gdn-pipeline", "--gdn-shared", "--gdn-shared-mlp", "--include-mlp", "--shared-qkv"],
        ["--down-pipeline", "--gdn-pipeline", "--gdn-shared", "--gdn-wavefront", "--include-mlp", "--shared-qkv"],
        ["--numerical-assessment"],
        ["--gdn-pipeline", "--gdn-probe", "--include-mlp", "--shared-qkv"],
        ["--parallel-probe"],
        ["--parallel-probe", "--include-mlp", "--shared-qkv", "--query", "1"],
        ["--parallel-probe", "--include-mlp", "--shared-qkv", "--trace"],
        ["--parallel-probe", "--include-mlp", "--shared-qkv", "--contention-probe"],
        ["--parallel-probe", "--include-mlp", "--shared-qkv", "--gdn-probe"],
        ["--parallel-probe", "--include-mlp", "--shared-qkv", "--gdn-pipeline"],
        ["--parallel-probe", "--include-mlp", "--shared-qkv", "--gdn-pipeline", "--gdn-wavefront"],
        ["--parallel-probe", "--include-mlp", "--shared-qkv", "--gate-up-format", "ND"],
        ["--parallel-probe", "--include-mlp", "--shared-qkv", "--gate-up-partitions", "2"],
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


def test_down_pipeline_keeps_gate_up_whole_and_only_splits_down(benchmark):
    result = benchmark.execution_layout("split_down_overlap", 400, (200, 200), True, True)
    assert result["input_and_qkv_tokens"] == [400]
    assert result["projection_tokens"] == result["reduction_tokens"] == [200, 200]
    assert result["mlp"]["projection_tokens"] == [400]
    assert result["mlp"]["down_projection_tokens"] == result["mlp"]["reduction_tokens"] == [200, 200]
    assert result["mlp"]["real_down_all_reduces_per_replay"] == 2
    assert result["mlp"]["hidden_and_residual_join_timed"]
    assert result["collective_order"] == ["input", "output_A", "output_B", "down_A", "down_B"]
    for include_mlp, shared_qkv in ((False, False), (True, False)):
        with pytest.raises(ValueError, match="shared QKV"):
            benchmark.execution_layout("split_down_overlap", 400, (200, 200), include_mlp, shared_qkv)


def test_gdn_runtime_refresh_keeps_capture_extent_when_live_batch_grows(benchmark):
    from multi_graph_tp_gdn import qualify_gdn

    function = next(
        node
        for node in ast.walk(ast.parse(inspect.getsource(qualify_gdn)))
        if isinstance(node, ast.FunctionDef) and node.name == "refresh"
    )
    intervals = dict(full=(0, 100), left=(0, 50), right=(50, 100))
    data = {}

    def metadata(live, name, accepted=1):
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


def test_numerical_assessment_reports_original_failures_without_hiding_them(benchmark):
    from multi_graph_numerical_assessment import error_metrics

    expected = torch.ones(128)
    actual = expected.clone()
    actual[7] += 0.125
    result = error_metrics(actual, expected)
    assert result["original_tolerance_failures"] == 1
    assert result["original_tolerance_failure_fraction"] == 1 / 128
    assert result["max_abs"] == 0.125
    assert result["normalized_rmse"] == pytest.approx(0.125 / 128**0.5)
    assert error_metrics(expected, expected)["normalized_rmse"] == 0
    for value in (float("nan"), float("inf")):
        with pytest.raises(ValueError, match="nonfinite"):
            error_metrics(torch.tensor([value]), torch.ones(1))
    with pytest.raises(ValueError, match="matching shapes"):
        error_metrics(torch.ones(2), torch.ones(1))


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
    entry.gdn_compute = None
    entry.gdn_projection_done = None
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


@pytest.mark.parametrize("overlap,resource_aware", [(False, False), (True, False), (True, True)])
def test_gdn_wavefront_issues_prefix_before_mlp_b_and_preserves_collective_order(benchmark, overlap, resource_aware):
    syntax = ast.parse(inspect.getsource(benchmark.experiment))
    definition = next(
        node for node in syntax.body[0].body if isinstance(node, ast.ClassDef) and node.name == "CapturedAttention"
    )
    log = []
    active = "root"
    names = iter(("down_A", "down_B", "gdn_output_A", "gdn_output_B", "gdn_down_A", "gdn_down_B"))

    class Stream:
        def __init__(self, name):
            self.name = name

        def wait_event(self, event):
            assert event.recorded, f"{self.name} consumes an unrecorded {event.name}"
            log.append(("wait", event.name, self.name))

    class Event:
        def __init__(self, name, recorded=False):
            self.name, self.recorded = name, recorded

        def record(self, selected):
            assert selected.name == active
            self.recorded = True
            log.append(("record", self.name, selected.name))

    @contextmanager
    def stream(selected):
        nonlocal active
        previous, active = active, selected.name
        try:
            yield
        finally:
            active = previous

    def all_reduce(value):
        assert active == "comm"
        log.append(("reduce", next(names), active))
        return value * 2

    def local_mlp(value, prior, gate_up, projection):
        assert active == "root"
        log.append(("mlp", int(value[0, 0]), active))
        return value + 10, prior + 20

    def prefix(value, prior, name, local, projection_done=None):
        assert local
        log.append(("prefix", name, active))
        log.append(("projection", name, active))
        if projection_done is not None:
            projection_done.record(entry.gdn_compute)
        log.append(("core", name, active))
        return value + 1, prior + 2

    def next_mlp(value, prior):
        assert active == "root"
        return value + 3, prior + 4

    namespace = dict(
        torch=SimpleNamespace(npu=SimpleNamespace(stream=stream)),
        adjacent=SimpleNamespace(run=prefix, local_mlp=next_mlp),
        args=SimpleNamespace(gdn_shared=False),
        group=SimpleNamespace(all_reduce=all_reduce),
        local_mlp=local_mlp,
        gate_up_weight_timed=None,
        down_weight_nz=None,
        boundary=2,
        rows=4,
    )
    exec(compile(ast.Module(body=[definition], type_ignores=[]), benchmark.__file__, "exec"), namespace)
    cls = namespace["CapturedAttention"]
    entry = cls.__new__(cls)
    entry.gdn_overlap = overlap
    entry.root, entry.comm, entry.gdn_compute = (Stream(name) for name in ("root", "comm", "gdn_compute"))
    entry.early_gdn = None
    entry.gdn_prefix_done = Event("prefix_done")
    entry.gdn_projection_done = Event("projection_done") if resource_aware else None
    entry.output_events = [Event(name, True) for name in ("OA_ready", "OA_done", "OB_ready", "OB_done")]
    entry.mlp_events = [Event(name) for name in ("DA_ready", "DA_done", "DB_ready", "DB_done")]
    entry.gdn_events = [Event(f"gdn_{index}") for index in range(8)]
    entry.attention_residual = torch.ones(4, 3)
    entry.joined_output, entry.joined_residual = torch.empty(4, 3), torch.empty(4, 3)
    first, second = torch.full((2, 3), 2.0), torch.full((2, 3), 5.0)
    output, residual = entry.finish_split_mlp((None, first), (None, second))
    torch.testing.assert_close(output, torch.cat((first, second)) * 8 + 90)
    torch.testing.assert_close(residual, torch.full((4, 3), 27.0))
    reductions = [item[1] for item in log if item[0] == "reduce"]
    assert reductions == ["down_A", "down_B", "gdn_output_A", "gdn_output_B", "gdn_down_A", "gdn_down_B"]
    first_prefix = ("prefix", "left", "gdn_compute" if overlap else "root")
    assert log.count(first_prefix) == 1
    assert log.index(("wait", "DB_done", "root")) < log.index(("reduce", "gdn_output_A", "comm"))
    if overlap:
        ordered = [
            ("reduce", "down_A", "comm"),
            ("wait", "DA_done", "gdn_compute"),
            first_prefix,
            ("record", "prefix_done", "gdn_compute"),
            ("mlp", 5, "root"),
            ("reduce", "down_B", "comm"),
            ("wait", "prefix_done", "root"),
            ("wait", "DB_done", "root"),
            ("reduce", "gdn_output_A", "comm"),
        ]
        positions = [log.index(item) for item in ordered]
        assert positions == sorted(positions)
        if resource_aware:
            projection_order = [
                ("projection", "left", "gdn_compute"),
                ("record", "projection_done", "gdn_compute"),
                ("core", "left", "gdn_compute"),
                ("wait", "projection_done", "root"),
                ("mlp", 5, "root"),
            ]
            positions = [log.index(item) for item in projection_order]
            assert positions == sorted(positions)
        else:
            assert not any(item[1] == "projection_done" for item in log)
    else:
        assert log.index(("wait", "DB_done", "root")) < log.index(first_prefix)
        assert not any(item[-1] == "gdn_compute" for item in log)
        assert not entry.gdn_prefix_done.recorded
    assert log[-2:] == [("wait", "gdn_5", "root"), ("wait", "gdn_7", "root")]


@pytest.mark.parametrize("unpacked", [False, True])
@pytest.mark.parametrize("failure", [None, "core", "missing_projection"])
def test_gdn_projection_hook_precedes_core_and_preserves_existing_hooks(benchmark, unpacked, failure):
    from multi_graph_tp_gdn import qualify_gdn

    function = next(
        node
        for node in ast.walk(ast.parse(inspect.getsource(qualify_gdn)))
        if isinstance(node, ast.FunctionDef) and node.name == "run"
    )
    log = []
    selected_stream = object()

    class Projection(torch.nn.Module):
        def __init__(self, name):
            super().__init__()
            self.name = name

        def forward(self, value):
            log.append(self.name)
            return value + 1

    class GDN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.prefix = "gdn"
            self.out_proj = SimpleNamespace(reduce_results=True)
            self.in_proj_ba = Projection("projection_ba")
            if unpacked:
                self.in_proj_qkv = Projection("projection_qkv")
                self.in_proj_z = Projection("projection_z")
            else:
                self.in_proj_qkvz = Projection("projection_qkvz")

        def forward(self, *, hidden_states):
            assert not self.out_proj.reduce_results
            if unpacked:
                self.in_proj_qkv(hidden_states)
                self.in_proj_ba(hidden_states)
                if failure != "missing_projection":
                    self.in_proj_z(hidden_states)
            else:
                self.in_proj_qkvz(hidden_states)
                if failure != "missing_projection":
                    self.in_proj_ba(hidden_states)
            log.append("core")
            if failure == "core":
                raise RuntimeError("core failed")
            return hidden_states + 7

    gdn = GDN()
    last_projection = gdn.in_proj_z if unpacked else gdn.in_proj_ba

    def existing_hook(module, inputs, output):
        assert module is last_projection
        assert len(module._forward_hooks) == 2
        log.append("existing_hook")

    original_handle = last_projection.register_forward_hook(existing_hook)
    original_hooks = {name: dict(module._forward_hooks) for name, module in gdn.named_modules()}

    def record(stream):
        assert stream is selected_stream
        log.append("projection_done")

    namespace = dict(
        torch=SimpleNamespace(npu=SimpleNamespace(current_stream=lambda: selected_stream)),
        intervals={"left": (0, 1)},
        set_ascend_forward_context=lambda *args, **kwargs: nullcontext(),
        gdn=gdn,
        data_by_name={"left": object()},
        config=object(),
        CUDAGraphMode=SimpleNamespace(NONE=object()),
        model=object(),
        positions=torch.arange(4),
        layer=SimpleNamespace(input_layernorm=lambda value, prior: (value + prior, prior + 2)),
    )
    exec(compile(ast.Module(body=[function], type_ignores=[]), "gdn_signal_run", "exec"), namespace)
    value, prior = torch.ones(4, 3), torch.full((4, 3), 2.0)
    if failure == "core":
        with pytest.raises(RuntimeError, match="core failed"):
            namespace["run"](value, prior, "left", local=True, projection_done=SimpleNamespace(record=record))
    elif failure == "missing_projection":
        with pytest.raises(ValueError, match="did not publish"):
            namespace["run"](value, prior, "left", local=True, projection_done=SimpleNamespace(record=record))
    else:
        projected, residual = namespace["run"](
            value, prior, "left", local=True, projection_done=SimpleNamespace(record=record)
        )
        torch.testing.assert_close(projected, torch.full((4, 3), 10.0))
        torch.testing.assert_close(residual, torch.full((4, 3), 4.0))
    expected = ["projection_qkv", "projection_ba"] if unpacked else ["projection_qkvz"]
    if failure != "missing_projection":
        expected += ["projection_z" if unpacked else "projection_ba", "existing_hook", "projection_done"]
    assert log == expected + ["core"]
    assert gdn.out_proj.reduce_results
    assert {name: dict(module._forward_hooks) for name, module in gdn.named_modules()} == original_hooks
    assert last_projection._forward_hooks[original_handle.id] is existing_hook


@pytest.mark.parametrize("zipper", [False, True])
def test_phased_gdn_overlaps_only_core_with_down_and_drains_all_collectives(benchmark, zipper):
    syntax = ast.parse(inspect.getsource(benchmark.experiment))
    definition = next(
        node for node in syntax.body[0].body if isinstance(node, ast.ClassDef) and node.name == "CapturedAttention"
    )
    log, reductions = [], []
    active = "root"
    names = iter(("down_A", "down_B", "gdn_output_A", "gdn_output_B", "gdn_down_A", "gdn_down_B"))

    class Stream:
        def __init__(self, name):
            self.name = name

        def wait_event(self, event):
            assert event.recorded, f"{self.name} consumes unrecorded {event.name}"
            log.append(("wait", event.name, self.name))

    class Event:
        def __init__(self, name, recorded=False):
            self.name, self.recorded = name, recorded

        def record(self, selected):
            assert selected.name == active
            self.recorded = True
            log.append(("record", self.name, active))

    @contextmanager
    def stream(selected):
        nonlocal active
        previous, active = active, selected.name
        try:
            yield
        finally:
            active = previous

    torch.manual_seed(149)
    rows, boundary, hidden, intermediate = 5, 2, 4, 3
    source, prior = (torch.randn(rows, hidden, dtype=torch.float64) for _ in range(2))
    gate_up = torch.randn(intermediate * 2, hidden, dtype=torch.float64)
    down = torch.randn(hidden, intermediate, dtype=torch.float64)

    def norm(value, residual):
        assert active == "root"
        log.append(("norm", value.shape[0], active))
        updated = value + residual
        return updated * torch.rsqrt(updated.square().mean(-1, keepdim=True) + 1e-6), updated

    def linear(value, weight):
        assert active == "root"
        label = "GateUp" if weight is gate_up else "Down"
        log.append((label, value.shape[0], active))
        return functional.linear(value, weight)

    def activation(value):
        assert active == "root"
        log.append(("SwiGLU", value.shape[0], active))
        gate, up = value.chunk(2, dim=-1)
        return functional.silu(gate) * up

    def local_mlp(value, residual, selected_gate, selected_down):
        normalized, updated = norm(value, residual)
        return linear(activation(linear(normalized, selected_gate)), selected_down), updated

    def all_reduce(value):
        assert active == "comm"
        label = next(names)
        result = value * 2
        reductions.append((label, result))
        log.append(("reduce", label, active))
        return result

    def project(value, residual, name):
        assert name == "left" and active == "root"
        log.append(("project", name, active))
        return SimpleNamespace(value=value + 1, residual=residual + 2, name=name, normalized=None)

    def core(phase, name):
        assert name == phase.name == "left"
        assert active == ("gdn_compute" if zipper else "root")
        assert phase.normalized is None
        phase.normalized = phase.value + 3
        log.append(("core", name, active))

    def output(phase, name):
        assert name == phase.name == "left" and active == "root"
        assert phase.normalized is not None
        log.append(("output", name, active))
        return phase.normalized + 5, phase.residual

    def prefix(value, residual, name, local):
        assert name == "right" and local and active == "root"
        log.append(("prefix", name, active))
        return value + 9, residual + 2

    def next_mlp(value, residual):
        assert active == "root"
        log.append(("next_mlp", value.shape[0], active))
        return value + 7, residual + 11

    namespace = dict(
        torch=SimpleNamespace(npu=SimpleNamespace(stream=stream)),
        functional=SimpleNamespace(linear=linear),
        adjacent=SimpleNamespace(
            phases=SimpleNamespace(project=project, core=core, output=output), run=prefix, local_mlp=next_mlp
        ),
        layer=SimpleNamespace(post_attention_layernorm=norm),
        mlp=SimpleNamespace(act_fn=activation),
        args=SimpleNamespace(gdn_shared=False),
        group=SimpleNamespace(all_reduce=all_reduce),
        local_mlp=local_mlp,
        gate_up_weight_timed=gate_up,
        down_weight_nz=down,
        boundary=boundary,
        rows=rows,
    )
    exec(compile(ast.Module(body=[definition], type_ignores=[]), benchmark.__file__, "exec"), namespace)
    cls = namespace["CapturedAttention"]
    entry = cls.__new__(cls)
    entry.gdn_phase_mode = "split_gdn_zipper" if zipper else "split_gdn_phased_barrier"
    entry.gdn_overlap = True
    entry.root, entry.comm = Stream("root"), Stream("comm")
    entry.gdn_compute = Stream("gdn_compute") if zipper else None
    entry.gdn_phase_ready, entry.gdn_core_done = Event("phase_ready"), Event("core_done")
    entry.gdn_prefix_done, entry.early_gdn = None, None
    entry.output_events = [Event(name, True) for name in ("OA_ready", "OA_done", "OB_ready", "OB_done")]
    entry.mlp_events = [Event(name) for name in ("DA_ready", "DA_done", "DB_ready", "DB_done")]
    entry.gdn_events = [Event(f"gdn_{index}") for index in range(8)]
    entry.attention_residual = prior.clone()
    entry.joined_output, entry.joined_residual = torch.empty_like(source), torch.empty_like(prior)
    first, second = (None, source[:boundary]), (None, source[boundary:])
    actual, actual_residual = entry.finish_split_mlp(first, second)
    expected_residual = source + prior
    normalized = expected_residual * torch.rsqrt(expected_residual.square().mean(-1, keepdim=True) + 1e-6)
    gate, up = functional.linear(normalized, gate_up).chunk(2, dim=-1)
    expected_down = functional.linear(functional.silu(gate) * up, down)
    torch.testing.assert_close(actual, expected_down * 8 + 50)
    torch.testing.assert_close(actual_residual, expected_residual + 13)
    assert [label for label, _ in reductions] == [
        "down_A",
        "down_B",
        "gdn_output_A",
        "gdn_output_B",
        "gdn_down_A",
        "gdn_down_B",
    ]
    assert entry.mlp_partitions[0][1] is reductions[0][1]
    assert entry.mlp_partitions[1][1] is reductions[1][1]
    assert entry.gdn_partitions[2] is reductions[2][1] and entry.gdn_partitions[3] is reductions[3][1]
    assert entry.gdn_partitions[6] is reductions[4][1] and entry.gdn_partitions[7] is reductions[5][1]
    core_event = ("core", "left", "gdn_compute" if zipper else "root")
    order = [
        ("wait", "OB_done", "root"),
        ("norm", rows - boundary, "root"),
        ("GateUp", rows - boundary, "root"),
        ("SwiGLU", rows - boundary, "root"),
        ("wait", "DA_done", "root"),
        ("project", "left", "root"),
    ]
    if zipper:
        order += [
            ("record", "phase_ready", "root"),
            ("wait", "phase_ready", "gdn_compute"),
            core_event,
            ("record", "core_done", "gdn_compute"),
        ]
    order += [("Down", rows - boundary, "root"), ("reduce", "down_B", "comm")]
    order += [("wait", "core_done", "root")] if zipper else [core_event]
    order += [("output", "left", "root")]
    if not zipper:
        order += [("record", "core_done", "root")]
    order += [
        ("wait", "core_done", "root"),
        ("wait", "DB_done", "root"),
        ("reduce", "gdn_output_A", "comm"),
        ("prefix", "right", "root"),
    ]
    # Sequential lookup handles repeated root waits on the same core event.
    last = -1
    for item in order:
        last = log.index(item, last + 1)
    assert log[-2:] == [("wait", "gdn_5", "root"), ("wait", "gdn_7", "root")]


@pytest.mark.parametrize("inplace_residual", [False, True])
def test_second_gdn_phase_overlaps_b_core_with_a_down_and_reorders_collectives(benchmark, inplace_residual):
    syntax = ast.parse(inspect.getsource(benchmark.experiment))
    definition = next(
        node for node in syntax.body[0].body if isinstance(node, ast.ClassDef) and node.name == "CapturedAttention"
    )
    active, log, returns = "root", [], []
    collective_names = iter(("output_A", "down_A", "output_B", "down_B"))

    class Stream:
        def __init__(self, name):
            self.name = name

        def wait_event(self, event):
            assert event.recorded, f"{self.name} consumes unrecorded {event.name}"
            log.append(("wait", event.name, self.name))

    class Event:
        def __init__(self, name, recorded=False):
            self.name, self.recorded = name, recorded

        def record(self, selected):
            assert active == selected.name
            self.recorded = True
            log.append(("record", self.name, active))

    @contextmanager
    def stream(selected):
        nonlocal active
        previous, active = active, selected.name
        try:
            yield
        finally:
            active = previous

    torch.manual_seed(317)
    rows, boundary, hidden, intermediate = 5, 2, 4, 3
    first_value, first_prior = (torch.randn(boundary, hidden, dtype=torch.float64) for _ in range(2))
    second_value, second_prior = (torch.randn(rows - boundary, hidden, dtype=torch.float64) for _ in range(2))
    first_prefix, first_residual = first_value + 9, first_prior + 2
    gate_up = torch.randn(intermediate * 2, hidden, dtype=torch.float64)
    down = torch.randn(hidden, intermediate, dtype=torch.float64)
    expected_prefix = torch.cat((first_prefix, second_value + 9)) * 2
    expected_residual = expected_prefix + torch.cat((first_residual, second_prior + 2))
    normalized = expected_residual * torch.rsqrt(expected_residual.square().mean(-1, keepdim=True) + 1e-6)
    gate, up = functional.linear(normalized, gate_up).chunk(2, dim=-1)
    expected = functional.linear(functional.silu(gate) * up, down) * 2

    def project(value, residual, name):
        assert active == "root" and name == "right"
        assert value is second_value and residual is second_prior
        log.append(("project", name, active))
        return SimpleNamespace(value=value + 1, residual=residual + 2, normalized=None, name=name)

    def core(phase, name):
        assert active == "gdn_compute" and name == phase.name == "right"
        assert phase.normalized is None
        phase.normalized = phase.value + 3
        log.append(("core", name, active))

    def output(phase, name):
        assert active == "root" and name == phase.name == "right"
        assert phase.normalized is not None
        log.append(("output", name, active))
        return phase.normalized + 5, phase.residual

    def prepare_mlp(value, residual):
        assert active == "root"
        size = value.shape[0]
        log.append(("norm", size, active))
        updated = residual.add_(value) if inplace_residual else residual + value
        normalized = updated * torch.rsqrt(updated.square().mean(-1, keepdim=True) + 1e-6)
        log.append(("GateUp", size, active))
        gate, up = functional.linear(normalized, gate_up).chunk(2, dim=-1)
        log.append(("SwiGLU", size, active))
        return functional.silu(gate) * up, updated

    def project_down(value):
        assert active == "root"
        log.append(("Down", value.shape[0], active))
        return functional.linear(value, down)

    def local_mlp(value, residual):
        prepared, updated = prepare_mlp(value, residual)
        return project_down(prepared), updated

    def all_reduce(value):
        assert active == "comm"
        name = next(collective_names)
        reduced = value * 2
        returns.append(reduced)
        log.append(("reduce", name, active))
        return reduced

    def unexpected_prefix(*args, **kwargs):
        raise AssertionError("zipper2 must consume the early left prefix and phased right prefix")

    namespace = dict(
        torch=SimpleNamespace(npu=SimpleNamespace(stream=stream)),
        adjacent=SimpleNamespace(
            phases=SimpleNamespace(project=project, core=core, output=output),
            prepare_mlp=prepare_mlp,
            project_down=project_down,
            local_mlp=local_mlp,
            run=unexpected_prefix,
        ),
        group=SimpleNamespace(all_reduce=all_reduce),
        boundary=boundary,
        rows=rows,
    )
    exec(compile(ast.Module(body=[definition], type_ignores=[]), benchmark.__file__, "exec"), namespace)
    cls = namespace["CapturedAttention"]
    entry = cls.__new__(cls)
    entry.gdn_phase_mode, entry.gdn_overlap = "split_gdn_zipper2", True
    entry.root, entry.comm, entry.gdn_compute = (Stream(name) for name in ("root", "comm", "gdn_compute"))
    entry.gdn_phase_ready, entry.gdn_core_done = Event("phase_ready"), Event("core_done")
    entry.gdn_prefix_done = Event("prefix_done", True)
    entry.mlp_events = [Event(name, True) for name in ("DA_ready", "DA_done", "DB_ready", "DB_done")]
    entry.gdn_events = [Event(f"gdn_{index}") for index in range(8)]
    entry.early_gdn = first_prefix, first_residual
    entry.joined_output, entry.joined_residual = torch.empty(rows, hidden), torch.empty(rows, hidden)
    entry.joined_output, entry.joined_residual = entry.joined_output.double(), entry.joined_residual.double()
    first_mlp, second_mlp = (None, first_value, first_prior), (None, second_value, second_prior)
    actual, actual_residual = entry.finish_adjacent(first_mlp, second_mlp)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_residual, expected_residual)
    assert entry.mlp_partitions == (first_mlp, second_mlp)
    assert [item[1] for item in log if item[0] == "reduce"] == ["output_A", "down_A", "output_B", "down_B"]
    assert entry.gdn_partitions[0] is returns[0]
    assert entry.gdn_partitions[2] is returns[2]
    assert entry.gdn_partitions[5] is returns[1] and entry.gdn_partitions[6] is returns[3]
    order = [
        ("wait", "prefix_done", "root"),
        ("wait", "DB_done", "root"),
        ("reduce", "output_A", "comm"),
        ("project", "right", "root"),
        ("wait", "gdn_1", "root"),
        ("norm", boundary, "root"),
        ("GateUp", boundary, "root"),
        ("SwiGLU", boundary, "root"),
        ("record", "phase_ready", "root"),
        ("wait", "phase_ready", "gdn_compute"),
        ("core", "right", "gdn_compute"),
        ("record", "core_done", "gdn_compute"),
        ("Down", boundary, "root"),
        ("reduce", "down_A", "comm"),
        ("wait", "core_done", "root"),
        ("output", "right", "root"),
        ("reduce", "output_B", "comm"),
        ("wait", "gdn_3", "root"),
        ("norm", rows - boundary, "root"),
        ("GateUp", rows - boundary, "root"),
        ("SwiGLU", rows - boundary, "root"),
        ("Down", rows - boundary, "root"),
        ("reduce", "down_B", "comm"),
        ("wait", "gdn_5", "root"),
        ("wait", "gdn_7", "root"),
    ]
    assert [log.index(item) for item in order] == sorted(log.index(item) for item in order)
    assert log[-2:] == order[-2:]


@pytest.mark.parametrize("inplace_residual", [False, True])
def test_shared_gate_up_down_pipeline_joins_returned_collectives_and_drains_before_gdn(benchmark, inplace_residual):
    syntax = ast.parse(inspect.getsource(benchmark.experiment))
    definition = next(
        node for node in syntax.body[0].body if isinstance(node, ast.ClassDef) and node.name == "CapturedAttention"
    )
    log, reduced = [], []
    active = "root"

    class Stream:
        def __init__(self, name):
            self.name = name

        def wait_event(self, event):
            assert event.recorded
            log.append(("wait", event.name, self.name))

    class Event:
        def __init__(self, name, recorded=False):
            self.name, self.recorded = name, recorded

        def record(self, selected):
            assert active == selected.name
            self.recorded = True
            log.append(("record", self.name, selected.name))

    @contextmanager
    def stream(selected):
        nonlocal active
        previous, active = active, selected.name
        try:
            yield
        finally:
            active = previous

    torch.manual_seed(149)
    rows, boundary, hidden, intermediate = 7, 3, 8, 6
    source, prior = (torch.randn(rows, hidden, dtype=torch.float64) for _ in range(2))
    gate_up = torch.randn(intermediate * 2, hidden, dtype=torch.float64)
    down = torch.randn(hidden, intermediate, dtype=torch.float64)
    expected_residual = source + prior
    normalized = expected_residual * torch.rsqrt(expected_residual.square().mean(-1, keepdim=True) + 1e-6)
    gate, up = functional.linear(normalized, gate_up).chunk(2, dim=-1)
    expected = functional.linear(functional.silu(gate) * up, down) * 2

    def norm(value, residual):
        log.append(("norm", value.shape[0], active))
        result = residual.add_(value) if inplace_residual else residual + value
        return result * torch.rsqrt(result.square().mean(-1, keepdim=True) + 1e-6), result

    def linear(value, weight):
        label = "GateUp" if weight is gate_up else "Down"
        log.append((label, value.shape[0], active))
        return functional.linear(value, weight)

    def activation(value):
        gate_value, up_value = value.chunk(2, dim=-1)
        return functional.silu(gate_value) * up_value

    def all_reduce(value):
        assert active == "comm"
        result = value * 2
        assert result.data_ptr() != value.data_ptr()
        reduced.append(result)
        log.append(("reduce", len(reduced), active))
        return result

    def adjacent_run(value, residual):
        log.append(("whole_gdn", active))
        assert value is entry.joined_output and residual is entry.joined_residual
        torch.testing.assert_close(value, expected)
        torch.testing.assert_close(residual, expected_residual)
        return value + 7, residual + 11

    namespace = dict(
        torch=SimpleNamespace(npu=SimpleNamespace(stream=stream)),
        functional=SimpleNamespace(linear=linear),
        adjacent=SimpleNamespace(run=adjacent_run),
        layer=SimpleNamespace(post_attention_layernorm=norm),
        mlp=SimpleNamespace(act_fn=activation),
        group=SimpleNamespace(all_reduce=all_reduce),
        gate_up_weight_timed=gate_up,
        down_weight_nz=down,
        boundary=boundary,
        rows=rows,
    )
    exec(compile(ast.Module(body=[definition], type_ignores=[]), benchmark.__file__, "exec"), namespace)
    cls = namespace["CapturedAttention"]
    entry = cls.__new__(cls)
    entry.mode = "split_down_overlap"
    entry.root, entry.comm = Stream("root"), Stream("comm")
    entry.output_events = [Event(name, True) for name in ("OA_ready", "OA_done", "OB_ready", "OB_done")]
    entry.mlp_events = [Event(name) for name in ("DA_ready", "DA_done", "DB_ready", "DB_done")]
    entry.attention_residual = prior.clone()
    entry.joined_output, entry.joined_residual = torch.empty_like(source), torch.empty_like(prior)
    first = (torch.full_like(source[:boundary], -100), source[:boundary])
    second = (torch.full_like(source[boundary:], -200), source[boundary:])
    output, residual = entry.finish_output_lanes(first, second)
    torch.testing.assert_close(output, expected + 7)
    torch.testing.assert_close(residual, expected_residual + 11)
    assert entry.output_partitions == (first, second)
    assert entry.mlp_partitions[0][1] is reduced[0]
    assert entry.mlp_partitions[1][1] is reduced[1]
    assert [item for item in log if item[0] in ("norm", "GateUp", "Down")] == [
        ("norm", rows, "root"),
        ("GateUp", rows, "root"),
        ("Down", boundary, "root"),
        ("Down", rows - boundary, "root"),
    ]
    order = [
        ("wait", "OA_done", "root"),
        ("wait", "OB_done", "root"),
        ("GateUp", rows, "root"),
        ("Down", boundary, "root"),
        ("reduce", 1, "comm"),
        ("Down", rows - boundary, "root"),
        ("reduce", 2, "comm"),
        ("wait", "DA_done", "root"),
        ("wait", "DB_done", "root"),
        ("whole_gdn", "root"),
    ]
    positions = [log.index(item) for item in order]
    assert positions == sorted(positions)


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
