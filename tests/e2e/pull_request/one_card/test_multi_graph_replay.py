"""Fixed-query target replay qualification; no drafter or acceptance sampling."""

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize("query", [1, 2, 4, 8])
@pytest.mark.parametrize("merge", ["cat", "captured"])
def test_fixed_query_replay(tmp_path, query, merge, request):
    pytest.importorskip("torch_npu")
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("requires Ascend NPU")
    repo = Path(__file__).resolve().parents[4]
    destination = tmp_path / "replay.json"
    subprocess.run(
        [
            sys.executable,
            str(repo / "benchmarks/multi_graph_replay.py"),
            "--model",
            request.config.getoption("--replay-model", default="Qwen/Qwen2.5-0.5B-Instruct"),
            "--batch-invariant",
            "--merge",
            merge,
            "--batch",
            "7",
            "--capacity",
            "8",
            "--query",
            str(query),
            "--kv-length",
            "64",
            "--rounds",
            "1",
            "--protocols",
            "update",
            "frozen",
            "--repeats",
            "3",
            "--output",
            str(destination),
        ],
        cwd=repo,
        check=True,
        timeout=240,
    )
    result = json.loads(destination.read_text())
    assert result["status"] == "passed"
    assert result["batch_invariant"]
    assert result["merge"] == merge
    assert all(case["passed"] for case in result["controls"])
    assert result["draft_invocations"] == 0
    assert len(result["correctness"]) == 6
    assert all(case["passed"] for case in result["correctness"])
    assert len(result["measurements"]) == 6
    for graph in result["graphs"]:
        assert (graph["update_tasks"] > 0) == (graph["protocol"] == "update")
        assert (graph["frozen_tasks"] > 0) == (graph["protocol"] == "frozen")


def test_native_same_shape_replay_does_not_claim_cross_shape_accuracy(tmp_path, request):
    pytest.importorskip("torch_npu")
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("requires Ascend NPU")
    repo = Path(__file__).resolve().parents[4]
    destination = tmp_path / "native.json"
    subprocess.run(
        [
            sys.executable,
            str(repo / "benchmarks/multi_graph_replay.py"),
            "--model",
            request.config.getoption("--replay-model", default="Qwen/Qwen2.5-0.5B-Instruct"),
            "--batch",
            "33",
            "--capacity",
            "64",
            "--query",
            "4",
            "--kv-length",
            "512",
            "--qualification",
            "same-shape",
            "--submission",
            "replay-first",
            "--merge",
            "captured",
            "--logit-check",
            "--rounds",
            "1",
            "--repeats",
            "3",
            "--output",
            str(destination),
        ],
        cwd=repo,
        env=dict(os.environ, VLLM_BATCH_INVARIANT="0"),
        check=True,
        timeout=240,
    )
    result = json.loads(destination.read_text())
    assert result["status"] == "passed"
    assert not result["batch_invariant"]
    assert not result["cross_shape_qualified"]
    assert result["qualification"] == "same-shape"
    padding_control = next(row for row in result["controls"] if row["name"] == "eager_padded")
    assert not padding_control["required"]
    assert padding_control["passed"] is None
    assert len(result["correctness"]) == 3
    for row in result["correctness"]:
        assert row["passed"]
        assert "relative_rms" in row["cross_shape"]
        assert row["last_query_logit_diagnostics"]["tokens"] == 33
    assert all(row["update_tasks"] > 0 for row in result["graphs"])


@pytest.mark.parametrize("query,lanes,merge", [(1, (4, 3), "cat"), (8, (6, 1), "captured")])
def test_shared_dense_attention_branches(tmp_path, request, query, lanes, merge):
    pytest.importorskip("torch_npu")
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("requires Ascend NPU")
    repo = Path(__file__).resolve().parents[4]
    destination = tmp_path / "shared_dense.json"
    subprocess.run(
        [
            sys.executable,
            str(repo / "benchmarks/multi_graph_shared_dense.py"),
            "--model",
            request.config.getoption("--replay-model", default="Qwen/Qwen2.5-0.5B-Instruct"),
            "--batch",
            "7",
            "--capacity",
            "8",
            "--query",
            str(query),
            "--kv-length",
            "128",
            "--lanes",
            *map(str, lanes),
            "--merge",
            merge,
            "--rounds",
            "1",
            "--repeats",
            "3",
            "--output",
            str(destination),
        ],
        cwd=repo,
        env=dict(os.environ, VLLM_BATCH_INVARIANT="0"),
        check=True,
        timeout=240,
    )
    result = json.loads(destination.read_text())
    assert result["status"] == "passed"
    assert result["full_split_merge"] == merge
    assert not result["batch_invariant"]
    assert not result["cross_architecture_qualified"]
    assert all(row["passed"] for row in result["checks"])
    assert len([row for row in result["checks"] if row["name"].endswith("/changed_input")]) == 8
    assert len(result["measurements"]) == 8
    for row in result["measurements"]:
        assert len(row["wall_us"]) == 3
        assert min(row["wall_us"]) > 0
    graphs = {row["name"]: row for row in result["graphs"]}
    for name in ("shared_dense_pre", "shared_dense_post"):
        assert graphs[name]["update_tasks"] == 0
    for name in ("attention_lane_0", "attention_lane_1"):
        assert graphs[name]["update_tasks"] == 1
    for name in ("joined_shared_serial", "joined_shared_parallel"):
        assert graphs[name]["update_tasks"] == 2
        assert graphs[name]["distinct_workspaces"] == 2


@pytest.mark.parametrize(
    "schedule,kv_write,cores",
    [("joined", "branches", 0), ("post-overlap", "branches", 0), ("joined", "shared", 0), ("joined", "branches", 10)],
)
def test_full_decoder_padding_vs_split_parallel(tmp_path, request, schedule, kv_write, cores):
    pytest.importorskip("torch_npu")
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("requires Ascend NPU")
    repo = Path(__file__).resolve().parents[4]
    destination = tmp_path / "model_scaling.json"
    subprocess.run(
        [
            sys.executable,
            str(repo / "benchmarks/multi_graph_model_scaling.py"),
            "--split-schedule",
            schedule,
            "--kv-write",
            kv_write,
            "--attention-cores",
            str(cores),
            "--model",
            request.config.getoption("--replay-model", default="Qwen/Qwen2.5-0.5B-Instruct"),
            "--batch",
            "7",
            "--capacity",
            "8",
            "--query",
            "8",
            "--kv-length",
            "128",
            "--rounds",
            "1",
            "--repeats",
            "3",
            "--output",
            str(destination),
        ],
        cwd=repo,
        env=dict(os.environ, VLLM_BATCH_INVARIANT="0"),
        check=True,
        timeout=300,
    )
    result = json.loads(destination.read_text())
    assert result["status"] == "passed"
    assert result["update_included"]
    assert result["split_schedule"] == schedule
    assert result["kv_write"] == kv_write
    assert result["attention_cores"] == cores
    assert not result["cross_architecture_qualified"]
    assert {row["mode"] for row in result["measurements"]} == {"padding", "split_parallel"}
    assert len(result["measurements"]) == 2
    assert result["tensor_parallel_size"] == 1
    for rank in result["ranks"]:
        assert all(check["passed"] for check in rank["checks"])
        assert len([check for check in rank["checks"] if check["name"].endswith("/changed_input")]) == 2
    graphs = {graph["mode"]: graph for graph in result["graphs"]}
    assert graphs["padding"]["update_tasks"] == result["layers"]
    assert graphs["split_parallel"]["update_tasks"] == 2 * result["layers"]
    assert graphs["split_parallel"]["workspaces"] == 2


def test_default_capture_case_selection(tmp_path, request):
    pytest.importorskip("torch_npu")
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("requires Ascend NPU")
    repo = Path(__file__).resolve().parents[4]
    destination = tmp_path / "default_case"
    subprocess.run(
        [
            sys.executable,
            str(repo / "benchmarks/multi_graph_default_capture.py"),
            "--model",
            request.config.getoption("--replay-model", default="Qwen/Qwen2.5-0.5B-Instruct"),
            "--batch-min",
            "7",
            "--batch-max",
            "7",
            "--max-num-seqs",
            "16",
            "--select-only",
            "--output-dir",
            str(destination),
        ],
        cwd=repo,
        env=dict(os.environ, VLLM_BATCH_INVARIANT="0"),
        check=True,
        timeout=120,
    )
    selection = json.loads((destination / "selection.json").read_text())
    assert selection["batch"] == 7
    assert selection["query"] == 1
    assert selection["selected_mode"] == "FULL"
    assert selection["selected_tokens"] == selection["selected_requests"] == 8
    assert selection["selected_tokens"] == min(size for size in selection["capture_sizes"] if size >= 7)
    assert len(selection["benchmark_sha256_at_start"]) == 64
    assert not (destination / "summary.json").exists()


@pytest.mark.parametrize("batch,strategy,pair", [(6, "exact", [4, 2]), (7, "balanced", [4, 4])])
@pytest.mark.parametrize("layout", ["baseline", "pinned", "nz_selected", "nz_all", "native"])
def test_complete_decoder_default_bucket_reuse(tmp_path, request, batch, strategy, pair, layout):
    pytest.importorskip("torch_npu")
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("requires Ascend NPU")
    repo = Path(__file__).resolve().parents[4]
    destination = tmp_path / "bucket_replay.json"
    optimized = layout != "baseline"
    projections = {
        "nz_selected": ["qkv", "o", "down"],
        "nz_all": ["qkv", "o", "gate_up", "down"],
        "native": ["qkv", "o", "gate_up", "down"],
    }.get(layout, [])
    subprocess.run(
        [
            sys.executable,
            str(repo / "benchmarks/multi_graph_bucket_replay.py"),
            "--model",
            request.config.getoption("--replay-model", default="Qwen/Qwen2.5-0.5B-Instruct"),
            "--batch-min",
            str(batch),
            "--batch-max",
            str(batch),
            "--max-num-seqs",
            "16",
            "--strategy",
            strategy,
            "--kv-length",
            "128",
            "--rounds",
            "1",
            "--repeats",
            "3",
            "--output",
            str(destination),
            *(["--pinned-inputs", "--update-chunk-size", "4", "--diagnose-lanes"] if optimized else []),
            *(["--nz-projections", *projections] if projections else []),
            *(["--native-update", "--host-timing"] if layout == "native" else []),
        ],
        cwd=repo,
        env=dict(os.environ, VLLM_BATCH_INVARIANT="0"),
        check=True,
        timeout=300,
    )
    report = json.loads(destination.read_text())
    assert report["status"] == "passed"
    assert report["pair"] == pair
    assert report["graph_count"] == 3
    assert report["combination_graph_count"] == 0
    assert report["reuse_graph_and_buffer_identity_unchanged"]
    assert all(graph["capacity"] in report["selection"]["capture_sizes"] for graph in report["graphs"])
    assert all(graph["update_tasks"] > 0 for graph in report["graphs"])
    assert all(check["passed"] for check in report["checks"])
    assert {check["batch"] for check in report["checks"]} == {batch, batch - 1}
    assert {check["variant"] for check in report["checks"]} == {0, 1}
    assert {row["mode"] for row in report["measurements"]} == {"padding", "split_parallel"}
    assert not report["cross_shape_qualified"]
    assert report["pinned_inputs"] == optimized
    assert report["update_chunk_size"] == (4 if optimized else 0)
    assert len(report["lane_diagnostics"]) == (2 if optimized else 0)
    assert report["nz_projections"] == projections
    assert report["original_nd_eager_reference"] == bool(projections)
    assert not report["compiled_requested"]
    assert report["native_update"] == (layout == "native")
    assert report["host_timing"] == (layout == "native")
    for row in report["measurements"]:
        assert len(row["host_samples_us"]) == (3 if layout == "native" else 0)
        if layout == "native":
            assert len(report["native_source_sha256"]) == 64
            for wall, phases in zip(row["wall_us"], row["host_samples_us"]):
                assert set(phases) == {"prepare", "submit", "update", "finish", "synchronize"}
                assert sum(phases.values()) == pytest.approx(wall)
    if projections:
        assert {row["projection"] for row in report["layout_changes"]} == set(projections)
        assert all(row["before"] == 2 and row["after"] == 29 for row in report["layout_changes"])
        assert len(report["layout_changes"]) == len(projections) * report["graphs"][0]["update_tasks"]


@pytest.mark.parametrize("weight_format,shared_input_probe", [("loaded", False), ("nz", False), ("nz", True)])
def test_dense_projection_diagnostic(tmp_path, request, weight_format, shared_input_probe):
    pytest.importorskip("torch_npu")
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("requires Ascend NPU")
    repo = Path(__file__).resolve().parents[4]
    destination = tmp_path / "dense.json"
    subprocess.run(
        [
            sys.executable,
            str(repo / "benchmarks/multi_graph_dense_probe.py"),
            "--model",
            request.config.getoption("--replay-model", default="Qwen/Qwen2.5-0.5B-Instruct"),
            "--strategy",
            "balanced",
            "--weight-format",
            weight_format,
            "--rounds",
            "1",
            "--repeats",
            "2",
            "--output",
            str(destination),
            *(["--shared-input-probe"] if shared_input_probe else []),
        ],
        cwd=repo,
        env=dict(os.environ, VLLM_BATCH_INVARIANT="0"),
        check=True,
        timeout=300,
    )
    report = json.loads(destination.read_text())
    assert report["status"] == "passed"
    assert report["selection"]["batch"] == 98
    assert report["pair"] == [56, 48]
    assert report["scope"] == "synthetic_input_real_weight_projection_sweep_not_decoder"
    assert {row["projection"] for row in report["results"]} == {"qkv", "o", "gate_up", "down"}
    for row in report["results"]:
        assert row["passed"]
        assert row["layers"] > 0
        assert set(row["summary_ms"]) == {
            "padding",
            "lane0",
            "lane1",
            "split_parallel",
            *(("shared_bucket", "shared_staged") if shared_input_probe else ()),
        }
        assert row["shared_bucket_reuses_padding_graph"] == shared_input_probe
        assert row["shared_bucket_includes_input_copies"] == shared_input_probe
        if shared_input_probe:
            assert row["shared_bucket_capacity"] == report["selection"]["selected_requests"]
            assert row["shared_bucket_output_alias_checks"] == 2 * row["layers"]
            assert row["shared_staged_graph_count"] == row["shared_staged_input_gathers_per_replay"] == row["layers"]
            assert row["shared_staged_capture_capacity"] == row["shared_bucket_capacity"]
        assert all(len(value["samples_us"]) == 2 for value in row["measurements"])
        if weight_format == "nz":
            assert row["weight_format"] == 29


def test_native_fia_binding_guards(monkeypatch):
    pytest.importorskip("torch_npu")
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("requires Ascend NPU")
    repo = Path(__file__).resolve().parents[4]
    monkeypatch.syspath_prepend(str(repo / "benchmarks"))
    from multi_graph_native_update import load_native_update

    native, _ = load_native_update()
    stream = torch.npu.Stream()
    with pytest.raises(RuntimeError, match="nonempty"):
        native.NativeLane((), stream)
    with pytest.raises(RuntimeError, match="nonempty"):
        native.submit([], 1)
    data = torch.ones((1, 1, 1, 128), device="npu", dtype=torch.bfloat16)
    output = torch.empty_like(data)
    stream.wait_stream(torch.npu.current_stream())
    graph = torch.npu.NPUGraph()
    # Only a real handle is needed here; these tests never submit a FIA update.
    with torch.npu.graph(graph, stream=stream):
        torch.npu.graph_task_group_begin(stream)
        torch.add(data, data, out=output)
        handle = torch.npu.graph_task_group_end(stream)
    kwargs = dict(
        query=data,
        key=data,
        value=data,
        scale=1.0,
        actual_seq_lengths=[1],
        actual_seq_lengths_kv=[1],
        block_table=None,
        out=[output, torch.empty(0, device="npu")],
    )
    task = SimpleNamespace(kwargs=kwargs, handle=handle, event=torch.npu.ExternalEvent())
    plan = native.NativeLane([task], stream)
    with pytest.raises(RuntimeError, match="task count"):
        plan.bind([])
    with pytest.raises(RuntimeError, match="fixed FIA argument changed"):
        plan.bind([SimpleNamespace(kwargs=dict(kwargs, scale=0.5), handle=handle, event=task.event)])
    with pytest.raises(RuntimeError, match="task event changed"):
        plan.bind([SimpleNamespace(kwargs=kwargs, handle=handle, event=torch.npu.ExternalEvent())])
    task.kwargs = dict(kwargs, actual_seq_lengths_kv=[2])
    plan.bind([task])
    plan.bind_kv_lengths([[3]])
    with pytest.raises(RuntimeError, match="task count"):
        plan.bind_kv_lengths([])
    with pytest.raises(RuntimeError, match="row count"):
        plan.bind_kv_lengths([[3, 4]])
    with pytest.raises(RuntimeError, match="negative KV"):
        plan.bind_kv_lengths([[-1]])
    with pytest.raises(RuntimeError, match="nonempty replay"):
        native.NativeReplay(graph, stream, stream, [], [])
    with pytest.raises(RuntimeError, match="zero or two timing"):
        native.NativeReplay(graph, stream, stream, [plan], [torch.npu.Event(enable_timing=True)])
    replay = native.NativeReplay(graph, stream, stream, [plan], [])
    with pytest.raises(RuntimeError, match="not configured"):
        replay.replay(True)
    task.event = torch.npu.ExternalEvent()
    with pytest.raises(RuntimeError, match="task event changed"):
        plan.bind([task])
    stream.synchronize()


@pytest.mark.parametrize(
    "nz,native,shared", [(False, False, False), (True, False, False), (True, True, False), (True, True, True)]
)
def test_default_bucket_internal_block_reuse(tmp_path, request, nz, native, shared, packed=False, native_replay=False):
    pytest.importorskip("torch_npu")
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("requires Ascend NPU")
    repo = Path(__file__).resolve().parents[4]
    destination = tmp_path / "block.json"
    subprocess.run(
        [
            sys.executable,
            str(repo / "benchmarks/multi_graph_bucket_block.py"),
            "--model",
            request.config.getoption("--replay-model", default="Qwen/Qwen2.5-0.5B-Instruct"),
            "--batch-min",
            "7",
            "--batch-max",
            "7",
            "--max-num-seqs",
            "16",
            "--kv-length",
            "128",
            "--rounds",
            "1",
            "--repeats",
            "4",
            "--output",
            str(destination),
            *(["--nz"] if nz else []),
            *(["--native-update"] if native else []),
            *(["--packed-metadata", "--phase-diagnostics"] if packed else []),
            *(["--native-replay"] if native_replay else []),
            *(
                ["--shared-cache", "--serial-control", "--independent-updates", "--static-task-plan", "--trace"]
                if shared
                else []
            ),
        ],
        cwd=repo,
        env=dict(os.environ, VLLM_BATCH_INVARIANT="0"),
        check=True,
        timeout=300,
    )
    report = json.loads(destination.read_text())
    assert report["status"] == "passed"
    assert report["dense_capacity"] == 8 and report["pair"] == [4, 4]
    assert report["fixed_boundary"] == 4
    assert report["real_lane_counts"] == [[4, 3], [4, 2], [4, 3], [4, 3]]
    assert report["timed_dynamic_cases"] == [[7, 0], [6, 0], [7, 0], [7, 0]]
    assert report["timed_kv_length"] == 128
    modes = 3 if shared else 2
    assert report["graph_count"] == modes and report["graphs_per_strategy_per_bucket"] == 1
    assert report["combination_graph_count"] == report["explicit_output_merge_copies"] == 0
    assert report["reuse_graph_and_buffer_identity_unchanged"]
    assert report["nz"] == nz and report["original_nd_eager_reference"]
    assert report["native_update"] == native
    assert report["shared_cache"] == shared and report["serial_control"] == shared
    assert report["static_task_plan"] == shared
    assert report["packed_metadata"] == packed
    assert report["native_replay"] == native_replay
    assert report["h2d_submissions_per_step"] == (1 if packed else 5)
    if packed:
        assert len(report["phase_diagnostics"]) == 32 * modes
        assert all(row["graph_plus_update_ready_span_us"] > 0 for row in report["phase_diagnostics"])
    if native:
        assert len(report["native_source_sha256"]) == 64
    assert not report["cross_shape_qualified"]
    assert all(check["passed"] for check in report["checks"])
    assert len([check for check in report["checks"] if check["name"].startswith("reuse/")]) == 5 * modes
    assert report["qualification_cases"][-1] == [6, 0]
    assert [(g["update_tasks"], g["workspaces"]) for g in report["graphs"]] == [(1, 1)] + [(2, 2)] * (modes - 1)
    assert len(report["measurements"]) == 2 * modes
    assert all(len(row["wall_us"]) == 4 for row in report["measurements"])
    assert set(report["summary_ms"]) == {"fixed", "dynamic"}
    if shared:
        assert [g["update_streams"] for g in report["graphs"]] == [1, 1, 2]
        assert len(report["trace_analysis"]) == 9
        for row in report["trace_analysis"]:
            writes = sum(v["count"] for name, v in row["kernel_inventory"].items() if "ScatterPaKvCache" in name)
            assert writes == 1


@pytest.mark.parametrize("native_replay", [False, True])
def test_packed_bucket_runtime_upload(tmp_path, request, native_replay):
    test_default_bucket_internal_block_reuse(
        tmp_path, request, nz=True, native=True, shared=True, packed=True, native_replay=native_replay
    )


def test_packed_metadata_integer_views_on_npu(monkeypatch):
    pytest.importorskip("torch_npu")
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("requires Ascend NPU")
    repo = Path(__file__).resolve().parents[4]
    monkeypatch.syspath_prepend(str(repo / "benchmarks"))
    from multi_graph_bucket_block import PackedStepBuffers

    targets = dict(
        ids=torch.empty(7, dtype=torch.int64, device="npu"),
        lengths=torch.empty(7, dtype=torch.int32, device="npu"),
        pages=torch.empty((7, 2), dtype=torch.int32, device="npu"),
        slots=torch.empty(7, dtype=torch.int64, device="npu"),
    )
    buffers = PackedStepBuffers(targets)
    assert buffers.host_buffer.is_pinned()
    for token in (31, 127, 42):
        payload = dict(
            ids=[token + i for i in range(7)],
            lengths=[token] * 6 + [0],
            pages=[[i, i + 1] for i in range(7)],
            slots=list(range(6)) + [-1],
        )
        for name, values in payload.items():
            buffers.write(name, values)
        buffers.transfer()
        for name, values in payload.items():
            torch.testing.assert_close(buffers.device[name].cpu(), torch.tensor(values, dtype=targets[name].dtype))
            assert buffers.device[name].data_ptr() == buffers.device_buffer.data_ptr() + buffers.schema[name]["offset"]
