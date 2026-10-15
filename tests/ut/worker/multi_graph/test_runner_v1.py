import dataclasses
import importlib.util
import json
import sys
import types
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tests.ut.worker.multi_graph.test_metadata import DecodeMetadata
from vllm_ascend.worker.multi_graph.config import MultiGraphConfig
from vllm_ascend.worker.multi_graph.costs import DecodeCostDocument, DecodeCostRecord
from vllm_ascend.worker.multi_graph.manager import ManagerState, MultiGraphManager
from vllm_ascend.worker.multi_graph.planner import ExactDecodePlanner


@pytest.fixture
def adapter_class(monkeypatch):
    modules = {
        "vllm.compilation": {"breakable_cudagraph": SimpleNamespace()},
        "vllm.config": {"CUDAGraphMode": SimpleNamespace()},
        "vllm.forward_context": {"BatchDescriptor": SimpleNamespace, "get_forward_context": lambda: None},
        "vllm.logger": {"logger": SimpleNamespace()},
        "vllm_ascend.ascend_forward_context": {"set_ascend_forward_context": lambda: None},
        "vllm_ascend.attention.attention_v1": {
            "AscendAttentionBackendImpl": SimpleNamespace,
            "AscendAttentionState": SimpleNamespace(DecodeOnly="DecodeOnly"),
            "AscendMetadata": DecodeMetadata,
        },
        "vllm_ascend.attention.utils": {"using_paged_attention": lambda *_: False},
        "vllm_ascend.compilation.updatable_graph": {"UpdatableGraph": SimpleNamespace},
        "vllm_ascend.device.hardware_profile": {
            "DeviceAdaptorFamily": SimpleNamespace(),
            "get_current_hardware_profile": lambda: None,
        },
        "vllm_ascend.worker.multi_graph.npu_backend": {"NPUExecutionBackend": SimpleNamespace},
        "vllm_ascend.worker.utils": {"disable_compilation": lambda *_: None},
    }
    for name, attributes in modules.items():
        module = types.ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    path = Path(__file__).resolve().parents[4] / "vllm_ascend/worker/multi_graph/runner_v1.py"
    spec = importlib.util.spec_from_file_location("multi_graph_adapter_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MultiGraphV1Adapter


@pytest.fixture
def adapter(adapter_class, signature, profile, execution):
    manager, backend, _, inputs, metadata = execution
    config = MultiGraphConfig(
        enabled=True,
        selection_policy="force",
        capture_sizes_per_lane=((2,), (2,)),
        graph_memory_budget_bytes=1024,
    )
    adapter = object.__new__(adapter_class)
    adapter.config = config
    adapter.planner = ExactDecodePlanner(config)
    adapter.profile = profile
    adapter.signature = signature
    adapter.manager = manager
    adapter.cost_table = {}
    adapter.execution_count = 0
    adapter.fallback_counts = Counter()
    adapter.layers = {}
    adapter.kv_addresses = ()
    adapter.runner = SimpleNamespace(
        block_size=128,
        input_batch=SimpleNamespace(
            req_ids=["a", "b", "c", "d"],
            block_table=[SimpleNamespace(block_table=SimpleNamespace(np=np.arange(8).reshape(4, 2)))],
        ),
    )
    context = SimpleNamespace(device_metadata_executor=None, attn_metadata=metadata)
    model_inputs = {"input_ids": inputs, "positions": inputs, "intermediate_tensors": None, "inputs_embeds": None}
    return adapter, backend, context, model_inputs


def test_adapter_merges_real_request_count_and_counts_execution(adapter):
    adapter, _, context, inputs = adapter
    output = adapter.try_run(context, inputs, 4)
    assert output[:, 0].tolist() == [0, 3, 6, 9]
    assert adapter.execution_count == 1


@pytest.mark.parametrize(
    "change", ["prefill", "embed", "kwargs", "device_metadata", "request_count", "duplicate_slots"]
)
def test_adapter_falls_back_before_launch_for_dynamic_misses(adapter, change):
    adapter, backend, context, inputs = adapter
    if change == "prefill":
        context.attn_metadata["layer"].attn_state = "PrefillCacheHit"
    elif change == "embed":
        inputs["inputs_embeds"] = object()
    elif change == "kwargs":
        inputs["unexpected_state"] = object()
    elif change == "device_metadata":
        context.device_metadata_executor = object()
    elif change == "request_count":
        adapter.runner.input_batch.req_ids.pop()
    elif change == "duplicate_slots":
        adapter.runner.input_batch.block_table[0].block_table.np.fill(0)
        context.attn_metadata["layer"].seq_lens_list[:] = [2] * 4
    assert adapter.try_run(context, inputs, 4) is None
    assert backend.calls == []


def test_default_measured_policy_never_forces_unmeasured_parallelism(adapter):
    adapter, backend, context, inputs = adapter
    adapter.planner = ExactDecodePlanner(
        MultiGraphConfig(
            enabled=True,
            capture_sizes_per_lane=((2,), (2,)),
            graph_memory_budget_bytes=1024,
        )
    )
    assert adapter.try_run(context, inputs, 4) is None
    assert backend.calls == []


def test_adapter_invalidates_changed_kv_storage(adapter):
    adapter, backend, context, inputs = adapter
    adapter.kv_addresses = (1234,)
    with pytest.raises(RuntimeError, match="KV cache addresses changed"):
        adapter.try_run(context, inputs, 4)
    assert backend.calls == []
    assert not adapter.manager.entries


def test_adapter_preflight_layout_miss_returns_baseline_before_launch(adapter):
    adapter, backend, context, inputs = adapter
    context.attn_metadata["layer"].block_tables = context.attn_metadata["layer"].block_tables[:, :1]
    assert adapter.try_run(context, inputs, 4) is None
    assert backend.calls == []


def test_adapter_submitted_value_error_never_returns_baseline(adapter, monkeypatch):
    adapter, backend, context, inputs = adapter

    def fail_update(*_):
        raise ValueError("update failed after replay")

    monkeypatch.setattr(backend, "update", fail_update)
    with pytest.raises(ValueError, match="after replay"):
        adapter.try_run(context, inputs, 4)


@pytest.fixture
def capture_adapter(adapter, monkeypatch):
    adapter, backend, context, inputs = adapter
    module = sys.modules["vllm.forward_context"]
    context.capturing = False
    monkeypatch.setattr(module, "get_forward_context", lambda: context)
    source_module = sys.modules["vllm_ascend.compilation.updatable_graph"]

    class Graph:
        def __init__(self):
            self.tasks = [object()]

    monkeypatch.setattr(source_module, "UpdatableGraph", Graph)
    adapter_module = sys.modules["vllm.config"]
    adapter_module.CUDAGraphMode.NONE = "NONE"
    runner_globals = type(adapter).capture.__globals__
    monkeypatch.setitem(runner_globals, "get_forward_context", lambda: context)
    monkeypatch.setitem(runner_globals, "UpdatableGraph", Graph)
    monkeypatch.setitem(runner_globals, "disable_compilation", lambda _: nullcontext())
    monkeypatch.setitem(runner_globals, "logger", SimpleNamespace(warning=lambda *_: None, info=lambda *_: None))
    streams = tuple(SimpleNamespace(wait_stream=lambda _: None, synchronize=lambda: None) for _ in range(2))
    adapter.compute_streams = streams
    adapter.update_streams = (object(), object())
    adapter.manager = MultiGraphManager(backend, 2)
    adapter.memory_start_free = 10000
    model_calls = []

    def model(**kwargs):
        model_calls.append(kwargs)
        return kwargs["input_ids"].float().unsqueeze(1)

    adapter.runner.get_model = lambda: model
    adapter.runner._dummy_run = lambda *_args, **_kwargs: adapter.runner._multi_graph_capture_callback(inputs)
    adapter.forward_context = lambda *_: nullcontext()
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(
            synchronize=lambda: None,
            mem_get_info=lambda: (10000, 20000),
            current_stream=lambda: object(),
            stream=lambda _: nullcontext(),
            graph_pool_handle=lambda: object(),
            graph=lambda *_args, **_kwargs: nullcontext(),
        ),
        raising=False,
    )
    return adapter, backend, context, model_calls


def test_startup_capture_owns_private_entries_and_clears_callback(capture_adapter):
    adapter, backend, context, calls = capture_adapter
    assert adapter.capture() == 0
    assert adapter.manager.state == ManagerState.READY
    assert len(adapter.manager.entries) == 2
    assert len(calls) == 6
    assert adapter.runner._multi_graph_capture_callback is None
    assert not context.capturing
    assert backend.calls == [("resolve", 0), ("resolve", 1)]
    for entry in adapter.manager.entries.values():
        assert entry.buffers.metadata["layer"].slot_mapping.tolist() == [-1, -1]
        assert not entry.buffers.metadata["layer"].block_tables.any()


@pytest.mark.parametrize("failure", ["budget", "capture", "resolve"])
def test_startup_failure_cannot_expose_partial_executor(capture_adapter, monkeypatch, failure):
    adapter, backend, context, _ = capture_adapter
    if failure == "budget":
        monkeypatch.setattr(torch.npu, "mem_get_info", lambda: (0, 20000))
    elif failure == "resolve":
        backend.fail_at = ("resolve", 1)
    else:

        def fail(*_args, **_kwargs):
            raise RuntimeError("capture failure")

        monkeypatch.setattr(torch.npu, "graph", fail)
    with pytest.raises(RuntimeError):
        adapter.capture()
    assert adapter.manager.state == ManagerState.FAILED
    assert adapter.runner._multi_graph_capture_callback is None
    assert not context.capturing


def test_adapter_reports_planner_fallback_reason(adapter):
    adapter, _, context, inputs = adapter
    adapter.planner = ExactDecodePlanner(
        MultiGraphConfig(enabled=True, capture_sizes_per_lane=((2,), (2,)), graph_memory_budget_bytes=1024)
    )
    assert adapter.try_run(context, inputs, 4) is None
    assert adapter.fallback_counts["missing_measurement"] == 1


def test_constructor_loads_calibrated_costs_without_changing_capture_identity(adapter_class, monkeypatch, tmp_path):
    class VllmConfig:
        def __init__(self, additional):
            self.additional_config = additional

        def compute_hash(self):
            return json.dumps(self.additional_config, sort_keys=True)

    config = MultiGraphConfig(
        enabled=True,
        selection_policy="force",
        capture_sizes_per_lane=((2,), (2,)),
        graph_memory_budget_bytes=1024,
    )
    runner = SimpleNamespace(
        ascend_config=SimpleNamespace(multi_graph_config=config),
        compilation_config=SimpleNamespace(static_forward_context={}),
        vllm_config=VllmConfig({"multi_graph_config": dataclasses.asdict(config)}),
        device="npu:0",
        dtype=torch.bfloat16,
        block_size=128,
    )
    created_streams = []

    def stream():
        result = object()
        created_streams.append(result)
        return result

    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(
            synchronize=lambda: None,
            mem_get_info=lambda: (10000, 20000),
            Stream=stream,
            get_device_name=lambda _: "test-npu",
        ),
        raising=False,
    )
    monkeypatch.setattr(adapter_class, "validate_runner", lambda _: None)
    monkeypatch.setattr(adapter_class, "current_kv_addresses", lambda _: ())
    globals_dict = adapter_class.__init__.__globals__
    monkeypatch.setitem(
        globals_dict,
        "get_current_hardware_profile",
        lambda: SimpleNamespace(device_addressing_mode=SimpleNamespace(name="SINGLE_CHIP")),
    )
    monkeypatch.setattr(globals_dict["importlib"].metadata, "version", lambda _: "1.0")
    first = adapter_class(runner)
    record = DecodeCostRecord(
        lane_sizes=(2, 2),
        execution="parallel",
        kv_lengths=(2, 3, 4, 5),
        single_graph_us=(100,) * 5,
        split_us=(70,) * 5,
    )
    document = DecodeCostDocument(metric="synchronized_decode_forward_us", signature=first.signature, records=(record,))
    path = tmp_path / "costs.json"
    path.write_text(document.model_dump_json(), encoding="utf-8")
    measured = dataclasses.replace(config, selection_policy="measured", cost_table_path=str(path))
    runner.ascend_config.multi_graph_config = measured
    runner.vllm_config.additional_config["multi_graph_config"] = dataclasses.asdict(measured)
    second = adapter_class(runner)
    assert second.signature == first.signature
    assert len(second.cost_table) == 1
    assert runner.vllm_config.additional_config["multi_graph_config"]["cost_table_path"] == str(path)
    created_streams.clear()
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        adapter_class(runner)
    assert not created_streams


def test_online_calibration_switches_paths_without_reexecuting_batch(adapter, monkeypatch, tmp_path):
    from vllm_ascend.worker.multi_graph.calibration import CalibrationPolicy, OnlineCalibrator

    adapter, backend, context, inputs = adapter
    adapter.signature = dataclasses.replace(adapter.signature, runtime_identity="cpu-test")
    old_entries = list(adapter.manager.entries.values())
    adapter.manager.entries.clear()
    for entry in old_entries:
        entry.key = dataclasses.replace(entry.key, signature=adapter.signature)
        adapter.manager.entries[entry.key] = entry
    adapter.config = dataclasses.replace(
        adapter.config,
        selection_policy="measured",
        online_calibration=True,
        minimum_cost_samples=1,
        calibration_output_path=str(tmp_path / "online.json"),
    )
    adapter.planner = ExactDecodePlanner(adapter.config)
    adapter.calibrator = OnlineCalibrator(adapter.signature, CalibrationPolicy(1, 1, 1))
    monkeypatch.setattr(torch, "npu", SimpleNamespace(synchronize=lambda: None), raising=False)
    baseline_calls = []

    def baseline():
        baseline_calls.append(1)
        return torch.zeros((4, 1))

    adapter.try_run(context, inputs, 4, baseline)
    assert baseline_calls == [1] and adapter.execution_count == 0
    adapter.try_run(context, inputs, 4, baseline)
    assert baseline_calls == [1] and adapter.execution_count == 1
    assert (tmp_path / "online.json").is_file()
    assert len(adapter.cost_table) == 1
