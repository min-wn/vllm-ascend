// Benchmark-only adapter built against the installed torch_npu headers and ABI.
#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch_npu/csrc/core/npu/NPUGraph.h>
#include <torch_npu/csrc/core/npu/NPUGuard.h>
#include <torch_npu/csrc/npu/Event.h>

namespace py = pybind11;

struct NativeLane {
  c10::OperatorHandle op;
  c10_npu::NPUStream stream;
  std::vector<c10_npu::NPUTaskGroupHandle> handles;
  std::vector<c10_npu::NPUEvent*> events;
  std::vector<py::object> owners;
  py::object stream_owner;
  std::vector<torch::jit::Stack> arguments;
  std::vector<py::dict> fixed_kwargs;
  std::vector<std::pair<std::string, size_t>> runtime_arguments;

  NativeLane(py::sequence tasks, py::object update_stream)
      : op(c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_fused_infer_attention_score", "out")),
        stream(c10_npu::NPUStream::unpack3(
            update_stream.attr("stream_id").cast<int64_t>(), update_stream.attr("device_index").cast<int>(),
            static_cast<c10::DeviceType>(update_stream.attr("device_type").cast<int>()))),
        stream_owner(update_stream) {
    auto event_type = py::module_::import("torch_npu").attr("_C").attr("_NPUEventBase");
    for (auto task : tasks) {
      py::object event = task.attr("event");
      TORCH_CHECK(py::isinstance(event, event_type), "expected NPU event");
      handles.push_back(task.attr("handle").cast<c10_npu::NPUTaskGroupHandle>());
      // The installed public header defines this Python object's layout.
      // Retain its owner for the whole plan; never copy or destroy the event.
      events.push_back(&reinterpret_cast<THNPEvent*>(event.ptr())->npu_event);
      owners.push_back(event);
      auto kwargs = task.attr("kwargs").cast<py::kwargs>();
      fixed_kwargs.push_back(py::dict(kwargs.attr("copy")()));
      arguments.push_back(
          torch::jit::createStackForSchema(op.schema(), torch::jit::tuple_slice(py::tuple(), 0), kwargs, std::nullopt));
    }
    TORCH_CHECK(!handles.empty(), "nonempty FIA task list required");
    for (size_t i = 0; i < op.schema().arguments().size(); ++i) {
      const auto& name = op.schema().arguments()[i].name();
      if (name == "actual_seq_lengths" || name == "actual_seq_lengths_kv" || name == "block_table") {
        runtime_arguments.emplace_back(name, i);
      }
    }
    bind(tasks);
  }

  void bind(py::sequence tasks) {
    TORCH_CHECK(tasks.size() == handles.size(), "task count changed");
    std::vector<std::pair<PyObject*, c10::IValue>> converted;
    for (size_t i = 0; i < handles.size(); ++i) {
      auto task = tasks[i];
      TORCH_CHECK(task.attr("event").ptr() == owners[i].ptr(), "task event changed");
      TORCH_CHECK(task.attr("handle").cast<c10_npu::NPUTaskGroupHandle>().task_group == handles[i].task_group,
                  "task handle changed");
      auto kwargs = task.attr("kwargs").cast<py::dict>();
      TORCH_CHECK(kwargs.size() == fixed_kwargs[i].size(), "FIA argument set changed");
      for (auto field : fixed_kwargs[i]) {
        const auto name = py::cast<std::string>(field.first);
        if (name != "actual_seq_lengths" && name != "actual_seq_lengths_kv" && name != "block_table") {
          TORCH_CHECK(kwargs.contains(field.first) && kwargs[field.first].ptr() == field.second.ptr(),
                      "fixed FIA argument changed: ", name);
        }
      }
      for (const auto& [name, index] : runtime_arguments) {
        auto value = kwargs[py::str(name)];
        auto found = std::find_if(converted.begin(), converted.end(),
                                  [&](const auto& item) { return item.first == value.ptr(); });
        if (found == converted.end()) {
          converted.emplace_back(value.ptr(), torch::jit::argumentToIValue(op.schema(), index, value));
          arguments[i][index] = converted.back().second;
        } else {
          arguments[i][index] = found->second;
        }
      }
    }
  }

  void update(size_t begin, size_t end) {
    c10_npu::NPUStreamGuard guard(stream);
    for (size_t i = begin; i < end; ++i) {
      auto stack = arguments[i];
      c10_npu::graph_task_update_begin(stream, handles[i]);
      op.callBoxed(&stack);
      c10_npu::graph_task_update_end(stream);
      events[i]->record(stream);
    }
  }

  void bind_kv_lengths(py::sequence values) {
    // Only for a validated plan with fixed query boundaries and tensor handles.
    // Buffer contents may change; replacing tensors requires constructing a plan.
    TORCH_CHECK(values.size() == handles.size(), "KV length task count changed");
    auto field = std::find_if(runtime_arguments.begin(), runtime_arguments.end(),
                              [](const auto& item) { return item.first == "actual_seq_lengths_kv"; });
    TORCH_CHECK(field != runtime_arguments.end(), "missing KV length argument");
    std::vector<c10::IValue> converted_values;
    for (size_t i = 0; i < handles.size(); ++i) {
      auto converted = torch::jit::argumentToIValue(op.schema(), field->second, values[i]);
      TORCH_CHECK(converted.toSymIntList().size() == arguments[i][field->second].toSymIntList().size(),
                  "KV length row count changed");
      for (const c10::SymInt length : converted.toSymIntList())
        TORCH_CHECK(length.expect_int() >= 0, "negative KV length");
      converted_values.push_back(std::move(converted));
    }
    for (size_t i = 0; i < handles.size(); ++i) arguments[i][field->second] = std::move(converted_values[i]);
  }
};

void submit(const std::vector<std::shared_ptr<NativeLane>>& lanes, size_t chunk) {
  TORCH_CHECK(!lanes.empty(), "nonempty lane list required");
  size_t count = 0;
  for (const auto& lane : lanes) count = std::max(count, lane->handles.size());
  const size_t width = chunk ? chunk : count;
  py::gil_scoped_release release;
  for (size_t begin = 0; begin < count; begin += width) {
    for (const auto& lane : lanes) {
      lane->update(begin, std::min(begin + width, lane->handles.size()));
    }
  }
}

struct NativeReplay {
  py::object graph_owner;
  py::object replay_method;
  c10_npu::NPUStream main_stream;
  c10_npu::NPUStream graph_stream;
  std::vector<std::shared_ptr<NativeLane>> lanes;
  std::vector<c10_npu::NPUEvent> updated;
  c10_npu::NPUEvent ready;
  c10_npu::NPUEvent done;
  std::vector<py::object> timing_owners;
  std::vector<c10_npu::NPUEvent*> timing;

  static c10_npu::NPUStream unpack(py::object stream) {
    return c10_npu::NPUStream::unpack3(stream.attr("stream_id").cast<int64_t>(),
                                       stream.attr("device_index").cast<int>(),
                                       static_cast<c10::DeviceType>(stream.attr("device_type").cast<int>()));
  }

  NativeReplay(py::object graph, py::object main, py::object replay_stream,
               std::vector<std::shared_ptr<NativeLane>> plans, py::sequence timing_events)
      : graph_owner(graph),
        replay_method(graph.attr("replay")),
        main_stream(unpack(main)),
        graph_stream(unpack(replay_stream)),
        lanes(std::move(plans)),
        updated(lanes.size()) {
    TORCH_CHECK(py::isinstance(graph, py::module_::import("torch_npu").attr("_C").attr("_NPUGraph")),
                "expected NPU graph");
    TORCH_CHECK(!lanes.empty(), "nonempty replay update plans required");
    TORCH_CHECK(main_stream.device_index() == graph_stream.device_index(), "replay stream device mismatch");
    for (const auto& lane : lanes)
      TORCH_CHECK(lane->stream.device_index() == main_stream.device_index(), "update stream device mismatch");
    TORCH_CHECK(timing_events.size() == 0 || timing_events.size() == 2, "expected zero or two timing events");
    auto event_type = py::module_::import("torch_npu").attr("_C").attr("_NPUEventBase");
    for (auto item : timing_events) {
      auto event = py::reinterpret_borrow<py::object>(item);
      TORCH_CHECK(py::isinstance(event, event_type), "expected NPU timing event");
      timing_owners.push_back(event);
      timing.push_back(&reinterpret_cast<THNPEvent*>(event.ptr())->npu_event);
    }
  }

  void replay(bool diagnostic) {
    TORCH_CHECK(!diagnostic || timing.size() == 2, "diagnostic timing events were not configured");
    py::gil_scoped_release release;
    ready.record(main_stream);
    ready.block(graph_stream);
    if (diagnostic) timing[0]->record(graph_stream);
    {
      c10_npu::NPUStreamGuard guard(graph_stream);
      // NPUGraph has no public Python-object layout. Invoke its retained public
      // replay binding, rather than reinterpreting undocumented object memory.
      py::gil_scoped_acquire acquire;
      replay_method();
    }
    for (const auto& lane : lanes) ready.block(lane->stream);
    for (const auto& lane : lanes) lane->update(0, lane->handles.size());
    for (size_t i = 0; i < lanes.size(); ++i) {
      updated[i].record(lanes[i]->stream);
      updated[i].block(graph_stream);
    }
    done.record(graph_stream);
    if (diagnostic) timing[1]->record(graph_stream);
    done.block(main_stream);
  }
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  py::class_<NativeLane, std::shared_ptr<NativeLane>>(module, "NativeLane")
      .def(py::init<py::sequence, py::object>())
      .def("bind", &NativeLane::bind)
      .def("bind_kv_lengths", &NativeLane::bind_kv_lengths);
  module.def("submit", &submit);
  py::class_<NativeReplay>(module, "NativeReplay")
      .def(py::init<py::object, py::object, py::object, std::vector<std::shared_ptr<NativeLane>>, py::sequence>())
      .def("replay", &NativeReplay::replay, py::arg("diagnostic") = false);
}
