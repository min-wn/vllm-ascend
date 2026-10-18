// SPDX-License-Identifier: Apache-2.0
#include <torch/extension.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/framework/OpCommand.h>

extern "C" void q4_log_launch(uint32_t cores, void* stream, void* products, void* query, void* key, void* value,
                              void* beta, void* gate, void* state, void* sequence, void* ids, void* accepted,
                              void* output, void* logs, void* valid, uint32_t batch, uint32_t slots,
                              uint64_t stateStride, bool exportOnly);

namespace {
void check(const at::Tensor& tensor, at::ScalarType dtype, int64_t rank, bool contiguous = true) {
  TORCH_CHECK(tensor.device().type() == c10::DeviceType::PrivateUse1, "NPU tensors required");
  TORCH_CHECK(tensor.scalar_type() == dtype && tensor.dim() == rank && (!contiguous || tensor.is_contiguous()),
              "tensor dtype, rank or layout disagree");
}

void state_contract(const at::Tensor& state, const at::Tensor& logs, const at::Tensor& valid, int64_t cores) {
  check(state, at::kFloat, 4, false);
  check(logs, at::kFloat, 3);
  check(valid, at::kInt, 3);
  TORCH_CHECK(state.size(0) >= 1 && state.size(1) == 24 && state.size(2) == 128 && state.size(3) == 128 &&
                  state.stride(1) == 16384 && state.stride(2) == 128 && state.stride(3) == 1 &&
                  state.stride(0) >= 24 * 16384,
              "FP32 TP2 state with contiguous inner dimensions required");
  TORCH_CHECK(logs.size(0) == state.size(0) && logs.size(1) == 24 && logs.size(2) == 1056 &&
                  valid.size(0) == state.size(0) && valid.size(1) == 24 && valid.size(2) == 8,
              "one 1056-float log and padded eight-int valid record per request/head required");
  TORCH_CHECK(logs.device() == state.device() && valid.device() == state.device() && cores >= 1 && cores <= 40,
              "buffers must share a device and use 1..40 AIV cores");
}

void step(at::Tensor products, at::Tensor query, at::Tensor key, at::Tensor value, at::Tensor beta, at::Tensor gate,
          at::Tensor state, at::Tensor sequence, at::Tensor ids, at::Tensor accepted, at::Tensor output,
          at::Tensor logs, at::Tensor valid, int64_t cores) {
  state_contract(state, logs, valid, cores);
  check(products, at::kFloat, 4);
  for (const auto& tensor : {query, key, value, output}) check(tensor, at::kBFloat16, 3);
  check(beta, at::kBFloat16, 2);
  check(gate, at::kFloat, 2);
  for (const auto& tensor : {sequence, ids, accepted}) check(tensor, at::kInt, 1);
  const auto batch = accepted.numel(), tokens = 4 * batch;
  TORCH_CHECK(batch >= 1 && batch <= 128, "1..128 fixed-capacity Q4 requests required");
  TORCH_CHECK(query.size(0) == tokens && query.size(1) == 8 && query.size(2) == 128 && key.sizes() == query.sizes(),
              "raw Q/K require [4B,8,128]");
  TORCH_CHECK(value.size(0) == tokens && value.size(1) == 24 && value.size(2) == 128 && output.sizes() == value.sizes(),
              "V/output require [4B,24,128]");
  TORCH_CHECK(beta.size(0) == tokens && beta.size(1) == 24 && gate.sizes() == beta.sizes(), "invalid gate/beta shape");
  TORCH_CHECK(products.size(0) == batch && products.size(1) == 24 && products.size(2) == 128 && products.size(3) == 8,
              "base products require [B,24,128,8]");
  TORCH_CHECK(sequence.numel() == batch + 1 && ids.numel() == tokens, "invalid fixed-address metadata");
  for (const auto& tensor : {products, query, key, value, beta, gate, sequence, ids, accepted, output}) {
    TORCH_CHECK(tensor.device() == state.device(), "all tensors must share one NPU");
  }
  auto stream = c10_npu::getCurrentNPUStream().stream();
  at_npu::native::OpCommand command;
  command.Name("GdnQ4LogPrototype");
  command.SetCustomHandler([=]() -> int {
    q4_log_launch(cores, stream, products.data_ptr(), query.data_ptr(), key.data_ptr(), value.data_ptr(),
                  beta.data_ptr(), gate.data_ptr(), state.data_ptr(), sequence.data_ptr(), ids.data_ptr(),
                  accepted.data_ptr(), output.data_ptr(), logs.data_ptr(), valid.data_ptr(), batch, state.size(0),
                  state.stride(0), false);
    return 0;
  });
  command.Run();
}

void export_candidates(at::Tensor state, at::Tensor logs, at::Tensor valid, at::Tensor output, int64_t cores) {
  state_contract(state, logs, valid, cores);
  check(output, at::kFloat, 5);
  TORCH_CHECK(output.device() == state.device() && output.size(0) == state.size(0) && output.size(1) == 4 &&
                  output.size(2) == 24 && output.size(3) == 128 && output.size(4) == 128,
              "export destination requires [Nrequest,4,24,128,128]");
  auto stream = c10_npu::getCurrentNPUStream().stream();
  at_npu::native::OpCommand command;
  command.Name("GdnQ4LogExportPrototype");
  command.SetCustomHandler([=]() -> int {
    q4_log_launch(cores, stream, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, state.data_ptr(), nullptr,
                  nullptr, nullptr, output.data_ptr(), logs.data_ptr(), valid.data_ptr(), 0, state.size(0),
                  state.stride(0), true);
    return 0;
  });
  command.Run();
}
}  // namespace

PYBIND11_MODULE(gdn_q4_log, module) {
  namespace py = pybind11;
  module.def("step", &step, py::arg("products"), py::arg("query"), py::arg("key"), py::arg("value"), py::arg("beta"),
             py::arg("gate"), py::arg("state"), py::arg("sequence"), py::arg("ids"), py::arg("accepted"),
             py::arg("output"), py::arg("logs"), py::arg("valid"), py::arg("cores") = 40);
  module.def("export_candidates", &export_candidates, py::arg("state"), py::arg("logs"), py::arg("valid"),
             py::arg("output"), py::arg("cores") = 40);
}
