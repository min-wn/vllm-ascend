// SPDX-License-Identifier: Apache-2.0
#include <torch/extension.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/framework/OpCommand.h>

extern "C" void q4_tile_transport_launch(uint32_t, void*, void*, void*, void*, void*, void*, void*, void*, void*,
                                         uint32_t, uint32_t, uint64_t, bool, uint32_t, uint32_t, uint32_t);
extern "C" void q4_tile_solve_launch(uint32_t, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*,
                                     void*, void*, void*, void*, void*, void*, void*, void*, uint32_t, uint32_t,
                                     uint64_t, bool, uint32_t, uint32_t);
extern "C" void q4_tile_consume_launch(uint32_t, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*,
                                       void*, void*, void*, void*, void*, void*, uint32_t, uint32_t, uint64_t, uint32_t,
                                       uint32_t);

namespace {
void check(const at::Tensor& tensor, at::ScalarType dtype, int64_t rank, bool contiguous = true) {
  TORCH_CHECK(tensor.device().type() == c10::DeviceType::PrivateUse1, "NPU tensors required");
  TORCH_CHECK(tensor.scalar_type() == dtype && tensor.dim() == rank && (!contiguous || tensor.is_contiguous()),
              "tensor dtype, rank or contiguity mismatch");
}

void disjoint(const at::Tensor& first, const at::Tensor& second) {
  const auto start = [](const at::Tensor& tensor) { return reinterpret_cast<uintptr_t>(tensor.data_ptr()); };
  const auto bytes = [](const at::Tensor& tensor) {
    const auto elements = tensor.dim() == 4 && !tensor.is_contiguous()
                              ? (tensor.size(0) - 1) * tensor.stride(0) + 24 * 128 * 128
                              : tensor.numel();
    return static_cast<uintptr_t>(elements) * tensor.element_size();
  };
  TORCH_CHECK(start(first) + bytes(first) <= start(second) || start(second) + bytes(second) <= start(first),
              "mutable and read-only buffers must not overlap");
}

int64_t solve_contract(const at::Tensor& products, const at::Tensor& query, const at::Tensor& key,
                       const at::Tensor& value, const at::Tensor& beta, const at::Tensor& gate, const at::Tensor& state,
                       const at::Tensor& sequence, const at::Tensor& ids, const at::Tensor& accepted,
                       const at::Tensor& output, const at::Tensor& logs, const at::Tensor& valid,
                       const at::Tensor& newLogs, const at::Tensor& newValid, int64_t requestStart,
                       int64_t& requestStop) {
  check(state, at::kFloat, 4, false);
  TORCH_CHECK(state.size(0) >= 1 && state.size(1) == 24 && state.size(2) == 128 && state.size(3) == 128 &&
                  state.stride(1) == 16384 && state.stride(2) == 128 && state.stride(3) == 1 &&
                  state.stride(0) >= 24 * 16384,
              "FP32 state with contiguous inner dimensions required");
  for (const auto& tensor : {logs, newLogs}) {
    check(tensor, at::kFloat, 3);
    TORCH_CHECK(tensor.size(0) == state.size(0) && tensor.size(1) == 24 && tensor.size(2) == 1056,
                "log bank must be [Nslots,24,1056]");
  }
  for (const auto& tensor : {valid, newValid}) {
    check(tensor, at::kInt, 3);
    TORCH_CHECK(tensor.size(0) == state.size(0) && tensor.size(1) == 24 && tensor.size(2) == 8,
                "validity bank must be [Nslots,24,8]");
  }
  for (const auto& tensor : {sequence, ids, accepted}) check(tensor, at::kInt, 1);
  const auto batch = accepted.numel(), tokens = 4 * batch;
  if (requestStop == -1) requestStop = batch;
  TORCH_CHECK(batch >= 1 && batch <= 128 && sequence.numel() == batch + 1 && ids.numel() == tokens,
              "invalid fixed-capacity Q4 metadata");
  TORCH_CHECK(requestStart >= 0 && requestStart < requestStop && requestStop <= batch, "invalid task range");
  check(products, at::kFloat, 4);
  for (const auto& tensor : {query, key, value, output}) check(tensor, at::kBFloat16, 3);
  check(beta, at::kBFloat16, 2);
  check(gate, at::kFloat, 2);
  TORCH_CHECK(query.size(0) == tokens && query.size(1) == 8 && query.size(2) == 128 && key.sizes() == query.sizes(),
              "raw Q/K must be [4B,8,128]");
  TORCH_CHECK(value.size(0) == tokens && value.size(1) == 24 && value.size(2) == 128 && output.sizes() == value.sizes(),
              "V/output must be [4B,24,128]");
  TORCH_CHECK(beta.size(0) == tokens && beta.size(1) == 24 && gate.sizes() == beta.sizes(), "invalid beta/gate");
  TORCH_CHECK(products.size(0) == batch && products.size(1) == 24 && products.size(2) == 128 && products.size(3) == 8,
              "products must be [B,24,128,8]");
  const std::vector<at::Tensor> readOnly{query, key, value, beta, gate, state, sequence, ids, accepted, logs, valid};
  const std::vector<at::Tensor> mutableBuffers{products, output, newLogs, newValid};
  for (const auto& tensor : readOnly) TORCH_CHECK(tensor.device() == state.device(), "one NPU required");
  for (size_t index = 0; index < mutableBuffers.size(); ++index) {
    TORCH_CHECK(mutableBuffers[index].device() == state.device(), "one NPU required");
    for (const auto& input : readOnly) disjoint(mutableBuffers[index], input);
    for (size_t other = index + 1; other < mutableBuffers.size(); ++other) {
      disjoint(mutableBuffers[index], mutableBuffers[other]);
    }
  }
  return batch;
}

void solve(at::Tensor state, at::Tensor queryFloat, at::Tensor keyFloat, at::Tensor query, at::Tensor key,
           at::Tensor value, at::Tensor beta, at::Tensor gate, at::Tensor sequence, at::Tensor ids, at::Tensor accepted,
           at::Tensor products, at::Tensor output, at::Tensor logs, at::Tensor valid, at::Tensor newLogs,
           at::Tensor newValid, int64_t cores, bool streaming, int64_t requestStart, int64_t requestStop) {
  TORCH_CHECK(cores >= 1 && cores <= 20, "1..20 Cube cores required");
  const auto batch = solve_contract(products, query, key, value, beta, gate, state, sequence, ids, accepted, output,
                                    logs, valid, newLogs, newValid, requestStart, requestStop);
  for (const auto& tensor : {queryFloat, keyFloat}) {
    check(tensor, at::kFloat, 3);
    TORCH_CHECK(tensor.sizes() == query.sizes() && tensor.device() == state.device(), "FP32 Q/K shape mismatch");
    for (const auto& buffer : {products, output, newLogs, newValid}) disjoint(tensor, buffer);
  }
  auto stream = c10_npu::getCurrentNPUStream().stream();
  at_npu::native::OpCommand command;
  command.Name("GdnQ4A2TileSolvePrototype");
  command.SetCustomHandler([=]() -> int {
    q4_tile_solve_launch(cores, stream, state.data_ptr(), queryFloat.data_ptr(), keyFloat.data_ptr(), query.data_ptr(),
                         key.data_ptr(), value.data_ptr(), beta.data_ptr(), gate.data_ptr(), sequence.data_ptr(),
                         ids.data_ptr(), accepted.data_ptr(), products.data_ptr(), output.data_ptr(), logs.data_ptr(),
                         valid.data_ptr(), newLogs.data_ptr(), newValid.data_ptr(), batch, state.size(0),
                         state.stride(0), streaming, requestStart, requestStop);
    return 0;
  });
  command.Run();
}

void consume(at::Tensor products, at::Tensor query, at::Tensor key, at::Tensor value, at::Tensor beta, at::Tensor gate,
             at::Tensor state, at::Tensor sequence, at::Tensor ids, at::Tensor accepted, at::Tensor output,
             at::Tensor logs, at::Tensor valid, at::Tensor newLogs, at::Tensor newValid, int64_t cores,
             int64_t requestStart, int64_t requestStop) {
  TORCH_CHECK(cores >= 2 && cores <= 40 && cores % 2 == 0, "even 2..40 AIV cores required");
  const auto batch = solve_contract(products, query, key, value, beta, gate, state, sequence, ids, accepted, output,
                                    logs, valid, newLogs, newValid, requestStart, requestStop);
  auto stream = c10_npu::getCurrentNPUStream().stream();
  at_npu::native::OpCommand command;
  command.Name("GdnQ4A2TileConsumePrototype");
  command.SetCustomHandler([=]() -> int {
    q4_tile_consume_launch(cores / 2, stream, products.data_ptr(), query.data_ptr(), key.data_ptr(), value.data_ptr(),
                           beta.data_ptr(), gate.data_ptr(), state.data_ptr(), sequence.data_ptr(), ids.data_ptr(),
                           accepted.data_ptr(), output.data_ptr(), logs.data_ptr(), valid.data_ptr(),
                           newLogs.data_ptr(), newValid.data_ptr(), batch, state.size(0), state.stride(0), requestStart,
                           requestStop);
    return 0;
  });
  command.Run();
}

void transport(at::Tensor state, at::Tensor query, at::Tensor key, at::Tensor sequence, at::Tensor ids,
               at::Tensor accepted, at::Tensor products, at::Tensor transported, int64_t cores, bool streamed,
               int64_t vectorIterations, int64_t requestStart, int64_t requestStop) {
  check(state, at::kFloat, 4, false);
  TORCH_CHECK(state.stride(3) == 1 && state.stride(2) == 128 && state.stride(1) == 128 * 128 &&
                  state.stride(0) >= 24 * 128 * 128,
              "state requires contiguous inner dimensions with optional page stride");
  for (const auto& tensor : {query, key}) check(tensor, at::kFloat, 3);
  for (const auto& tensor : {sequence, ids, accepted}) check(tensor, at::kInt, 1);
  for (const auto& tensor : {products, transported}) check(tensor, at::kFloat, 4);
  const int64_t batch = accepted.numel(), tokens = batch * 4;
  if (requestStop == -1) requestStop = batch;
  TORCH_CHECK(batch >= 1 && batch <= 128 && cores >= 1 && cores <= 20, "invalid A2 prototype extent");
  TORCH_CHECK(vectorIterations >= 0 && vectorIterations <= 4096, "invalid synthetic vector work count");
  TORCH_CHECK(requestStart >= 0 && requestStart < requestStop && requestStop <= batch, "invalid task range");
  for (const auto& tensor : {query, key, sequence, ids, accepted, products, transported}) {
    TORCH_CHECK(tensor.device() == state.device(), "all buffers must be on one NPU");
  }
  TORCH_CHECK(state.size(0) >= 1 && state.size(1) == 24 && state.size(2) == 128 && state.size(3) == 128,
              "state must be [Nslots,24,128,128]");
  TORCH_CHECK(query.size(0) == tokens && query.size(1) == 8 && query.size(2) == 128 && key.sizes() == query.sizes(),
              "Q/K must be [4B,8,128]");
  TORCH_CHECK(products.size(0) == batch && products.size(1) == 24 && products.size(2) == 128 && products.size(3) == 8 &&
                  transported.sizes() == products.sizes(),
              "products and transported must be [B,24,128,8]");
  TORCH_CHECK(sequence.numel() == batch + 1 && ids.numel() == tokens, "invalid Q4 metadata shape");
  // The transport copy must not mutate its producer's read-only inputs.
  const auto stateStart = reinterpret_cast<uintptr_t>(state.data_ptr());
  const auto stateEnd = stateStart + ((state.size(0) - 1) * state.stride(0) + 24 * 128 * 128) * sizeof(float);
  const auto productsStart = reinterpret_cast<uintptr_t>(products.data_ptr());
  const auto productsEnd = productsStart + products.nbytes();
  const auto transportedStart = reinterpret_cast<uintptr_t>(transported.data_ptr());
  const auto transportedEnd = transportedStart + transported.nbytes();
  TORCH_CHECK(productsEnd <= stateStart || stateEnd <= productsStart, "products must not alias state");
  TORCH_CHECK(transportedEnd <= stateStart || stateEnd <= transportedStart, "transport must not alias state");
  TORCH_CHECK(productsEnd <= transportedStart || transportedEnd <= productsStart, "transport must not alias products");
  auto stream = c10_npu::getCurrentNPUStream().stream();
  at_npu::native::OpCommand command;
  command.Name("GdnQ4A2TileTransportPrototype");
  command.SetCustomHandler([=]() -> int {
    q4_tile_transport_launch(cores, stream, state.data_ptr(), query.data_ptr(), key.data_ptr(), sequence.data_ptr(),
                             ids.data_ptr(), accepted.data_ptr(), products.data_ptr(), transported.data_ptr(), batch,
                             state.size(0), state.stride(0), streamed, vectorIterations, requestStart, requestStop);
    return 0;
  });
  command.Run();
}
}  // namespace

PYBIND11_MODULE(gdn_q4_tile_pipeline, module) {
  namespace py = pybind11;
  module.def("transport", &transport, py::arg("state"), py::arg("query"), py::arg("key"), py::arg("sequence"),
             py::arg("state_ids"), py::arg("accepted"), py::arg("products"), py::arg("transported"),
             py::arg("cores") = 20, py::arg("streaming") = true, py::arg("vector_iterations") = 1,
             py::arg("request_start") = 0, py::arg("request_stop") = -1);
  module.def("solve", &solve, py::arg("state"), py::arg("query_float"), py::arg("key_float"), py::arg("query"),
             py::arg("key"), py::arg("value"), py::arg("beta"), py::arg("gate"), py::arg("sequence"),
             py::arg("state_ids"), py::arg("accepted"), py::arg("products"), py::arg("output"), py::arg("old_logs"),
             py::arg("old_valid"), py::arg("new_logs"), py::arg("new_valid"), py::arg("cores") = 20,
             py::arg("streaming") = true, py::arg("request_start") = 0, py::arg("request_stop") = -1);
  module.def("consume", &consume, py::arg("products"), py::arg("query"), py::arg("key"), py::arg("value"),
             py::arg("beta"), py::arg("gate"), py::arg("state"), py::arg("sequence"), py::arg("state_ids"),
             py::arg("accepted"), py::arg("output"), py::arg("old_logs"), py::arg("old_valid"), py::arg("new_logs"),
             py::arg("new_valid"), py::arg("cores") = 40, py::arg("request_start") = 0, py::arg("request_stop") = -1);
}
