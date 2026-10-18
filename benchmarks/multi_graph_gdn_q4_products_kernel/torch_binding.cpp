// SPDX-License-Identifier: Apache-2.0
#include <torch/extension.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/framework/OpCommand.h>

extern "C" void q4_products_launch(uint32_t, void*, void*, void*, void*, void*, void*, void*, void*, uint32_t, uint32_t,
                                   uint64_t, bool);

namespace {
void check(const at::Tensor& tensor, at::ScalarType dtype, int64_t rank, bool contiguous = true) {
  TORCH_CHECK(tensor.device().type() == c10::DeviceType::PrivateUse1, "NPU tensors required");
  TORCH_CHECK(tensor.scalar_type() == dtype && tensor.dim() == rank && (!contiguous || tensor.is_contiguous()),
              "contiguous tensor with declared dtype and rank required");
}

void products(at::Tensor state, at::Tensor query, at::Tensor key, at::Tensor sequence, at::Tensor stateIds,
              at::Tensor accepted, at::Tensor output, int64_t cores) {
  TORCH_CHECK(state.scalar_type() == at::kBFloat16 || state.scalar_type() == at::kFloat, "BF16/FP32 state required");
  check(state, state.scalar_type(), 4, false);
  TORCH_CHECK(state.stride(3) == 1 && state.stride(2) == 128 && state.stride(1) == 128 * 128 &&
                  state.stride(0) >= 24 * 128 * 128,
              "state requires contiguous heads/rows with optional page padding");
  for (const auto& tensor : {query, key}) check(tensor, state.scalar_type(), 3);
  for (const auto& tensor : {sequence, stateIds, accepted}) check(tensor, at::kInt, 1);
  check(output, at::kFloat, 4);
  const int64_t batch = accepted.numel(), tokens = batch * 4;
  TORCH_CHECK(batch >= 1 && batch <= 128 && cores >= 1 && cores <= 20, "invalid A2 Cube prototype extent");
  for (const auto& tensor : {query, key, sequence, stateIds, accepted, output}) {
    TORCH_CHECK(tensor.device() == state.device(), "all buffers must be on one NPU");
  }
  TORCH_CHECK(query.size(0) == tokens && query.size(1) == 8 && query.size(2) == 128 && key.sizes() == query.sizes(),
              "Q/K must have shape [4B,8,128]");
  TORCH_CHECK(state.size(0) >= 1 && state.size(1) == 24 && state.size(2) == 128 && state.size(3) == 128,
              "read-only state must have shape [Nslots>=1,24,128,128]");
  TORCH_CHECK(output.size(0) == batch && output.size(1) == 24 && output.size(2) == 128 && output.size(3) == 8,
              "products output must have shape [B,24,128,8]");
  TORCH_CHECK(sequence.numel() == batch + 1 && stateIds.numel() == tokens, "invalid Q4 metadata shape");
  auto stream = c10_npu::getCurrentNPUStream().stream();
  at_npu::native::OpCommand command;
  command.Name("GdnQ4DirectCubeProductsPrototype");
  command.SetCustomHandler([=]() -> int {
    q4_products_launch(cores, stream, state.data_ptr(), query.data_ptr(), key.data_ptr(), sequence.data_ptr(),
                       stateIds.data_ptr(), accepted.data_ptr(), output.data_ptr(), batch, state.size(0),
                       state.stride(0), state.scalar_type() == at::kFloat);
    return 0;
  });
  command.Run();
}
}  // namespace

PYBIND11_MODULE(gdn_q4_products, module) {
  namespace py = pybind11;
  module.def("products", &products, py::arg("state"), py::arg("query"), py::arg("key"), py::arg("sequence"),
             py::arg("state_ids"), py::arg("accepted"), py::arg("output"), py::arg("cores") = 20);
}
