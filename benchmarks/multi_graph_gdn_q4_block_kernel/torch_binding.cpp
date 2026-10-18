// SPDX-License-Identifier: Apache-2.0
#include <torch/extension.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/framework/OpCommand.h>

extern "C" void q4_block_launch(uint32_t, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*,
                                void*, uint32_t, uint32_t, uint64_t, bool);

namespace {
void check(const at::Tensor& tensor, at::ScalarType dtype, int64_t rank, bool contiguous = true) {
  TORCH_CHECK(tensor.device().type() == c10::DeviceType::PrivateUse1, "NPU tensors required");
  TORCH_CHECK(tensor.scalar_type() == dtype && tensor.dim() == rank && (!contiguous || tensor.is_contiguous()),
              "contiguous tensor with declared dtype and rank required");
}

void emit(at::Tensor products, at::Tensor query, at::Tensor key, at::Tensor value, at::Tensor beta, at::Tensor gate,
          at::Tensor state, at::Tensor sequence, at::Tensor stateIds, at::Tensor accepted, at::Tensor output,
          int64_t cores) {
  check(products, at::kFloat, 4);
  for (const auto& tensor : {query, key, value, output}) check(tensor, at::kBFloat16, 3);
  check(beta, at::kBFloat16, 2);
  check(gate, at::kFloat, 2);
  TORCH_CHECK(state.scalar_type() == at::kBFloat16 || state.scalar_type() == at::kFloat, "BF16/FP32 state required");
  check(state, state.scalar_type(), 4, false);
  TORCH_CHECK(state.stride(3) == 1 && state.stride(2) == 128 && state.stride(1) == 128 * 128 &&
                  state.stride(0) >= 24 * 128 * 128,
              "state requires contiguous heads/rows with optional page padding");
  for (const auto& tensor : {sequence, stateIds, accepted}) check(tensor, at::kInt, 1);
  const int64_t batch = accepted.numel(), tokens = batch * 4;
  TORCH_CHECK(batch >= 1 && batch <= 128 && cores >= 1 && cores <= 40, "invalid A2 prototype extent");
  for (const auto& tensor : {products, key, value, beta, gate, state, sequence, stateIds, accepted, output}) {
    TORCH_CHECK(tensor.device() == query.device(), "all buffers must be on the same NPU");
  }
  TORCH_CHECK(query.size(0) == tokens && query.size(1) == 8 && query.size(2) == 128 && key.sizes() == query.sizes(),
              "Q/K must have shape [4B,8,128]");
  TORCH_CHECK(value.size(0) == tokens && value.size(1) == 24 && value.size(2) == 128 && output.sizes() == value.sizes(),
              "V/output must have shape [4B,24,128]");
  TORCH_CHECK(beta.size(0) == tokens && beta.size(1) == 24 && gate.sizes() == beta.sizes(),
              "beta/gate must have shape [4B,24]");
  TORCH_CHECK(products.size(0) == batch && products.size(1) == 24 && products.size(2) == 128 && products.size(3) == 8,
              "products must have shape [B,24,128,8]");
  TORCH_CHECK(state.size(0) >= 1 + tokens && state.size(1) == 24 && state.size(2) == 128 && state.size(3) == 128,
              "state must have shape [Nslots>=1+4B,24,128,128]");
  TORCH_CHECK(sequence.numel() == batch + 1 && stateIds.numel() == tokens, "invalid Q4 metadata shape");
  auto stream = c10_npu::getCurrentNPUStream().stream();
  at_npu::native::OpCommand command;
  command.Name("GdnQ4BlockPrototype");
  command.SetCustomHandler([=]() -> int {
    q4_block_launch(cores, stream, products.data_ptr(), query.data_ptr(), key.data_ptr(), value.data_ptr(),
                    beta.data_ptr(), gate.data_ptr(), state.data_ptr(), sequence.data_ptr(), stateIds.data_ptr(),
                    accepted.data_ptr(), output.data_ptr(), batch, state.size(0), state.stride(0),
                    state.scalar_type() == at::kFloat);
    return 0;
  });
  command.Run();
}
}  // namespace

PYBIND11_MODULE(gdn_q4_block, module) {
  namespace py = pybind11;
  module.def("emit", &emit, py::arg("products"), py::arg("query"), py::arg("key"), py::arg("value"), py::arg("beta"),
             py::arg("gate"), py::arg("state"), py::arg("sequence"), py::arg("state_ids"), py::arg("accepted"),
             py::arg("output"), py::arg("cores") = 40);
}
