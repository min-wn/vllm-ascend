// SPDX-License-Identifier: Apache-2.0
#include <torch/extension.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/framework/OpCommand.h>

extern "C" void replay_solve_launch(uint32_t, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*,
                                    void*, void*, void*, void*, void*, void*, void*, uint32_t, uint32_t, uint64_t,
                                    bool);
extern "C" void replay_state_launch(void*, void*, void*, void*, void*, void*, void*, void*, uint32_t, uint32_t);

namespace {
void check(const at::Tensor& tensor, at::ScalarType type, at::IntArrayRef shape) {
  TORCH_CHECK(tensor.device().type() == c10::DeviceType::PrivateUse1 && tensor.scalar_type() == type &&
                  tensor.sizes() == shape && tensor.is_contiguous(),
              "ReplaySSM tensor contract mismatch");
}

void common(const at::Tensor& state, const at::Tensor& ring, const at::Tensor& gates, const at::Tensor& cursors,
            const at::Tensor& sequence, const at::Tensor& ids, const at::Tensor& accepted) {
  TORCH_CHECK(state.dim() == 4 && state.size(0) >= 2 && state.size(0) <= 129, "invalid owner capacity");
  const auto slots = state.size(0), batch = accepted.numel();
  TORCH_CHECK(batch >= 1 && batch <= 128, "invalid batch capacity");
  check(state, at::kFloat, {slots, 24, 128, 128});
  check(ring, at::kFloat, {slots, 24, 32, 256});
  check(gates, at::kFloat, {slots, 24, 32, 8});
  check(cursors, at::kInt, {slots, 4, 8});
  check(sequence, at::kInt, {batch + 1});
  check(ids, at::kInt, {batch * 4});
  check(accepted, at::kInt, {batch});
  const std::vector<at::Tensor> tensors{state, ring, gates, cursors, sequence, ids, accepted};
  for (size_t i = 0; i < tensors.size(); ++i) {
    TORCH_CHECK(tensors[i].device() == state.device(), "all tensors must use one device");
    for (size_t j = 0; j < i; ++j) {
      const auto a = reinterpret_cast<uintptr_t>(tensors[i].data_ptr());
      const auto b = reinterpret_cast<uintptr_t>(tensors[j].data_ptr());
      TORCH_CHECK(a + tensors[i].nbytes() <= b || b + tensors[j].nbytes() <= a, "buffers must not alias");
    }
  }
}

void state_op(const at::Tensor& state, const at::Tensor& ring, const at::Tensor& gates, const at::Tensor& cursors,
              const at::Tensor& sequence, const at::Tensor& ids, const at::Tensor& accepted, int64_t phase) {
  common(state, ring, gates, cursors, sequence, ids, accepted);
  TORCH_CHECK(phase >= 0 && phase <= 3, "phase is advance/flush/drain/reset");
  const auto stream = c10_npu::getCurrentNPUStream().stream();
  at_npu::native::OpCommand command;
  command.Name("ReplaySsmA2StatePrototype");
  command.SetCustomHandler([=]() -> int {
    replay_state_launch(stream, state.data_ptr(), ring.data_ptr(), gates.data_ptr(), cursors.data_ptr(),
                        sequence.data_ptr(), ids.data_ptr(), accepted.data_ptr(), accepted.numel(), phase);
    return 0;
  });
  command.Run();
}

void solve(const at::Tensor& state, const at::Tensor& queryFloat, const at::Tensor& keyFloat, const at::Tensor& query,
           const at::Tensor& key, const at::Tensor& value, const at::Tensor& beta, const at::Tensor& gate,
           const at::Tensor& sequence, const at::Tensor& ids, const at::Tensor& accepted, const at::Tensor& products,
           const at::Tensor& output, const at::Tensor& ring, const at::Tensor& gates, const at::Tensor& cursors,
           bool streamed) {
  common(state, ring, gates, cursors, sequence, ids, accepted);
  const auto batch = accepted.numel(), tokens = batch * 4;
  for (const auto& tensor : {queryFloat, keyFloat}) check(tensor, at::kFloat, {tokens, 8, 128});
  for (const auto& tensor : {query, key}) check(tensor, at::kBFloat16, {tokens, 8, 128});
  for (const auto& tensor : {value, output}) check(tensor, at::kBFloat16, {tokens, 24, 128});
  check(beta, at::kBFloat16, {tokens, 24});
  check(gate, at::kFloat, {tokens, 24});
  check(products, at::kFloat, {batch, 24, 128, 8});
  const std::vector<at::Tensor> tensors{state,    queryFloat, keyFloat, query,    key,    value, beta,  gate,
                                        sequence, ids,        accepted, products, output, ring,  gates, cursors};
  for (const auto& tensor : tensors) TORCH_CHECK(tensor.device() == state.device(), "mixed device operands");
  // Inputs can share read-only storage, but all mutable buffers are exclusive.
  for (size_t i : {size_t(0), size_t(11), size_t(12), size_t(13), size_t(14), size_t(15)}) {
    const auto a = reinterpret_cast<uintptr_t>(tensors[i].data_ptr());
    for (size_t j = 0; j < tensors.size(); ++j) {
      if (i == j) continue;
      const auto b = reinterpret_cast<uintptr_t>(tensors[j].data_ptr());
      TORCH_CHECK(a + tensors[i].nbytes() <= b || b + tensors[j].nbytes() <= a, "mutable buffers must not alias");
    }
  }
  const auto stream = c10_npu::getCurrentNPUStream().stream();
  at_npu::native::OpCommand command;
  command.Name("ReplaySsmA2ReadyPrototype");
  command.SetCustomHandler([=]() -> int {
    replay_solve_launch(20, stream, state.data_ptr(), queryFloat.data_ptr(), keyFloat.data_ptr(), query.data_ptr(),
                        key.data_ptr(), value.data_ptr(), beta.data_ptr(), gate.data_ptr(), sequence.data_ptr(),
                        ids.data_ptr(), accepted.data_ptr(), products.data_ptr(), output.data_ptr(), ring.data_ptr(),
                        cursors.data_ptr(), gates.data_ptr(), batch, state.size(0), state.stride(0), streamed);
    return 0;
  });
  command.Run();
}
}  // namespace

PYBIND11_MODULE(gdn_replayssm, module) {
  module.def("state", &state_op);
  module.def("solve", &solve);
}
