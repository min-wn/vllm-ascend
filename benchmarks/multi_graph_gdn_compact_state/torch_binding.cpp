#include <torch/extension.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/framework/OpCommand.h>

extern "C" void compact_emit_launch(uint32_t, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*,
                                    void*, void*, void*, void*, void*, void*, void*, uint32_t, uint32_t, uint32_t,
                                    uint32_t, bool, bool);
extern "C" void compact_reconstruct_launch(uint32_t, void*, void*, void*, void*, void*, void*, void*, void*, uint32_t,
                                           uint32_t);

namespace {
void check(const at::Tensor& value, at::ScalarType dtype, int64_t dimension) {
  TORCH_CHECK(value.device().type() == c10::DeviceType::PrivateUse1, "NPU tensor required");
  TORCH_CHECK(value.scalar_type() == dtype && value.dim() == dimension && value.is_contiguous(),
              "contiguous tensor of the declared dtype and rank required");
}

void run_emit(at::Tensor query, at::Tensor key, at::Tensor value, at::Tensor beta, at::Tensor gate, at::Tensor state,
              at::Tensor sequence, at::Tensor stateIds, at::Tensor accepted, at::Tensor output, at::Tensor gamma,
              at::Tensor keys, at::Tensor updates, bool compact, int64_t cores, bool fusedPrefix) {
  for (const auto& tensor : {query, key, value, output}) check(tensor, at::kBFloat16, 3);
  check(beta, at::kBFloat16, 2);
  check(gate, at::kFloat, 2);
  check(state, at::kBFloat16, 4);
  for (const auto& tensor : {sequence, stateIds, accepted}) check(tensor, at::kInt, 1);
  for (const auto& tensor : {gamma, keys, updates}) check(tensor, at::kFloat, 4);
  const int64_t batch = accepted.numel(), tokens = query.size(0), keyHeads = query.size(1), heads = value.size(1);
  TORCH_CHECK(cores >= 1 && cores <= 40 && batch > 0 && batch <= 100, "invalid A2 launch extent");
  TORCH_CHECK(keyHeads == 8 && heads == 24, "prototype qualifies only the actual TP2 8/24 head layout");
  for (const auto& tensor :
       {key, value, beta, gate, state, sequence, stateIds, accepted, output, gamma, keys, updates}) {
    TORCH_CHECK(tensor.device() == query.device(), "all emit buffers must reside on one NPU");
  }
  TORCH_CHECK(query.sizes() == key.sizes() && query.size(2) == 128 && value.size(0) == tokens && value.size(2) == 128 &&
                  heads % keyHeads == 0 && output.sizes() == value.sizes(),
              "invalid Q/K/V shape");
  TORCH_CHECK(tokens == batch * 4 && sequence.numel() == batch + 1 && stateIds.numel() == tokens,
              "only fixed-capacity Q4 metadata is supported");
  TORCH_CHECK(beta.size(0) == tokens && beta.size(1) == heads && gate.sizes() == beta.sizes(), "invalid gating shape");
  TORCH_CHECK(state.size(0) == 1 + batch * (compact ? 1 : 4) && state.size(1) == heads && state.size(2) == 128 &&
                  state.size(3) == 128,
              "invalid dense/base state layout");
  TORCH_CHECK(gamma.size(0) == batch + 1 && gamma.size(1) == 4 && gamma.size(2) == heads && gamma.size(3) == 8,
              "decay log must contain padded eight-float records");
  TORCH_CHECK(keys.size(0) == batch + 1 && keys.size(1) == 4 && keys.size(2) == heads && keys.size(3) == 128 &&
                  updates.sizes() == keys.sizes(),
              "invalid key/update log layout");
  auto stream = c10_npu::getCurrentNPUStream().stream();
  at_npu::native::OpCommand command;
  command.Name("GdnCompactEmitPrototype");
  command.SetCustomHandler([=]() -> int {
    compact_emit_launch(cores, stream, query.data_ptr(), key.data_ptr(), value.data_ptr(), beta.data_ptr(),
                        gate.data_ptr(), state.data_ptr(), sequence.data_ptr(), stateIds.data_ptr(),
                        accepted.data_ptr(), output.data_ptr(), gamma.data_ptr(), keys.data_ptr(), updates.data_ptr(),
                        gamma.data_ptr(), keys.data_ptr(), updates.data_ptr(), batch, tokens, keyHeads, heads, compact,
                        fusedPrefix);
    return 0;
  });
  command.Run();
}

void emit(at::Tensor query, at::Tensor key, at::Tensor value, at::Tensor beta, at::Tensor gate, at::Tensor state,
          at::Tensor sequence, at::Tensor stateIds, at::Tensor accepted, at::Tensor output, at::Tensor gamma,
          at::Tensor keys, at::Tensor updates, bool compact, int64_t cores) {
  run_emit(query, key, value, beta, gate, state, sequence, stateIds, accepted, output, gamma, keys, updates, compact,
           cores, false);
}

void fused_emit(at::Tensor query, at::Tensor key, at::Tensor value, at::Tensor beta, at::Tensor gate, at::Tensor base,
                at::Tensor sequence, at::Tensor stateIds, at::Tensor accepted, at::Tensor output, at::Tensor gamma,
                at::Tensor keys, at::Tensor updates, int64_t cores) {
  run_emit(query, key, value, beta, gate, base, sequence, stateIds, accepted, output, gamma, keys, updates, true, cores,
           true);
}

void reconstruct(at::Tensor base, at::Tensor gamma, at::Tensor keys, at::Tensor updates, at::Tensor lengths,
                 at::Tensor accepted, at::Tensor destination, int64_t cores) {
  check(base, at::kBFloat16, 4);
  check(destination, at::kBFloat16, 4);
  for (const auto& tensor : {gamma, keys, updates}) check(tensor, at::kFloat, 4);
  for (const auto& tensor : {lengths, accepted}) check(tensor, at::kInt, 1);
  const int64_t batch = accepted.numel(), heads = base.size(1);
  TORCH_CHECK(cores >= 1 && cores <= 40 && batch > 0 && batch <= 100, "invalid A2 reconstruction extent");
  TORCH_CHECK(heads == 24, "prototype qualifies only 24 local value heads");
  for (const auto& tensor : {gamma, keys, updates, lengths, accepted, destination}) {
    TORCH_CHECK(tensor.device() == base.device(), "all reconstruction buffers must reside on one NPU");
  }
  TORCH_CHECK(base.sizes() == destination.sizes() && base.size(0) == batch + 1 && base.size(2) == 128 &&
                  base.size(3) == 128 && lengths.numel() == batch,
              "invalid base or request metadata shape");
  TORCH_CHECK(gamma.size(0) == batch + 1 && gamma.size(1) == 4 && gamma.size(2) == heads && gamma.size(3) == 8 &&
                  keys.size(0) == batch + 1 && keys.size(1) == 4 && keys.size(2) == heads && keys.size(3) == 128 &&
                  updates.sizes() == keys.sizes(),
              "invalid compact log layout");
  auto stream = c10_npu::getCurrentNPUStream().stream();
  at_npu::native::OpCommand command;
  command.Name("GdnCompactReconstructPrototype");
  command.SetCustomHandler([=]() -> int {
    compact_reconstruct_launch(cores, stream, base.data_ptr(), gamma.data_ptr(), keys.data_ptr(), updates.data_ptr(),
                               lengths.data_ptr(), accepted.data_ptr(), destination.data_ptr(), batch, heads);
    return 0;
  });
  command.Run();
}
}  // namespace

PYBIND11_MODULE(gdn_compact_state, module) {
  namespace py = pybind11;
  module.def("emit", &emit, py::arg("query"), py::arg("key"), py::arg("value"), py::arg("beta"), py::arg("gate"),
             py::arg("state"), py::arg("sequence"), py::arg("state_ids"), py::arg("accepted"), py::arg("output"),
             py::arg("gamma_log"), py::arg("key_log"), py::arg("update_log"), py::arg("compact"),
             py::arg("cores") = 40);
  module.def("reconstruct", &reconstruct, py::arg("base"), py::arg("gamma_log"), py::arg("key_log"),
             py::arg("update_log"), py::arg("lengths"), py::arg("accepted"), py::arg("destination"),
             py::arg("cores") = 40);
  module.def("fused_emit", &fused_emit, py::arg("query"), py::arg("key"), py::arg("value"), py::arg("beta"),
             py::arg("gate"), py::arg("base"), py::arg("sequence"), py::arg("state_ids"), py::arg("accepted"),
             py::arg("output"), py::arg("gamma_log"), py::arg("key_log"), py::arg("update_log"), py::arg("cores") = 40);
}
