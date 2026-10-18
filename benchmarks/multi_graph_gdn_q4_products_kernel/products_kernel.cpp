// SPDX-License-Identifier: Apache-2.0
// Direct selected-state Cube products; no gathered H0 or GM Q/K pack is generated.
#define ASCENDC_CUBE_ONLY
#define TORCH_MODE 1
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "q4_products_tiling.h"

using namespace AscendC;
using namespace matmul;

namespace {
constexpr uint32_t kTokens = 4;
constexpr uint32_t kKeyHeads = 8;
constexpr uint32_t kHeads = 24;
constexpr uint32_t kWidth = 128;
constexpr uint32_t kColumns = 8;
constexpr uint32_t kPaddedColumns = 16;
constexpr uint32_t kFoldedRows = (kHeads / kKeyHeads) * kWidth;
using CType = MatmulType<TPosition::GM, CubeFormat::ND, float>;

template <typename InputType>
class DirectProducts {
  using AType = MatmulType<TPosition::GM, CubeFormat::ND, InputType>;
  using BType = MatmulType<TPosition::TSCM, CubeFormat::NZ, InputType, true>;

 public:
  __aicore__ inline void Init(GM_ADDR state, GM_ADDR query, GM_ADDR key, GM_ADDR sequence, GM_ADDR ids,
                              GM_ADDR accepted, GM_ADDR output, uint32_t batch, uint32_t slots, uint64_t stateStride,
                              TPipe* pipe) {
    pipe_ = pipe;
    batch_ = batch;
    slots_ = slots;
    stateStride_ = stateStride;
    state_.SetGlobalBuffer((__gm__ InputType*)state);
    query_.SetGlobalBuffer((__gm__ InputType*)query);
    key_.SetGlobalBuffer((__gm__ InputType*)key);
    sequence_.SetGlobalBuffer((__gm__ int32_t*)sequence);
    ids_.SetGlobalBuffer((__gm__ int32_t*)ids);
    accepted_.SetGlobalBuffer((__gm__ int32_t*)accepted);
    output_.SetGlobalBuffer((__gm__ float*)output);
    if constexpr (sizeof(InputType) == sizeof(float)) {
      FillProductsTilingFloat(tiling_);
    } else {
      FillProductsTiling(tiling_);
    }
    mm_.Init(&tiling_, pipe);
    if constexpr (sizeof(InputType) == sizeof(float)) {
      mm_.SetHF32(false, 0);
    }
    pipe->InitBuffer(rightBuffer_, kPaddedColumns * kWidth * sizeof(InputType));
  }

  __aicore__ inline void Process(uint32_t requestStart, uint32_t requestStop) {
    int32_t begin = sequence_.GetValue(0);
    uint32_t cursor = 0;
    for (uint32_t task = requestStart * kKeyHeads + GetBlockIdx(); task < requestStop * kKeyHeads;
         task += GetBlockNum()) {
      const uint32_t row = task / kKeyHeads, keyHead = task % kKeyHeads;
      for (; cursor < row; ++cursor) begin += sequence_.GetValue(cursor + 1);
      if (sequence_.GetValue(row + 1) != kTokens) continue;
      const int32_t accepted = accepted_.GetValue(row);
      if (begin < 0 || begin + kTokens > batch_ * kTokens || accepted < 1 || accepted > kTokens) continue;
      const int32_t selected = ids_.GetValue(begin + accepted - 1);
      if (selected < 0 || selected >= slots_) continue;
      LoadRight(begin, keyHead);
      const uint32_t firstHead = keyHead * (kHeads / kKeyHeads);
      const uint64_t stateOffset = static_cast<uint64_t>(selected) * stateStride_ + firstHead * kWidth * kWidth;
      const uint64_t outputOffset = static_cast<uint64_t>(row * kHeads + firstHead) * kWidth * kColumns;
      // The three consecutive value heads share K/Q. Fold their V rows into
      // one M384 matmul, avoiding three separate Matmul API invocations.
      mm_.SetOrgShape(kFoldedRows, kColumns, kWidth);
      mm_.SetTail(kFoldedRows, kColumns, kWidth);
      mm_.SetTensorA(state_[stateOffset], false);
      mm_.SetTensorB(rightBuffer_.Get<InputType>(), true);
      mm_.IterateAll(output_[outputOffset]);
      mm_.End();
      // The next task must not overwrite B before its final L1->L0B read.
      Sync<HardEvent::MTE1_MTE2>();
    }
  }

 private:
  template <HardEvent event>
  __aicore__ inline void Sync() {
    auto id = pipe_->FetchEventID(event);
    SetFlag<event>(id);
    WaitFlag<event>(id);
  }

  __aicore__ inline void LoadRight(uint32_t begin, uint32_t head) {
    const uint64_t offset = (static_cast<uint64_t>(begin) * kKeyHeads + head) * kWidth;
    auto right = rightBuffer_.Get<InputType>();
    Nd2NzParams params{};
    params.ndNum = 1;
    params.nValue = kTokens;
    params.dValue = kWidth;
    params.srcDValue = kKeyHeads * kWidth;
    params.dstNzC0Stride = kPaddedColumns;
    params.dstNzNStride = 1;
    DataCopy(right, key_[offset], params);
    // NZ C0 is one 32-byte block: 16 BF16 or 8 FP32 elements.
    DataCopy(right[kTokens * (32 / sizeof(InputType))], query_[offset], params);
    Sync<HardEvent::MTE2_MTE1>();
  }

  TPipe* pipe_;
  uint32_t batch_, slots_;
  uint64_t stateStride_;
  TCubeTiling tiling_{};
  Matmul<AType, BType, CType> mm_;
  TBuf<TPosition::B1> rightBuffer_;
  GlobalTensor<InputType> state_, query_, key_;
  GlobalTensor<int32_t> sequence_, ids_, accepted_;
  GlobalTensor<float> output_;
};
}  // namespace

__global__ __aicore__ void q4_products(GM_ADDR state, GM_ADDR query, GM_ADDR key, GM_ADDR sequence, GM_ADDR ids,
                                       GM_ADDR accepted, GM_ADDR output, uint32_t batch, uint32_t slots,
                                       uint64_t stateStride, bool stateFp32, uint32_t requestStart,
                                       uint32_t requestStop) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
  TPipe pipe;
  if (stateFp32) {
    DirectProducts<float> op;
    op.Init(state, query, key, sequence, ids, accepted, output, batch, slots, stateStride, &pipe);
    op.Process(requestStart, requestStop);
  } else {
    DirectProducts<bfloat16_t> op;
    op.Init(state, query, key, sequence, ids, accepted, output, batch, slots, stateStride, &pipe);
    op.Process(requestStart, requestStop);
  }
}

extern "C" void q4_products_launch(uint32_t cores, void* stream, void* state, void* query, void* key, void* sequence,
                                   void* ids, void* accepted, void* output, uint32_t batch, uint32_t slots,
                                   uint64_t stateStride, bool stateFp32, uint32_t requestStart, uint32_t requestStop) {
  q4_products<<<cores, nullptr, stream>>>(state, query, key, sequence, ids, accepted, output, batch, slots, stateStride,
                                          stateFp32, requestStart, requestStop);
}
