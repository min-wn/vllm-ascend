// SPDX-License-Identifier: Apache-2.0
// Isolated A2 mixed-core products/solve pipeline and transport qualification.
#define TORCH_MODE 1
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "q4_products_tiling.h"
#include "ring_solve.h"

using namespace AscendC;
using namespace matmul;

namespace {
constexpr uint32_t kTokens = 4;
constexpr uint32_t kKeyHeads = 8;
constexpr uint32_t kHeads = 24;
constexpr uint32_t kWidth = 128;
constexpr uint32_t kColumns = 8;
constexpr uint32_t kPaddedColumns = 16;
constexpr uint32_t kFoldedRows = 384;
constexpr uint32_t kTileElements = kFoldedRows * kColumns;
constexpr uint32_t kKeysPerGroup = 2;
constexpr uint32_t kGroupsPerRequest = kKeyHeads / kKeysPerGroup;
constexpr uint32_t kHeadsPerGroup = 3 * kKeysPerGroup;
constexpr uint32_t kGroupElements = kTileElements * kKeysPerGroup;
constexpr uint32_t kHalfElements = kGroupElements / 2;
constexpr uint16_t kReadyFlag = 8;
constexpr uint16_t kConsumedFlag = 9;

template <HardEvent event>
__aicore__ inline void Sync(TPipe* pipe) {
  const auto id = pipe->FetchEventID(event);
  SetFlag<event>(id);
  WaitFlag<event>(id);
}

class Metadata {
 public:
  __aicore__ inline void Init(GM_ADDR sequence, GM_ADDR ids, GM_ADDR accepted, uint32_t batch, uint32_t slots) {
    sequence_.SetGlobalBuffer((__gm__ int32_t*)sequence);
    ids_.SetGlobalBuffer((__gm__ int32_t*)ids);
    accepted_.SetGlobalBuffer((__gm__ int32_t*)accepted);
    batch_ = batch;
    slots_ = slots;
    begin_ = sequence_.GetValue(0);
  }

  __aicore__ inline bool Locate(uint32_t row, uint32_t& begin, uint32_t& selected) {
    for (; cursor_ < row; ++cursor_) begin_ += sequence_.GetValue(cursor_ + 1);
    const int32_t accepted = accepted_.GetValue(row);
    if (sequence_.GetValue(row + 1) != kTokens || accepted < 1 || accepted > kTokens || begin_ < 0 ||
        begin_ + kTokens > batch_ * kTokens)
      return false;
    const int32_t owner = ids_.GetValue(begin_ + accepted - 1);
    if (owner <= 0 || owner >= slots_) return false;
    for (uint32_t token = 0; token < kTokens; ++token) {
      if (ids_.GetValue(begin_ + token) != owner) return false;
    }
    begin = static_cast<uint32_t>(begin_);
    selected = static_cast<uint32_t>(owner);
    return true;
  }

 private:
  GlobalTensor<int32_t> sequence_, ids_, accepted_;
  uint32_t batch_, slots_, cursor_ = 0;
  int32_t begin_;
};

class Producer {
  using AType = MatmulType<TPosition::GM, CubeFormat::ND, float>;
  using BType = MatmulType<TPosition::TSCM, CubeFormat::NZ, float, true>;
  using CType = MatmulType<TPosition::GM, CubeFormat::ND, float>;

 public:
  __aicore__ inline void Init(GM_ADDR state, GM_ADDR query, GM_ADDR key, GM_ADDR sequence, GM_ADDR ids,
                              GM_ADDR accepted, GM_ADDR products, uint32_t batch, uint32_t slots, uint64_t stateStride,
                              TPipe* pipe) {
    pipe_ = pipe;
    stateStride_ = stateStride;
    state_.SetGlobalBuffer((__gm__ float*)state);
    query_.SetGlobalBuffer((__gm__ float*)query);
    key_.SetGlobalBuffer((__gm__ float*)key);
    products_.SetGlobalBuffer((__gm__ float*)products);
    metadata_.Init(sequence, ids, accepted, batch, slots);
    FillProductsTilingFloat(tiling_);
    mm_.Init(&tiling_, pipe);
    mm_.SetHF32(false, 0);
    pipe->InitBuffer(rightBuffer_, kPaddedColumns * kWidth * sizeof(float));
  }

  __aicore__ inline void Process(uint32_t requestStart, uint32_t requestStop, bool streamed) {
    uint32_t outstanding = 0;
    for (uint32_t group = requestStart * kGroupsPerRequest + GetBlockIdx(); group < requestStop * kGroupsPerRequest;
         group += GetBlockNum()) {
      // Two task slots let P(next) overlap S(current). All task products have
      // independent GM addresses; this bounds hardware flags, not buffer reuse.
      if (streamed && outstanding == 2) {
        CrossCoreWaitFlag(kConsumedFlag);
        --outstanding;
      }
      uint32_t begin, selected;
      if (metadata_.Locate(group / kGroupsPerRequest, begin, selected)) {
        for (uint32_t key = 0; key < kKeysPerGroup; ++key) Produce(group * kKeysPerGroup + key, begin, selected);
      }
      // Even null tasks participate, so consumer flags never depend on validity.
      if (streamed) {
        CrossCoreSetFlag<2, PIPE_FIX>(kReadyFlag);
        ++outstanding;
      }
    }
    while (outstanding > 0) {
      CrossCoreWaitFlag(kConsumedFlag);
      --outstanding;
    }
    if (!streamed) CrossCoreSetFlag<2, PIPE_FIX>(kReadyFlag);
  }

 private:
  __aicore__ inline void Produce(uint32_t task, uint32_t begin, uint32_t selected) {
    const uint32_t keyHead = task % kKeyHeads;
    const uint64_t offset = (static_cast<uint64_t>(begin) * kKeyHeads + keyHead) * kWidth;
    auto right = rightBuffer_.Get<float>();
    Nd2NzParams params{};
    params.ndNum = 1;
    params.nValue = kTokens;
    params.dValue = kWidth;
    params.srcDValue = kKeyHeads * kWidth;
    params.dstNzC0Stride = kPaddedColumns;
    params.dstNzNStride = 1;
    DataCopy(right, key_[offset], params);
    DataCopy(right[kTokens * (32 / sizeof(float))], query_[offset], params);
    Sync<HardEvent::MTE2_MTE1>(pipe_);
    mm_.SetOrgShape(kFoldedRows, kColumns, kWidth);
    mm_.SetTail(kFoldedRows, kColumns, kWidth);
    mm_.SetTensorA(state_[static_cast<uint64_t>(selected) * stateStride_ + keyHead * 3 * kWidth * kWidth], false);
    mm_.SetTensorB(right, true);
    mm_.IterateAll(products_[static_cast<uint64_t>(task) * kTileElements]);
    mm_.End();
    Sync<HardEvent::MTE1_MTE2>(pipe_);
  }

  // The Matmul alias selects a service/client in mixed compilation. Use the
  // AIC implementation directly; neither AIV is reserved for Matmul service.
  MatmulImpl<AType, BType, CType> mm_;
  TCubeTiling tiling_{};
  TBuf<TPosition::B1> rightBuffer_;
  GlobalTensor<float> state_, query_, key_, products_;
  Metadata metadata_;
  TPipe* pipe_;
  uint64_t stateStride_;
};

}  // namespace

namespace {
__aicore__ inline void ConsumeSolve(GM_ADDR products, GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR beta,
                                    GM_ADDR gate, GM_ADDR state, GM_ADDR sequence, GM_ADDR ids, GM_ADDR accepted,
                                    GM_ADDR output, GM_ADDR logs, GM_ADDR valid, GM_ADDR newLogs, GM_ADDR newValid,
                                    uint32_t batch, uint32_t slots, uint64_t stateStride, uint32_t requestStart,
                                    uint32_t requestStop, bool mixed, bool streamed, TPipe* pipe) {
  ReplaySSM::RingSolve solve;
  solve.Init(products, query, key, value, beta, gate, state, sequence, ids, accepted, output, logs, valid, newLogs,
             newValid, batch, slots, stateStride, pipe);
  Metadata metadata;
  metadata.Init(sequence, ids, accepted, batch, slots);
  const uint32_t core = GetBlockIdx() / 2, sub = GetBlockIdx() % 2;
  const uint32_t producerCores = mixed ? GetBlockNum() : GetBlockNum() / 2;
  if (mixed && !streamed) CrossCoreWaitFlag(kReadyFlag);
  for (uint32_t group = requestStart * kGroupsPerRequest + core; group < requestStop * kGroupsPerRequest;
       group += producerCores) {
    if (mixed && streamed) CrossCoreWaitFlag(kReadyFlag);
    uint32_t begin, selected;
    if (metadata.Locate(group / kGroupsPerRequest, begin, selected)) {
      // Two key heads provide six value heads: each paired AIV owns three.
      const uint32_t firstHead = (group % kGroupsPerRequest) * kHeadsPerGroup;
      for (uint32_t headOffset = sub; headOffset < kHeadsPerGroup; headOffset += 2) {
        solve.SolveTask(group / kGroupsPerRequest, firstHead + headOffset, static_cast<int32_t>(begin));
      }
    }
    if (mixed && streamed) CrossCoreSetFlag<2, PIPE_MTE2>(kConsumedFlag);
  }
  Sync<HardEvent::MTE3_S>(pipe);
}
}  // namespace

__global__ __aicore__ void q4_tile_solve(GM_ADDR state, GM_ADDR queryFloat, GM_ADDR keyFloat, GM_ADDR query,
                                         GM_ADDR key, GM_ADDR value, GM_ADDR beta, GM_ADDR gate, GM_ADDR sequence,
                                         GM_ADDR ids, GM_ADDR accepted, GM_ADDR products, GM_ADDR output, GM_ADDR logs,
                                         GM_ADDR valid, GM_ADDR newLogs, GM_ADDR newValid, uint32_t batch,
                                         uint32_t slots, uint64_t stateStride, bool streamed, uint32_t requestStart,
                                         uint32_t requestStop) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
  TPipe pipe;
  if ASCEND_IS_AIC {
    Producer op;
    op.Init(state, queryFloat, keyFloat, sequence, ids, accepted, products, batch, slots, stateStride, &pipe);
    op.Process(requestStart, requestStop, streamed);
  }
  if ASCEND_IS_AIV {
    ConsumeSolve(products, query, key, value, beta, gate, state, sequence, ids, accepted, output, logs, valid, newLogs,
                 newValid, batch, slots, stateStride, requestStart, requestStop, true, streamed, &pipe);
  }
}

extern "C" void replay_solve_launch(uint32_t cores, void* stream, void* state, void* queryFloat, void* keyFloat,
                                    void* query, void* key, void* value, void* beta, void* gate, void* sequence,
                                    void* ids, void* accepted, void* products, void* output, void* ring, void* cursors,
                                    void* gates, uint32_t batch, uint32_t slots, uint64_t stateStride, bool streamed) {
  q4_tile_solve<<<cores, nullptr, stream>>>(state, queryFloat, keyFloat, query, key, value, beta, gate, sequence, ids,
                                            accepted, products, output, ring, cursors, gates, cursors, batch, slots,
                                            stateStride, streamed, 0, batch);
}
