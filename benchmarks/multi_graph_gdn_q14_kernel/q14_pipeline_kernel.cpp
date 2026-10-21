// SPDX-License-Identifier: Apache-2.0
// Isolated A2 mixed-core products/solve pipeline and transport qualification.
#define TORCH_MODE 1
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "q14_products_tiling.h"
#include "q14_tile_solve.h"

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
    const int32_t current = sequence_.GetValue(row + 1);
    if ((current != 1 && current != kTokens) || accepted < 1 || accepted > kTokens || begin_ < 0 ||
        begin_ + current > batch_ * kTokens)
      return false;
    const int32_t owner = ids_.GetValue(begin_);
    if (owner <= 0 || owner >= slots_) return false;
    for (uint32_t token = 0; token < current; ++token) {
      if (ids_.GetValue(begin_ + token) != owner) return false;
    }
    begin = static_cast<uint32_t>(begin_);
    selected = static_cast<uint32_t>(owner);
    return true;
  }

  __aicore__ inline uint32_t Width(uint32_t row) { return sequence_.GetValue(row + 1); }

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
        for (uint32_t key = 0; key < kKeysPerGroup; ++key)
          Produce(group * kKeysPerGroup + key, begin, selected, metadata_.Width(group / kGroupsPerRequest));
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
  __aicore__ inline void Produce(uint32_t task, uint32_t begin, uint32_t selected, uint32_t current) {
    const uint32_t keyHead = task % kKeyHeads;
    const uint64_t offset = (static_cast<uint64_t>(begin) * kKeyHeads + keyHead) * kWidth;
    auto right = rightBuffer_.Get<float>();
    Nd2NzParams params{};
    params.ndNum = 1;
    params.nValue = current;
    params.dValue = kWidth;
    params.srcDValue = kKeyHeads * kWidth;
    params.dstNzC0Stride = kPaddedColumns;
    params.dstNzNStride = 1;
    DataCopy(right, key_[offset], params);
    DataCopy(right[kTokens * (32 / sizeof(float))], query_[offset], params);
    Sync<HardEvent::MTE2_MTE1>(pipe_);
    mm_.SetOrgShape(kFoldedRows, kColumns, kWidth);
    // Q1 copies only K0/Q0 into columns 0/4. Cube remains M384 N8 K128;
    // the six unwritten tail columns are invalid and never used by solve.
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

class TransportConsumer {
 public:
  __aicore__ inline void Init(GM_ADDR products, GM_ADDR transported, GM_ADDR sequence, GM_ADDR ids, GM_ADDR accepted,
                              uint32_t batch, uint32_t slots, TPipe* pipe) {
    pipe_ = pipe;
    products_.SetGlobalBuffer((__gm__ float*)products);
    transported_.SetGlobalBuffer((__gm__ float*)transported);
    metadata_.Init(sequence, ids, accepted, batch, slots);
    pipe->InitBuffer(input_, kHalfElements * sizeof(float));
  }

  __aicore__ inline void Process(uint32_t requestStart, uint32_t requestStop, bool streamed,
                                 uint32_t vectorIterations) {
    const uint32_t core = GetBlockIdx() / 2, sub = GetBlockIdx() % 2;
    if (!streamed) CrossCoreWaitFlag(kReadyFlag);
    for (uint32_t group = requestStart * kGroupsPerRequest + core; group < requestStop * kGroupsPerRequest;
         group += GetBlockNum()) {
      if (streamed) CrossCoreWaitFlag(kReadyFlag);
      uint32_t begin, selected;
      const bool active = metadata_.Locate(group / kGroupsPerRequest, begin, selected);
      if (active) {
        Sync<HardEvent::MTE3_MTE2>(pipe_);
        auto local = input_.Get<float>();
        const uint64_t offset = static_cast<uint64_t>(group) * kGroupElements + sub * kHalfElements;
        DataCopy(local, products_[offset], kHalfElements);
        Sync<HardEvent::MTE2_V>(pipe_);
        if (streamed) CrossCoreSetFlag<2, PIPE_MTE2>(kConsumedFlag);
        for (uint32_t iteration = 0; iteration < vectorIterations; ++iteration) {
          Muls(local, local, 1.0f, kHalfElements);
          PipeBarrier<PIPE_V>();
        }
        Sync<HardEvent::V_MTE3>(pipe_);
        DataCopy(transported_[offset], local, kHalfElements);
      } else if (streamed) {
        CrossCoreSetFlag<2, PIPE_MTE2>(kConsumedFlag);
      }
    }
    Sync<HardEvent::MTE3_S>(pipe_);
  }

 private:
  TBuf<TPosition::VECCALC> input_;
  GlobalTensor<float> products_, transported_;
  Metadata metadata_;
  TPipe* pipe_;
};
}  // namespace

__global__ __aicore__ void q14_tile_transport(GM_ADDR state, GM_ADDR query, GM_ADDR key, GM_ADDR sequence, GM_ADDR ids,
                                             GM_ADDR accepted, GM_ADDR products, GM_ADDR transported, uint32_t batch,
                                             uint32_t slots, uint64_t stateStride, bool streamed,
                                             uint32_t vectorIterations, uint32_t requestStart, uint32_t requestStop) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
  TPipe pipe;
  if ASCEND_IS_AIC {
    Producer op;
    op.Init(state, query, key, sequence, ids, accepted, products, batch, slots, stateStride, &pipe);
    op.Process(requestStart, requestStop, streamed);
  }
  if ASCEND_IS_AIV {
    TransportConsumer op;
    op.Init(products, transported, sequence, ids, accepted, batch, slots, &pipe);
    op.Process(requestStart, requestStop, streamed, vectorIterations);
  }
}

extern "C" void q14_tile_transport_launch(uint32_t cores, void* stream, void* state, void* query, void* key,
                                         void* sequence, void* ids, void* accepted, void* products, void* transported,
                                         uint32_t batch, uint32_t slots, uint64_t stateStride, bool streamed,
                                         uint32_t vectorIterations, uint32_t requestStart, uint32_t requestStop) {
  q14_tile_transport<<<cores, nullptr, stream>>>(state, query, key, sequence, ids, accepted, products, transported,
                                                batch, slots, stateStride, streamed, vectorIterations, requestStart,
                                                requestStop);
}

namespace {
__aicore__ inline void ConsumeSolve(GM_ADDR products, GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR beta,
                                    GM_ADDR gate, GM_ADDR state, GM_ADDR sequence, GM_ADDR ids, GM_ADDR accepted,
                                    GM_ADDR output, GM_ADDR logs, GM_ADDR valid, GM_ADDR newLogs, GM_ADDR newValid,
                                    uint32_t batch, uint32_t slots, uint64_t stateStride, uint32_t requestStart,
                                    uint32_t requestStop, bool mixed, bool streamed, TPipe* pipe) {
  Q14Tile::Q14TileSolve solve;
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

__global__ __aicore__ void q14_tile_solve(GM_ADDR state, GM_ADDR queryFloat, GM_ADDR keyFloat, GM_ADDR query,
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

__global__ __aicore__ void q14_tile_consume(GM_ADDR products, GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR beta,
                                           GM_ADDR gate, GM_ADDR state, GM_ADDR sequence, GM_ADDR ids, GM_ADDR accepted,
                                           GM_ADDR output, GM_ADDR logs, GM_ADDR valid, GM_ADDR newLogs,
                                           GM_ADDR newValid, uint32_t batch, uint32_t slots, uint64_t stateStride,
                                           uint32_t requestStart, uint32_t requestStop) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  TPipe pipe;
  ConsumeSolve(products, query, key, value, beta, gate, state, sequence, ids, accepted, output, logs, valid, newLogs,
               newValid, batch, slots, stateStride, requestStart, requestStop, false, false, &pipe);
}

extern "C" void q14_tile_solve_launch(uint32_t cores, void* stream, void* state, void* queryFloat, void* keyFloat,
                                     void* query, void* key, void* value, void* beta, void* gate, void* sequence,
                                     void* ids, void* accepted, void* products, void* output, void* logs, void* valid,
                                     void* newLogs, void* newValid, uint32_t batch, uint32_t slots,
                                     uint64_t stateStride, bool streamed, uint32_t requestStart, uint32_t requestStop) {
  q14_tile_solve<<<cores, nullptr, stream>>>(state, queryFloat, keyFloat, query, key, value, beta, gate, sequence, ids,
                                            accepted, products, output, logs, valid, newLogs, newValid, batch, slots,
                                            stateStride, streamed, requestStart, requestStop);
}

extern "C" void q14_tile_consume_launch(uint32_t cores, void* stream, void* products, void* query, void* key,
                                       void* value, void* beta, void* gate, void* state, void* sequence, void* ids,
                                       void* accepted, void* output, void* logs, void* valid, void* newLogs,
                                       void* newValid, uint32_t batch, uint32_t slots, uint64_t stateStride,
                                       uint32_t requestStart, uint32_t requestStop) {
  q14_tile_consume<<<2 * cores, nullptr, stream>>>(products, query, key, value, beta, gate, state, sequence, ids,
                                                  accepted, output, logs, valid, newLogs, newValid, batch, slots,
                                                  stateStride, requestStart, requestStop);
}
