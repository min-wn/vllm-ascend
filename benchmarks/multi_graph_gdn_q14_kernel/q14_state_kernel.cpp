// SPDX-License-Identifier: Apache-2.0
// Isolated accepted-base commit and candidate export for compact Q1/Q4 logs.
#include "kernel_operator.h"

using namespace AscendC;

namespace {
constexpr uint32_t kTokens = 4;
constexpr uint32_t kHeads = 24;
constexpr uint32_t kWidth = 128;
constexpr uint32_t kVTile = 64;
constexpr uint32_t kLogElements = 1056;

class Q14State {
 public:
  __aicore__ inline void Init(GM_ADDR state, GM_ADDR sequence, GM_ADDR ids, GM_ADDR accepted, GM_ADDR logs,
                              GM_ADDR valid, GM_ADDR newLogs, GM_ADDR newValid, uint32_t batch, uint32_t slots,
                              uint64_t stateStride, TPipe* pipe) {
    pipe_ = pipe;
    batch_ = batch;
    slots_ = slots;
    stateStride_ = stateStride;
    state_.SetGlobalBuffer((__gm__ float*)state);
    sequence_.SetGlobalBuffer((__gm__ int32_t*)sequence);
    ids_.SetGlobalBuffer((__gm__ int32_t*)ids);
    accepted_.SetGlobalBuffer((__gm__ int32_t*)accepted);
    logs_.SetGlobalBuffer((__gm__ float*)logs);
    valid_.SetGlobalBuffer((__gm__ int32_t*)valid);
    newLogs_.SetGlobalBuffer((__gm__ float*)newLogs);
    newValid_.SetGlobalBuffer((__gm__ int32_t*)newValid);
    pipe->InitBuffer(logBuffer_, kLogElements * sizeof(float));
    pipe->InitBuffer(stateBuffer_, kVTile * kWidth * sizeof(float));
    pipe->InitBuffer(outerBuffer_, kVTile * kWidth * sizeof(float));
    pipe->InitBuffer(bf16Buffer_, kVTile * kWidth * sizeof(float));
    pipe->InitBuffer(scalarBuffer_, 8 * sizeof(int32_t));
    pipe->InitBuffer(stateOutputQueue_, 2, kVTile * kWidth * sizeof(float));
  }

  // Products and solve readers must finish before this phase modifies the base.
  __aicore__ inline void ProcessCommit(uint32_t requestStart, uint32_t requestStop) {
    int32_t begin = sequence_.GetValue(0);
    uint32_t cursor = 0;
    for (uint32_t task = requestStart * kHeads + GetBlockIdx(); task < requestStop * kHeads; task += GetBlockNum()) {
      const uint32_t row = task / kHeads, head = task % kHeads;
      for (; cursor < row; ++cursor) begin += sequence_.GetValue(cursor + 1);
      const int32_t current = sequence_.GetValue(row + 1), accepted = accepted_.GetValue(row);
      if ((current != 1 && current != kTokens) || begin < 0 || begin + current > batch_ * kTokens ||
          accepted < 1 || accepted > kTokens) continue;
      const int32_t selected = ids_.GetValue(begin);
      if (selected <= 0 || selected >= slots_) continue;
      bool sameBase = true;
      for (uint32_t token = 0; token < current; ++token) {
        if (ids_.GetValue(begin + token) != selected) sameBase = false;
      }
      if (!sameBase) continue;
      // A zero staged width closes a terminal debt without publishing a new log.
      const int32_t stagedWidth = newValid_.GetValue((selected * kHeads + head) * 8);
      if (stagedWidth != 0 && stagedWidth != current) continue;
      if (!LoadOldLog(selected, head, accepted)) continue;
      CommitBase(head, selected);
      PublishLog(selected, head);
    }
    Sync<HardEvent::MTE3_S>();
  }

 private:
  template <HardEvent event>
  __aicore__ inline void Sync() {
    const auto id = pipe_->FetchEventID(event);
    SetFlag<event>(id);
    WaitFlag<event>(id);
  }

  __aicore__ inline bool LoadOldLog(uint32_t selected, uint32_t head, int32_t accepted) {
    Sync<HardEvent::MTE3_MTE2>();
    Sync<HardEvent::V_MTE2>();
    const int32_t width = valid_.GetValue((selected * kHeads + head) * 8);
    if (width != 0 && width != 1 && width != kTokens) return false;
    if (width != 0 && accepted > width) return false;
    oldCount_ = width == 0 ? 0 : accepted;
    DataCopy(logBuffer_.Get<float>(), logs_[(static_cast<uint64_t>(selected) * kHeads + head) * kLogElements],
             kLogElements);
    Sync<HardEvent::MTE2_V>();
    return true;
  }

  __aicore__ inline void ApplyOldLog(LocalTensor<float> state, LocalTensor<float> outer, uint32_t vStart,
                                     uint32_t token) {
    auto oldLog = logBuffer_.Get<float>();
    Muls(state, state, oldLog.GetValue(1024 + token * 8), kVTile * kWidth);
    PipeBarrier<PIPE_V>();
    uint32_t outerShape[2] = {kVTile, kWidth}, vectorShape[2] = {kVTile, 1};
    Broadcast<float, 2, 1>(outer, oldLog[512 + token * kWidth + vStart], outerShape, vectorShape);
    PipeBarrier<PIPE_V>();
    for (uint32_t column = 0; column < kWidth; column += 64) {
      MulAddDst(state[column], outer[column], oldLog[token * kWidth + column], 64, kVTile, {1, 1, 1, 16, 16, 0});
    }
    PipeBarrier<PIPE_V>();
  }

  __aicore__ inline void LoadBase(uint32_t selected, uint32_t head, uint32_t vStart) {
    Sync<HardEvent::MTE3_MTE2>();
    Sync<HardEvent::V_MTE2>();
    const uint64_t source = static_cast<uint64_t>(selected) * stateStride_ + head * kWidth * kWidth + vStart * kWidth;
    DataCopy(bf16Buffer_.Get<float>(), state_[source], kVTile * kWidth);
    Sync<HardEvent::MTE2_V>();
    DataCopy(stateBuffer_.Get<float>(), bf16Buffer_.Get<float>(), kVTile * kWidth);
    PipeBarrier<PIPE_V>();
  }

  __aicore__ inline void StoreTile(GlobalTensor<float> target, uint64_t offset) {
    auto candidate = stateOutputQueue_.AllocTensor<float>();
    DataCopy(candidate, stateBuffer_.Get<float>(), kVTile * kWidth);
    stateOutputQueue_.EnQue<float>(candidate);
    auto ready = stateOutputQueue_.DeQue<float>();
    DataCopy(target[offset], ready, kVTile * kWidth);
    stateOutputQueue_.FreeTensor(ready);
  }

  __aicore__ inline void CommitBase(uint32_t head, uint32_t selected) {
    if (oldCount_ == 0) return;
    for (uint32_t vStart = 0; vStart < kWidth; vStart += kVTile) {
      LoadBase(selected, head, vStart);
      for (uint32_t token = 0; token < oldCount_; ++token) {
        ApplyOldLog(stateBuffer_.Get<float>(), outerBuffer_.Get<float>(), vStart, token);
      }
      const uint64_t destination =
          static_cast<uint64_t>(selected) * stateStride_ + head * kWidth * kWidth + vStart * kWidth;
      StoreTile(state_, destination);
    }
  }

  __aicore__ inline void PublishLog(uint32_t selected, uint32_t head) {
    // ApplyOldLog's final vector instructions must finish reading the old
    // log before the new record can reuse its UB storage. Candidate state
    // copy-outs use the independent two-deep VECOUT queue.
    Sync<HardEvent::V_MTE2>();
    const uint64_t offset = (static_cast<uint64_t>(selected) * kHeads + head) * kLogElements;
    auto log = logBuffer_.Get<float>();
    auto flag = scalarBuffer_.Get<int32_t>();
    DataCopy(log, newLogs_[offset], kLogElements);
    DataCopy(flag, newValid_[(selected * kHeads + head) * 8], 8);
    Sync<HardEvent::MTE2_V>();
    Sync<HardEvent::V_MTE3>();
    DataCopy(logs_[offset], log, kLogElements);
    DataCopy(valid_[(selected * kHeads + head) * 8], flag, 8);
    // The next LoadOldLog waits MTE3_MTE2 before reusing log/flag scratch.
  }

 public:
  __aicore__ inline void Export(GM_ADDR destination) {
    GlobalTensor<float> output;
    output.SetGlobalBuffer((__gm__ float*)destination);
    for (uint32_t task = GetBlockIdx(); task < slots_ * kHeads; task += GetBlockNum()) {
      const uint32_t selected = task / kHeads, head = task % kHeads;
      Sync<HardEvent::MTE3_MTE2>();
      Sync<HardEvent::V_MTE2>();
      const int32_t width = valid_.GetValue(task * 8);
      if (width != 0 && width != 1 && width != kTokens) continue;
      DataCopy(logBuffer_.Get<float>(), logs_[static_cast<uint64_t>(task) * kLogElements], kLogElements);
      Sync<HardEvent::MTE2_V>();
      for (uint32_t vStart = 0; vStart < kWidth; vStart += kVTile) {
        LoadBase(selected, head, vStart);
        for (uint32_t token = 0; token < kTokens; ++token) {
          if (token < width) ApplyOldLog(stateBuffer_.Get<float>(), outerBuffer_.Get<float>(), vStart, token);
          const uint64_t offset =
              (static_cast<uint64_t>(selected * kTokens + token) * kHeads + head) * kWidth * kWidth + vStart * kWidth;
          StoreTile(output, offset);
        }
      }
    }
    Sync<HardEvent::MTE3_S>();
  }

 private:
  TPipe* pipe_;
  uint32_t batch_, slots_, oldCount_;
  uint64_t stateStride_;
  GlobalTensor<float> state_, logs_, newLogs_;
  GlobalTensor<int32_t> sequence_, ids_, accepted_, valid_, newValid_;
  TBuf<TPosition::VECCALC> logBuffer_, stateBuffer_, outerBuffer_, bf16Buffer_, scalarBuffer_;
  TQue<QuePosition::VECOUT, 2> stateOutputQueue_;
};
}  // namespace

__global__ __aicore__ void q14_state(GM_ADDR state, GM_ADDR sequence, GM_ADDR ids, GM_ADDR accepted,
                                    GM_ADDR output, GM_ADDR logs, GM_ADDR valid, GM_ADDR newLogs,
                                    GM_ADDR newValid, uint32_t batch, uint32_t slots, uint64_t stateStride,
                                    uint32_t phase, uint32_t requestStart, uint32_t requestStop) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  TPipe pipe;
  Q14State op;
  op.Init(state, sequence, ids, accepted, logs, valid, newLogs, newValid, batch, slots, stateStride, &pipe);
  if (phase == 1)
    op.ProcessCommit(requestStart, requestStop);
  else
    op.Export(output);
}

extern "C" void q14_state_launch(uint32_t cores, void* stream, void* state, void* sequence, void* ids,
                                  void* accepted, void* output, void* logs, void* valid, void* newLogs,
                                  void* newValid, uint32_t batch, uint32_t slots, uint64_t stateStride,
                                  uint32_t phase, uint32_t requestStart, uint32_t requestStop) {
  q14_state<<<cores, nullptr, stream>>>(state, sequence, ids, accepted, output, logs, valid, newLogs, newValid,
                                        batch, slots, stateStride, phase, requestStart, requestStop);
}
