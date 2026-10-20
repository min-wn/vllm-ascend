// SPDX-License-Identifier: Apache-2.0
// A2 port of the ReplaySSM ring lifecycle; no speculative state snapshots.
#include "kernel_operator.h"

using namespace AscendC;

namespace {
constexpr uint32_t kHeads = 24, kWidth = 128, kRing = 32, kTile = 64;

class RingState {
 public:
  __aicore__ inline void Init(GM_ADDR state, GM_ADDR ring, GM_ADDR gates, GM_ADDR cursors, GM_ADDR sequence,
                              GM_ADDR ids, GM_ADDR accepted, uint32_t batch, TPipe* pipe) {
    pipe_ = pipe;
    batch_ = batch;
    state_.SetGlobalBuffer((__gm__ float*)state);
    ring_.SetGlobalBuffer((__gm__ float*)ring);
    gates_.SetGlobalBuffer((__gm__ float*)gates);
    cursors_.SetGlobalBuffer((__gm__ int32_t*)cursors);
    sequence_.SetGlobalBuffer((__gm__ int32_t*)sequence);
    ids_.SetGlobalBuffer((__gm__ int32_t*)ids);
    accepted_.SetGlobalBuffer((__gm__ int32_t*)accepted);
    pipe->InitBuffer(flags_, 32 * sizeof(int32_t));
    pipe->InitBuffer(half_, 16 * 256 * sizeof(float));
    pipe->InitBuffer(log_, 4224 * sizeof(float));
    pipe->InitBuffer(stateBuffer_, kTile * kWidth * sizeof(float));
    pipe->InitBuffer(outer_, kTile * kWidth * sizeof(float));
  }

  __aicore__ inline void Process(uint32_t phase) {
    int32_t begin = sequence_.GetValue(0);
    uint32_t cursor = 0;
    const uint32_t tasks = phase == 0 || phase == 3 ? batch_ : batch_ * kHeads;
    for (uint32_t task = GetBlockIdx(); task < tasks; task += GetBlockNum()) {
      const uint32_t row = phase == 0 || phase == 3 ? task : task / kHeads;
      for (; cursor < row; ++cursor) begin += sequence_.GetValue(cursor + 1);
      if (sequence_.GetValue(row + 1) != 4 || begin < 0 || begin + 4 > batch_ * 4) continue;
      const int32_t selected = ids_.GetValue(begin);
      if (selected <= 0) continue;
      if (phase == 0)
        Advance(selected, accepted_.GetValue(row));
      else if (phase == 3)
        Reset(selected);
      else if (phase == 2 || cursors_.GetValue(selected * 32 + 16) != 0)
        Materialize(selected, task % kHeads);
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

  __aicore__ inline void StoreCursors(uint32_t selected, int32_t wp, int32_t base, int32_t flush, int32_t pending) {
    Sync<HardEvent::MTE3_V>();
    auto local = flags_.Get<int32_t>();
    Duplicate(local, wp, 8);
    Duplicate(local[8], base, 8);
    Duplicate(local[16], flush, 8);
    Duplicate(local[24], pending, 8);
    Sync<HardEvent::V_MTE3>();
    DataCopy(cursors_[selected * 32], local, 32);
  }

  __aicore__ inline void Advance(uint32_t selected, int32_t accepted) {
    if (accepted < 1 || accepted > 4) return;
    int32_t wp = cursors_.GetValue(selected * 32), base = cursors_.GetValue(selected * 32 + 8);
    const bool flush = cursors_.GetValue(selected * 32 + 16) != 0;
    const bool pending = cursors_.GetValue(selected * 32 + 24) != 0;
    if (pending) {
      if (flush) base = (base + wp) & 31;
      wp = flush ? accepted : wp + accepted;
    }
    StoreCursors(selected, wp, base, wp + 8 > 20, 1);
  }

  __aicore__ inline void Reset(uint32_t selected) { StoreCursors(selected, 0, 0, 0, 0); }

  __aicore__ inline void Materialize(uint32_t selected, uint32_t head) {
    const uint32_t count = cursors_.GetValue(selected * 32), base = cursors_.GetValue(selected * 32 + 8);
    if (count == 0 || count > 16) return;
    const uint64_t task = static_cast<uint64_t>(selected) * kHeads + head;
    auto logs = log_.Get<float>();
    auto halves = half_.Get<float>();
    Sync<HardEvent::MTE3_MTE2>();
    Sync<HardEvent::V_MTE2>();
    for (uint32_t token = 0; token < count; ++token) {
      const uint32_t physical = (base + token) & 31;
      DataCopy(halves[token * 256], ring_[(task * 32 + physical) * 256], 256);
      DataCopy(logs[4096 + token * 8], gates_[(task * 32 + physical) * 8], 8);
    }
    Sync<HardEvent::MTE2_V>();
    for (uint32_t token = 0; token < count; ++token) {
      DataCopy(logs[token * 128], halves[token * 256], 128);
      DataCopy(logs[2048 + token * 128], halves[token * 256 + 128], 128);
    }
    Exp(logs[4096], logs[4096], count * 8);
    Sync<HardEvent::V_S>();
    for (uint32_t start = 0; start < kWidth; start += kTile) {
      Sync<HardEvent::MTE3_MTE2>();
      Sync<HardEvent::V_MTE2>();
      auto state = stateBuffer_.Get<float>();
      auto outer = outer_.Get<float>();
      const uint64_t offset = task * kWidth * kWidth + start * kWidth;
      DataCopy(state, state_[offset], kTile * kWidth);
      Sync<HardEvent::MTE2_V>();
      for (uint32_t token = 0; token < count; ++token) {
        Muls(state, state, logs.GetValue(4096 + token * 8), kTile * kWidth);
        PipeBarrier<PIPE_V>();
        uint32_t shape[2] = {kTile, kWidth}, vector[2] = {kTile, 1};
        Broadcast<float, 2, 1>(outer, logs[2048 + token * kWidth + start], shape, vector);
        PipeBarrier<PIPE_V>();
        for (uint32_t column = 0; column < kWidth; column += 64)
          MulAddDst(state[column], outer[column], logs[token * kWidth + column], 64, kTile, {1, 1, 1, 16, 16, 0});
        PipeBarrier<PIPE_V>();
      }
      Sync<HardEvent::V_MTE3>();
      DataCopy(state_[offset], state, kTile * kWidth);
    }
  }

  TPipe* pipe_;
  uint32_t batch_;
  GlobalTensor<float> state_, gates_;
  GlobalTensor<float> ring_;
  GlobalTensor<int32_t> cursors_, sequence_, ids_, accepted_;
  TBuf<TPosition::VECCALC> flags_, half_, log_, stateBuffer_, outer_;
};
}  // namespace

__global__ __aicore__ void replay_state(GM_ADDR state, GM_ADDR ring, GM_ADDR gates, GM_ADDR cursors, GM_ADDR sequence,
                                        GM_ADDR ids, GM_ADDR accepted, uint32_t batch, uint32_t phase) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  TPipe pipe;
  RingState op;
  op.Init(state, ring, gates, cursors, sequence, ids, accepted, batch, &pipe);
  op.Process(phase);
}

extern "C" void replay_state_launch(void* stream, void* state, void* ring, void* gates, void* cursors, void* sequence,
                                    void* ids, void* accepted, uint32_t batch, uint32_t phase) {
  replay_state<<<40, nullptr, stream>>>(state, ring, gates, cursors, sequence, ids, accepted, batch, phase);
}
