// SPDX-License-Identifier: Apache-2.0
// Benchmark-only compact Q4 state; no vendor implementation is copied.
#include "kernel_operator.h"

using namespace AscendC;

namespace {
constexpr uint32_t kTokens = 4;
constexpr uint32_t kKeyHeads = 8;
constexpr uint32_t kHeads = 24;
constexpr uint32_t kWidth = 128;
constexpr uint32_t kVTile = 64;
constexpr uint32_t kLogElements = 1056;
constexpr float kScale = 0.08838834764831845f;

class Q4Log {
 public:
  __aicore__ inline void Init(GM_ADDR products, GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR beta, GM_ADDR gate,
                              GM_ADDR state, GM_ADDR sequence, GM_ADDR ids, GM_ADDR accepted, GM_ADDR output,
                              GM_ADDR logs, GM_ADDR valid, uint32_t batch, uint32_t slots, uint64_t stateStride,
                              TPipe* pipe) {
    logs_.SetGlobalBuffer((__gm__ float*)logs);
    valid_.SetGlobalBuffer((__gm__ int32_t*)valid);
    pipe_ = pipe;
    batch_ = batch;
    slots_ = slots;
    stateStride_ = stateStride;
    products_.SetGlobalBuffer((__gm__ float*)products);
    query_.SetGlobalBuffer((__gm__ bfloat16_t*)query);
    key_.SetGlobalBuffer((__gm__ bfloat16_t*)key);
    value_.SetGlobalBuffer((__gm__ bfloat16_t*)value);
    beta_.SetGlobalBuffer((__gm__ bfloat16_t*)beta);
    gate_.SetGlobalBuffer((__gm__ float*)gate);
    state_.SetGlobalBuffer((__gm__ float*)state);
    sequence_.SetGlobalBuffer((__gm__ int32_t*)sequence);
    ids_.SetGlobalBuffer((__gm__ int32_t*)ids);
    accepted_.SetGlobalBuffer((__gm__ int32_t*)accepted);
    output_.SetGlobalBuffer((__gm__ bfloat16_t*)output);
    pipe->InitBuffer(logBuffer_, kLogElements * sizeof(float));
    pipe->InitBuffer(stateBuffer_, kVTile * kWidth * sizeof(float));
    pipe->InitBuffer(outerBuffer_, kVTile * kWidth * sizeof(float));
    pipe->InitBuffer(bf16Buffer_, kVTile * kWidth * sizeof(float));
    pipe->InitBuffer(stateOutputQueue_, 2, kVTile * kWidth * sizeof(float));
    pipe->InitBuffer(queryBuffer_, kTokens * kWidth * sizeof(float));
    pipe->InitBuffer(keyBuffer_, kTokens * kWidth * sizeof(float));
    pipe->InitBuffer(valueBuffer_, kTokens * kWidth * sizeof(float));
    pipe->InitBuffer(updateBuffer_, kTokens * kWidth * sizeof(float));
    pipe->InitBuffer(outputBuffer_, kTokens * kWidth * sizeof(float));
    pipe->InitBuffer(outputBf16Buffer_, kTokens * kWidth * sizeof(bfloat16_t));
    pipe->InitBuffer(productsBuffer_, kWidth * 8 * sizeof(float));
    pipe->InitBuffer(scalarBuffer_, 256 * sizeof(float));
    pipe->InitBuffer(temporary_, kWidth * sizeof(float));
    pipe->InitBuffer(indicesBuffer_, kWidth * sizeof(uint32_t));
    auto indices = indicesBuffer_.Get<uint32_t>();
    for (uint32_t row = 0; row < kWidth; ++row) indices.SetValue(row, row * 8 * sizeof(float));
    Sync<HardEvent::S_V>();
  }

  __aicore__ inline void Process(uint32_t requestStart, uint32_t requestStop) {
    int32_t begin = sequence_.GetValue(0);
    uint32_t cursor = 0;
    for (uint32_t task = requestStart * kHeads + GetBlockIdx(); task < requestStop * kHeads; task += GetBlockNum()) {
      const uint32_t row = task / kHeads, head = task % kHeads;
      // Task rows increase monotonically. Each core scans the request counts
      // once, rather than repeating a prefix scan for every request/head task.
      for (; cursor < row; ++cursor) begin += sequence_.GetValue(cursor + 1);
      if (sequence_.GetValue(row + 1) == 0) continue;
      if (sequence_.GetValue(row + 1) != kTokens) continue;
      const int32_t accepted = accepted_.GetValue(row);
      if (begin < 0 || begin + kTokens > batch_ * kTokens || accepted < 1 || accepted > kTokens) continue;
      const int32_t selected = ids_.GetValue(begin + accepted - 1);
      if (selected < 0 || selected >= slots_) continue;
      if (selected == 0) continue;
      bool sameBase = true;
      for (uint32_t token = 0; token < kTokens; ++token) {
        if (ids_.GetValue(begin + token) != selected) sameBase = false;
      }
      if (!sameBase) continue;
      LoadInputs(row, head, begin);
      DataCopy(logBuffer_.Get<float>(), logs_[(static_cast<uint64_t>(selected) * kHeads + head) * kLogElements],
               kLogElements);
      Sync<HardEvent::MTE2_V>();
      oldCount_ = valid_.GetValue((selected * kHeads + head) * 8) ? accepted : 0;
      ComputeGram();
      Solve(head);
      WriteOutput(head, begin);
      CommitBase(head, selected);
      WriteLog(head, selected);
    }
  }

 private:
  template <HardEvent event>
  __aicore__ inline void Sync() {
    auto id = pipe_->FetchEventID(event);
    SetFlag<event>(id);
    WaitFlag<event>(id);
  }

  __aicore__ inline void LoadInputs(uint32_t row, uint32_t head, uint32_t begin) {
    // This buffer was the preceding task's candidate copy-out source. A
    // MTE3_V wait alone does not stop MTE2 from overwriting it ahead of V.
    Sync<HardEvent::MTE3_MTE2>();
    auto scratch = bf16Buffer_.Get<bfloat16_t>();
    auto query = queryBuffer_.Get<float>();
    auto key = keyBuffer_.Get<float>();
    auto value = valueBuffer_.Get<float>();
    auto scalar = scalarBuffer_.Get<float>();
    const uint64_t qkOffset = (static_cast<uint64_t>(begin) * kKeyHeads + head / 3) * kWidth;
    const uint64_t vOffset = (static_cast<uint64_t>(begin) * kHeads + head) * kWidth;
    DataCopyExtParams qkParams{kTokens, kWidth * sizeof(bfloat16_t), (kKeyHeads - 1) * kWidth * sizeof(bfloat16_t), 0,
                               0};
    DataCopyExtParams vParams{kTokens, kWidth * sizeof(bfloat16_t), (kHeads - 1) * kWidth * sizeof(bfloat16_t), 0, 0};
    DataCopyPadExtParams<bfloat16_t> noPadding{false, 0, 0, 0};
    DataCopyPad(scratch, query_[qkOffset], qkParams, noPadding);
    DataCopyPad(scratch[kTokens * kWidth], key_[qkOffset], qkParams, noPadding);
    DataCopyPad(scratch[2 * kTokens * kWidth], value_[vOffset], vParams, noPadding);
    DataCopy(scratch[3 * kTokens * kWidth], beta_[begin * kHeads], kTokens * kHeads);
    DataCopy(scalar, gate_[begin * kHeads], kTokens * kHeads);
    DataCopy(productsBuffer_.Get<float>(), products_[static_cast<uint64_t>(row * kHeads + head) * kWidth * 8],
             kWidth * 8);
    Sync<HardEvent::MTE2_V>();
    Cast(query, scratch, RoundMode::CAST_NONE, kTokens * kWidth);
    Cast(key, scratch[kTokens * kWidth], RoundMode::CAST_NONE, kTokens * kWidth);
    Cast(value, scratch[2 * kTokens * kWidth], RoundMode::CAST_NONE, kTokens * kWidth);
    Cast(scalar[kTokens * kHeads], scratch[3 * kTokens * kWidth], RoundMode::CAST_NONE, kTokens * kHeads);
    Exp(scalar, scalar, kTokens * kHeads);
    PipeBarrier<PIPE_V>();
  }

  __aicore__ inline void ComputeGram() {
    auto query = queryBuffer_.Get<float>();
    auto key = keyBuffer_.Get<float>();
    // Gram products finish before materialization uses the outer-product buffer.
    auto work = outerBuffer_.Get<float>();
    auto scalar = scalarBuffer_.Get<float>();
    for (uint32_t current = 0; current < kTokens; ++current) {
      for (uint32_t previous = 0; previous < kTokens; ++previous) {
        const uint32_t index = current * kTokens + previous;
        Mul(work[index * kWidth], key[current * kWidth], key[previous * kWidth], kWidth);
        Mul(work[(16 + index) * kWidth], query[current * kWidth], key[previous * kWidth], kWidth);
      }
    }
    auto oldLog = logBuffer_.Get<float>();
    for (uint32_t previous = 0; previous < kTokens; ++previous) {
      for (uint32_t current = 0; current < kTokens; ++current) {
        const uint32_t index = previous * 8 + current;
        Mul(work[(32 + index) * kWidth], oldLog[previous * kWidth], key[current * kWidth], kWidth);
        Mul(work[(36 + index) * kWidth], oldLog[previous * kWidth], query[current * kWidth], kWidth);
      }
    }
    PipeBarrier<PIPE_V>();
    Add(work[64], work, work[64], 64, 64, {1, 1, 1, 16, 16, 16});
    PipeBarrier<PIPE_V>();
    WholeReduceSum(scalar[192], work[64], 64, 64, 1, 1, 16);
    Sync<HardEvent::V_S>();
  }

  __aicore__ inline void Projection(LocalTensor<float> destination, uint32_t column) {
    auto oldLog = logBuffer_.Get<float>();
    auto temporary = temporary_.Get<float>();
    auto scalar = scalarBuffer_.Get<float>();
    Gather(destination, productsBuffer_.Get<float>(), indicesBuffer_.Get<uint32_t>(), column * sizeof(float), kWidth);
    PipeBarrier<PIPE_V>();
    for (uint32_t previous = 0; previous < oldCount_; ++previous) {
      Muls(destination, destination, oldLog.GetValue(1024 + previous * 8), kWidth);
      Muls(temporary, oldLog[512 + previous * kWidth], scalar.GetValue(224 + previous * 8 + column), kWidth);
      PipeBarrier<PIPE_V>();
      Add(destination, destination, temporary, kWidth);
      PipeBarrier<PIPE_V>();
    }
  }

  __aicore__ inline void Solve(uint32_t head) {
    auto scalar = scalarBuffer_.Get<float>();
    auto updates = updateBuffer_.Get<float>();
    auto outputs = outputBuffer_.Get<float>();
    auto values = valueBuffer_.Get<float>();
    auto temporary = temporary_.Get<float>();
    auto products = productsBuffer_.Get<float>();
    auto indices = indicesBuffer_.Get<uint32_t>();
    float alpha[kTokens], beta[kTokens], decay[kTokens][kTokens];
    for (uint32_t token = 0; token < kTokens; ++token) {
      gamma_[token] = scalar.GetValue(token * kHeads + head);
      beta[token] = scalar.GetValue(kTokens * kHeads + token * kHeads + head);
      alpha[token] = token == 0 ? gamma_[token] : alpha[token - 1] * gamma_[token];
      for (uint32_t previous = 0; previous <= token; ++previous) {
        decay[token][previous] = 1.0f;
        for (uint32_t offset = previous + 1; offset <= token; ++offset) {
          decay[token][previous] *= gamma_[offset];
        }
      }
    }
    Sync<HardEvent::S_V>();
    for (uint32_t token = 0; token < kTokens; ++token) {
      auto update = updates[token * kWidth];
      Projection(update, token);
      DataCopy(temporary, update, kWidth);
      PipeBarrier<PIPE_V>();
      Muls(temporary, temporary, alpha[token], kWidth);
      PipeBarrier<PIPE_V>();
      Sub(update, values[token * kWidth], temporary, kWidth);
      PipeBarrier<PIPE_V>();
      Muls(update, update, beta[token], kWidth);
      PipeBarrier<PIPE_V>();
      for (uint32_t previous = 0; previous < token; ++previous) {
        const float coefficient =
            beta[token] * decay[token][previous] * scalar.GetValue(192 + token * kTokens + previous);
        Muls(temporary, updates[previous * kWidth], coefficient, kWidth);
        PipeBarrier<PIPE_V>();
        Sub(update, update, temporary, kWidth);
        PipeBarrier<PIPE_V>();
      }
      auto output = outputs[token * kWidth];
      Projection(output, 4 + token);
      PipeBarrier<PIPE_V>();
      Muls(output, output, alpha[token], kWidth);
      PipeBarrier<PIPE_V>();
      for (uint32_t previous = 0; previous <= token; ++previous) {
        const float coefficient = decay[token][previous] * scalar.GetValue(192 + 16 + token * kTokens + previous);
        Muls(temporary, updates[previous * kWidth], coefficient, kWidth);
        PipeBarrier<PIPE_V>();
        Add(output, output, temporary, kWidth);
        PipeBarrier<PIPE_V>();
      }
      Muls(output, output, kScale, kWidth);
      PipeBarrier<PIPE_V>();
    }
  }

  __aicore__ inline void WriteOutput(uint32_t head, uint32_t begin) {
    auto output = outputBf16Buffer_.Get<bfloat16_t>();
    Cast(output, outputBuffer_.Get<float>(), RoundMode::CAST_RINT, kTokens * kWidth);
    Sync<HardEvent::V_MTE3>();
    DataCopyExtParams params{kTokens, kWidth * sizeof(bfloat16_t), 0, (kHeads - 1) * kWidth * sizeof(bfloat16_t), 0};
    DataCopyPad(output_[(static_cast<uint64_t>(begin) * kHeads + head) * kWidth], output, params);
    Sync<HardEvent::MTE3_V>();
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

  __aicore__ inline void WriteLog(uint32_t head, uint32_t selected) {
    auto log = logBuffer_.Get<float>();
    DataCopy(log, keyBuffer_.Get<float>(), 512);
    DataCopy(log[512], updateBuffer_.Get<float>(), 512);
    for (uint32_t token = 0; token < kTokens; ++token) {
      for (uint32_t lane = 0; lane < 8; ++lane) log.SetValue(1024 + token * 8 + lane, gamma_[token]);
    }
    Sync<HardEvent::S_V>();
    Sync<HardEvent::V_MTE3>();
    DataCopy(logs_[(static_cast<uint64_t>(selected) * kHeads + head) * kLogElements], log, kLogElements);
    // Every task owns a full DMA/cache-line flag. Scalar writes to neighboring
    // 4-byte flags from different cores can lose updates on A2.
    auto flag = scalarBuffer_.Get<int32_t>();
    Duplicate(flag, int32_t(1), 8);
    Sync<HardEvent::V_MTE3>();
    DataCopy(valid_[(selected * kHeads + head) * 8], flag, 8);
  }

 public:
  __aicore__ inline void Export(GM_ADDR destination) {
    GlobalTensor<float> output;
    output.SetGlobalBuffer((__gm__ float*)destination);
    for (uint32_t task = GetBlockIdx(); task < slots_ * kHeads; task += GetBlockNum()) {
      const uint32_t selected = task / kHeads, head = task % kHeads;
      Sync<HardEvent::V_MTE2>();
      DataCopy(logBuffer_.Get<float>(), logs_[static_cast<uint64_t>(task) * kLogElements], kLogElements);
      Sync<HardEvent::MTE2_V>();
      const bool valid = valid_.GetValue(task * 8) != 0;
      for (uint32_t vStart = 0; vStart < kWidth; vStart += kVTile) {
        LoadBase(selected, head, vStart);
        for (uint32_t token = 0; token < kTokens; ++token) {
          if (valid) ApplyOldLog(stateBuffer_.Get<float>(), outerBuffer_.Get<float>(), vStart, token);
          const uint64_t offset =
              (static_cast<uint64_t>(selected * 4 + token) * kHeads + head) * kWidth * kWidth + vStart * kWidth;
          StoreTile(output, offset);
        }
      }
    }
  }

 private:
  TPipe* pipe_;
  uint32_t batch_, slots_, oldCount_;
  GlobalTensor<float> logs_;
  GlobalTensor<int32_t> valid_;
  TBuf<TPosition::VECCALC> logBuffer_;
  uint64_t stateStride_;
  float gamma_[kTokens];
  GlobalTensor<float> products_, gate_;
  GlobalTensor<bfloat16_t> query_, key_, value_, beta_, output_;
  GlobalTensor<float> state_;
  GlobalTensor<int32_t> sequence_, ids_, accepted_;
  TBuf<TPosition::VECCALC> stateBuffer_, outerBuffer_, bf16Buffer_, queryBuffer_, keyBuffer_, valueBuffer_;
  TBuf<TPosition::VECCALC> updateBuffer_, outputBuffer_, outputBf16Buffer_, productsBuffer_;
  TBuf<TPosition::VECCALC> scalarBuffer_, temporary_, indicesBuffer_;
  TQue<QuePosition::VECOUT, 2> stateOutputQueue_;
};
}  // namespace

__global__ __aicore__ void q4_log(GM_ADDR products, GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR beta,
                                  GM_ADDR gate, GM_ADDR state, GM_ADDR sequence, GM_ADDR ids, GM_ADDR accepted,
                                  GM_ADDR output, GM_ADDR logs, GM_ADDR valid, uint32_t batch, uint32_t slots,
                                  uint64_t stateStride, bool exportOnly, uint32_t requestStart, uint32_t requestStop) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  TPipe pipe;
  Q4Log op;
  op.Init(products, query, key, value, beta, gate, state, sequence, ids, accepted, output, logs, valid, batch, slots,
          stateStride, &pipe);
  if (exportOnly)
    op.Export(output);
  else
    op.Process(requestStart, requestStop);
}

extern "C" void q4_log_launch(uint32_t cores, void* stream, void* products, void* query, void* key, void* value,
                              void* beta, void* gate, void* state, void* sequence, void* ids, void* accepted,
                              void* output, void* logs, void* valid, uint32_t batch, uint32_t slots,
                              uint64_t stateStride, bool exportOnly, uint32_t requestStart, uint32_t requestStop) {
  q4_log<<<cores, nullptr, stream>>>(products, query, key, value, beta, gate, state, sequence, ids, accepted, output,
                                     logs, valid, batch, slots, stateStride, exportOnly, requestStart, requestStop);
}
