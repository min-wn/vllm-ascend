// SPDX-License-Identifier: Apache-2.0
// Benchmark-only implementation of the Q4 block identity. No vendor kernel is copied.
#include "kernel_operator.h"

using namespace AscendC;

namespace {
constexpr uint32_t kTokens = 4;
constexpr uint32_t kKeyHeads = 8;
constexpr uint32_t kHeads = 24;
constexpr uint32_t kWidth = 128;
constexpr uint32_t kVTile = 64;
constexpr float kScale = 0.08838834764831845f;

class Q4Block {
 public:
  __aicore__ inline void Init(GM_ADDR products, GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR beta, GM_ADDR gate,
                              GM_ADDR state, GM_ADDR sequence, GM_ADDR ids, GM_ADDR accepted, GM_ADDR output,
                              uint32_t batch, uint32_t slots, uint64_t stateStride, TPipe* pipe) {
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
    state_.SetGlobalBuffer((__gm__ bfloat16_t*)state);
    sequence_.SetGlobalBuffer((__gm__ int32_t*)sequence);
    ids_.SetGlobalBuffer((__gm__ int32_t*)ids);
    accepted_.SetGlobalBuffer((__gm__ int32_t*)accepted);
    output_.SetGlobalBuffer((__gm__ bfloat16_t*)output);
    pipe->InitBuffer(stateBuffer_, kVTile * kWidth * sizeof(float));
    pipe->InitBuffer(outerBuffer_, kVTile * kWidth * sizeof(float));
    pipe->InitBuffer(bf16Buffer_, kVTile * kWidth * sizeof(bfloat16_t));
    pipe->InitBuffer(stateOutputQueue_, 2, kVTile * kWidth * sizeof(bfloat16_t));
    pipe->InitBuffer(queryBuffer_, kTokens * kWidth * sizeof(float));
    pipe->InitBuffer(keyBuffer_, kTokens * kWidth * sizeof(float));
    pipe->InitBuffer(valueBuffer_, kTokens * kWidth * sizeof(float));
    pipe->InitBuffer(updateBuffer_, kTokens * kWidth * sizeof(float));
    pipe->InitBuffer(outputBuffer_, kTokens * kWidth * sizeof(float));
    pipe->InitBuffer(outputBf16Buffer_, kTokens * kWidth * sizeof(bfloat16_t));
    pipe->InitBuffer(productsBuffer_, kWidth * 8 * sizeof(float));
    pipe->InitBuffer(gramBuffer_, 32 * kWidth * sizeof(float));
    pipe->InitBuffer(scalarBuffer_, 256 * sizeof(float));
    pipe->InitBuffer(temporary_, kWidth * sizeof(float));
    pipe->InitBuffer(indicesBuffer_, kWidth * sizeof(uint32_t));
    auto indices = indicesBuffer_.Get<uint32_t>();
    for (uint32_t row = 0; row < kWidth; ++row) indices.SetValue(row, row * 8 * sizeof(float));
    Sync<HardEvent::S_V>();
  }

  __aicore__ inline void Process() {
    int32_t begin = sequence_.GetValue(0);
    uint32_t cursor = 0;
    for (uint32_t task = GetBlockIdx(); task < batch_ * kHeads; task += GetBlockNum()) {
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
      bool valid = true;
      for (uint32_t token = 0; token < kTokens; ++token) {
        const int32_t target = ids_.GetValue(begin + token);
        if (target < 0 || target >= slots_) valid = false;
      }
      if (!valid) continue;
      LoadInputs(row, head, begin);
      ComputeGram();
      Solve(head);
      WriteOutput(head, begin);
      Materialize(head, begin, selected);
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
    auto work = gramBuffer_.Get<float>();
    auto scalar = scalarBuffer_.Get<float>();
    for (uint32_t current = 0; current < kTokens; ++current) {
      for (uint32_t previous = 0; previous < kTokens; ++previous) {
        const uint32_t index = current * kTokens + previous;
        Mul(work[index * kWidth], key[current * kWidth], key[previous * kWidth], kWidth);
        Mul(work[(16 + index) * kWidth], query[current * kWidth], key[previous * kWidth], kWidth);
      }
    }
    PipeBarrier<PIPE_V>();
    // Explicit FP32 fold/reduction; Cube precision is used only for the supplied H0 products.
    Add(work[64], work, work[64], 64, 32, {1, 1, 1, 16, 16, 16});
    PipeBarrier<PIPE_V>();
    WholeReduceSum(scalar[192], work[64], 64, 32, 1, 1, 16);
    Sync<HardEvent::V_S>();
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
      Gather(temporary, products, indices, token * sizeof(float), kWidth);
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
      Gather(output, products, indices, (4 + token) * sizeof(float), kWidth);
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

  __aicore__ inline void Materialize(uint32_t head, uint32_t begin, uint32_t selected) {
    auto state = stateBuffer_.Get<float>();
    auto outer = outerBuffer_.Get<float>();
    auto bf16 = bf16Buffer_.Get<bfloat16_t>();
    auto key = keyBuffer_.Get<float>();
    auto updates = updateBuffer_.Get<float>();
    for (uint32_t vStart = 0; vStart < kWidth; vStart += kVTile) {
      const uint64_t source = static_cast<uint64_t>(selected) * stateStride_ + head * kWidth * kWidth + vStart * kWidth;
      Sync<HardEvent::V_MTE2>();
      DataCopy(bf16, state_[source], kVTile * kWidth);
      Sync<HardEvent::MTE2_V>();
      Cast(state, bf16, RoundMode::CAST_NONE, kVTile * kWidth);
      PipeBarrier<PIPE_V>();
      // The complete selected tile is resident before any candidate writes.
      // Candidate destinations may alias the selected slot within this request.
      for (uint32_t token = 0; token < kTokens; ++token) {
        Muls(state, state, gamma_[token], kVTile * kWidth);
        PipeBarrier<PIPE_V>();
        uint32_t outerShape[2] = {kVTile, kWidth};
        uint32_t vectorShape[2] = {kVTile, 1};
        Broadcast<float, 2, 1>(outer, updates[token * kWidth + vStart], outerShape, vectorShape);
        PipeBarrier<PIPE_V>();
        for (uint32_t column = 0; column < kWidth; column += 64) {
          MulAddDst(state[column], outer[column], key[token * kWidth + column], 64, kVTile, {1, 1, 1, 16, 16, 0});
        }
        PipeBarrier<PIPE_V>();
        const int32_t target = ids_.GetValue(begin + token);
        if (target != 0) {
          auto candidate = stateOutputQueue_.AllocTensor<bfloat16_t>();
          Cast(candidate, state, RoundMode::CAST_RINT, kVTile * kWidth);
          stateOutputQueue_.EnQue<bfloat16_t>(candidate);
          auto ready = stateOutputQueue_.DeQue<bfloat16_t>();
          const uint64_t destination =
              static_cast<uint64_t>(target) * stateStride_ + head * kWidth * kWidth + vStart * kWidth;
          DataCopy(state_[destination], ready, kVTile * kWidth);
          // The queue owns the MTE3->V reuse dependency. A second candidate
          // buffer permits the next token's arithmetic during this DMA.
          stateOutputQueue_.FreeTensor(ready);
        }
      }
    }
  }

  TPipe* pipe_;
  uint32_t batch_, slots_;
  uint64_t stateStride_;
  float gamma_[kTokens];
  GlobalTensor<float> products_, gate_;
  GlobalTensor<bfloat16_t> query_, key_, value_, beta_, state_, output_;
  GlobalTensor<int32_t> sequence_, ids_, accepted_;
  TBuf<TPosition::VECCALC> stateBuffer_, outerBuffer_, bf16Buffer_, queryBuffer_, keyBuffer_, valueBuffer_;
  TBuf<TPosition::VECCALC> updateBuffer_, outputBuffer_, outputBf16Buffer_, productsBuffer_, gramBuffer_;
  TBuf<TPosition::VECCALC> scalarBuffer_, temporary_, indicesBuffer_;
  TQue<QuePosition::VECOUT, 2> stateOutputQueue_;
};
}  // namespace

__global__ __aicore__ void q4_block(GM_ADDR products, GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR beta,
                                    GM_ADDR gate, GM_ADDR state, GM_ADDR sequence, GM_ADDR ids, GM_ADDR accepted,
                                    GM_ADDR output, uint32_t batch, uint32_t slots, uint64_t stateStride) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  TPipe pipe;
  Q4Block op;
  op.Init(products, query, key, value, beta, gate, state, sequence, ids, accepted, output, batch, slots, stateStride,
          &pipe);
  op.Process();
}

extern "C" void q4_block_launch(uint32_t cores, void* stream, void* products, void* query, void* key, void* value,
                                void* beta, void* gate, void* state, void* sequence, void* ids, void* accepted,
                                void* output, uint32_t batch, uint32_t slots, uint64_t stateStride) {
  q4_block<<<cores, nullptr, stream>>>(products, query, key, value, beta, gate, state, sequence, ids, accepted, output,
                                       batch, slots, stateStride);
}
