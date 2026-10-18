// Benchmark-only Q4 BF16 state/log prototype; no installed operator is replaced.
#define TORCH_MODE 1
#include "kernel_operator.h"
#include "vendor_recurrent.h"

using namespace AscendC;
using namespace RecurrentGatedDeltaRule;

__global__ __aicore__ void compact_emit(GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR beta, GM_ADDR gate,
                                        GM_ADDR state, GM_ADDR sequence, GM_ADDR stateIds, GM_ADDR accepted,
                                        GM_ADDR output, GM_ADDR gammaLog, GM_ADDR keyLog, GM_ADDR updateLog,
                                        GM_ADDR oldGammaLog, GM_ADDR oldKeyLog, GM_ADDR oldUpdateLog, uint32_t batch,
                                        uint32_t tokens, uint32_t keyHeads, uint32_t valueHeads, bool compact,
                                        bool fusedPrefix) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  RecurrentGatedDeltaRuleTilingData tiling{};
  tiling.b = batch;
  tiling.t = tokens;
  tiling.nk = keyHeads;
  tiling.nv = valueHeads;
  tiling.dk = tiling.dv = 128;
  // The matching FLA host tiling chooses 64 at 192 KiB UB for this BF16 profile.
  tiling.vStep = 64;
  tiling.stateOutBufferNum = 2;
  tiling.attnOutBufferNum = 2;
  tiling.scale = 0.08838834764831845f;
  tiling.hasGama = 1;
  tiling.hasGamaK = 0;
  tiling.hasAcceptedTokens = 1;
  tiling.vectorCoreNum = GetBlockNum();
  tiling.stateStride0 = valueHeads * 128 * 128;
  tiling.stateStride1 = 128 * 128;
  tiling.stateStride2 = 128;
  // Reserve the actual working tensors rather than all remaining UB, leaving
  // space for queued log output and the accepted-prefix input records.
  tiling.ubRestBytes = (64 * 2 + 8 * 128 * 3 + 128 * 64 * 2 + 8 * valueHeads * 4) * sizeof(float);
  TPipe pipe;
  RGDR<bfloat16_t, bfloat16_t, bfloat16_t> op(&tiling);
  RGDRInitParams params{query, key, value, gate, nullptr, beta, state, sequence, stateIds, accepted, output, state};
  op.Init(params, &pipe);
  if (compact) {
    op.EnableLogOutput(gammaLog, keyLog, updateLog);
  }
  if (fusedPrefix) {
    op.EnableAcceptedPrefix(oldGammaLog, oldKeyLog, oldUpdateLog);
  }
  op.Process();
}

class Reconstruct {
 public:
  __aicore__ inline void Init(GM_ADDR base, GM_ADDR gamma, GM_ADDR key, GM_ADDR update, GM_ADDR lengths,
                              GM_ADDR accepted, GM_ADDR destination, uint32_t batch, uint32_t heads, TPipe* pipe) {
    batch_ = batch;
    heads_ = heads;
    pipe_ = pipe;
    base_.SetGlobalBuffer((__gm__ bfloat16_t*)base);
    destination_.SetGlobalBuffer((__gm__ bfloat16_t*)destination);
    gamma_.SetGlobalBuffer((__gm__ float*)gamma);
    key_.SetGlobalBuffer((__gm__ float*)key);
    update_.SetGlobalBuffer((__gm__ float*)update);
    lengths_.SetGlobalBuffer((__gm__ int32_t*)lengths);
    accepted_.SetGlobalBuffer((__gm__ int32_t*)accepted);
    pipe->InitBuffer(stateBuffer_, 16 * 128 * sizeof(float));
    pipe->InitBuffer(updateBuffer_, 16 * 128 * sizeof(float));
    pipe->InitBuffer(keyBuffer_, 128 * sizeof(float));
    pipe->InitBuffer(vectorBuffer_, 16 * sizeof(float));
    pipe->InitBuffer(gammaBuffer_, 8 * sizeof(float));
    pipe->InitBuffer(temporary_, 16 * 128 * sizeof(bfloat16_t));
  }

  __aicore__ inline void Process() {
    for (uint32_t task = GetBlockIdx(); task < batch_ * heads_ * 8; task += GetBlockNum()) {
      uint32_t row = task / (heads_ * 8);
      if (lengths_.GetValue(row) == 0) {
        continue;
      }
      uint32_t head = (task / 8) % heads_;
      uint32_t vStart = (task % 8) * 16;
      int32_t accepted = accepted_.GetValue(row);
      if (accepted < 1 || accepted > 4) {
        continue;
      }
      uint64_t stateOffset = ((row + 1) * heads_ + head) * 128 * 128 + vStart * 128;
      auto state = stateBuffer_.Get<float>();
      auto outer = updateBuffer_.Get<float>();
      auto key = keyBuffer_.Get<float>();
      auto update = vectorBuffer_.Get<float>();
      auto gamma = gammaBuffer_.Get<float>();
      auto bf16 = temporary_.Get<bfloat16_t>();
      DataCopy(bf16, base_[stateOffset], 16 * 128);
      Sync<HardEvent::MTE2_V>();
      Cast(state, bf16, RoundMode::CAST_NONE, 16 * 128);
      PipeBarrier<PIPE_V>();
      for (int32_t token = 0; token < accepted; token++) {
        uint64_t index = (((row + 1) * 4 + token) * heads_ + head);
        DataCopy(gamma, gamma_[index * 8], 8);
        DataCopy(key, key_[index * 128], 128);
        DataCopy(update, update_[index * 128 + vStart], 16);
        Sync<HardEvent::MTE2_V>();
        Muls(state, state, gamma.GetValue(0), 16 * 128);
        PipeBarrier<PIPE_V>();
        uint32_t outerShape[2] = {16, 128};
        uint32_t vectorShape[2] = {16, 1};
        Broadcast<float, 2, 1>(outer, update, outerShape, vectorShape);
        PipeBarrier<PIPE_V>();
        for (uint32_t column = 0; column < 128; column += 64) {
          MulAddDst(state[column], outer[column], key[column], 64, 16, {1, 1, 1, 16, 16, 0});
        }
        PipeBarrier<PIPE_V>();
        Sync<HardEvent::V_MTE2>();
      }
      Cast(bf16, state, RoundMode::CAST_RINT, 16 * 128);
      Sync<HardEvent::V_MTE3>();
      DataCopy(destination_[stateOffset], bf16, 16 * 128);
      Sync<HardEvent::MTE3_MTE2>();
    }
  }

 private:
  template <HardEvent event>
  __aicore__ inline void Sync() {
    auto id = pipe_->FetchEventID(event);
    SetFlag<event>(id);
    WaitFlag<event>(id);
  }
  TPipe* pipe_;
  uint32_t batch_, heads_;
  GlobalTensor<bfloat16_t> base_, destination_;
  GlobalTensor<float> gamma_, key_, update_;
  GlobalTensor<int32_t> lengths_, accepted_;
  TBuf<TPosition::VECCALC> stateBuffer_, updateBuffer_, keyBuffer_, vectorBuffer_, gammaBuffer_, temporary_;
};

__global__ __aicore__ void compact_reconstruct(GM_ADDR base, GM_ADDR gamma, GM_ADDR key, GM_ADDR update,
                                               GM_ADDR lengths, GM_ADDR accepted, GM_ADDR destination, uint32_t batch,
                                               uint32_t heads) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  TPipe pipe;
  Reconstruct op;
  op.Init(base, gamma, key, update, lengths, accepted, destination, batch, heads, &pipe);
  op.Process();
}

extern "C" void compact_emit_launch(uint32_t cores, void* stream, void* query, void* key, void* value, void* beta,
                                    void* gate, void* state, void* sequence, void* stateIds, void* accepted,
                                    void* output, void* gammaLog, void* keyLog, void* updateLog, void* oldGammaLog,
                                    void* oldKeyLog, void* oldUpdateLog, uint32_t batch, uint32_t tokens,
                                    uint32_t keyHeads, uint32_t valueHeads, bool compact, bool fusedPrefix) {
  compact_emit<<<cores, nullptr, stream>>>(query, key, value, beta, gate, state, sequence, stateIds, accepted, output,
                                           gammaLog, keyLog, updateLog, oldGammaLog, oldKeyLog, oldUpdateLog, batch,
                                           tokens, keyHeads, valueHeads, compact, fusedPrefix);
}

extern "C" void compact_reconstruct_launch(uint32_t cores, void* stream, void* base, void* gamma, void* key,
                                           void* update, void* lengths, void* accepted, void* destination,
                                           uint32_t batch, uint32_t heads) {
  compact_reconstruct<<<cores, nullptr, stream>>>(base, gamma, key, update, lengths, accepted, destination, batch,
                                                  heads);
}
