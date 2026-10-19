
#include <cstdint>
#include <cassert>
#include <vector>
#include <cstdio>
#include <type_traits>
#define __aicore__
#define __CCE_AICORE__ 220
#define THESIS_FIA_MM2_DOUBLE_V_LAYOUT 1
#define THESIS_FIA_MM2_DOUBLE_V_WIDE 1
constexpr uint32_t BLOCK_CUBE = 16;
template <typename T> T Align(T v, T a) { return (v + a - 1) / a * a; }
/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file axis.h
 * \brief
 */
#ifndef AXIS_H
#define AXIS_H

#include <type_traits>



struct AxisSlices;

struct Axis {
    uint32_t start;
    uint32_t sizeAct;

    __aicore__ inline Axis(uint32_t sizeAct_) : start(0), sizeAct(sizeAct_) {}
    __aicore__ inline Axis(uint32_t start_, uint32_t sizeAct_) : start(start_), sizeAct(sizeAct_) {}

    __aicore__ inline AxisSlices Split(uint32_t splitSize) const;

    template <uint32_t ALIGN_SIZE=BLOCK_CUBE>
    __aicore__ inline uint32_t AlignedSize() const
    {
        return Align(sizeAct, ALIGN_SIZE);
    }

    __aicore__ inline bool IsTailOf(const Axis& parent) const
    {
        return (start + sizeAct) >= parent.sizeAct;
    }
};

struct AxisSlices {
    struct Sentinel {
        uint32_t end_;
    };
    struct Iterator {
        Axis cur_;
        uint32_t tSizeAct_;
        uint32_t splitSize_;

        template <typename T1, typename T2>
        __aicore__ inline T1 Min(T1 a, T2 b)
        {
            return (a > b) ? (b) : (a);
        }
        __aicore__ inline Iterator(const AxisSlices& slices)
            : cur_(Min(slices.splitSize_, slices.sizeAct_)),
              tSizeAct_(slices.sizeAct_), splitSize_(slices.splitSize_) {}

        __aicore__ inline Axis& operator*()
        {
            return cur_;
        }

        __aicore__ inline Iterator& operator++()
        {
            cur_.start += splitSize_;
            cur_.sizeAct = Min(splitSize_, tSizeAct_ - cur_.start);
            return *this;
        }

        __aicore__ inline bool operator!=(const Sentinel& end) const
        {
            return cur_.start < end.end_;
        }
    };

    uint32_t sizeAct_;
    uint32_t splitSize_;

    __aicore__ inline AxisSlices(uint32_t sizeAct, uint32_t splitSize) : sizeAct_(sizeAct), splitSize_(splitSize) {}

    __aicore__ inline Iterator begin()
    {
        return {*this};
    }

    __aicore__ inline Sentinel end()
    {
        return Sentinel{sizeAct_};
    }

    __aicore__ inline uint32_t size() const
    {
        return (sizeAct_ + (splitSize_ - 1)) / splitSize_;
    }
};

__aicore__ inline AxisSlices Axis::Split(uint32_t splitSize) const
{
    return {sizeAct, splitSize};
}
#endif // AXIS_H
struct bfloat16_t { uint16_t value; };
template <typename A, typename B> using IsSameType = std::is_same<A, B>;
enum class FIA_LAYOUT { TND, BSH, NZ };
template <typename T> struct LocalTensor {
    uint32_t offset = 0;
    LocalTensor operator[](uint32_t i) const { return {offset + i * sizeof(T)}; }
};
struct Buffer {
    uint32_t offset = 0, bytes = 0;
    template <typename T> LocalTensor<T> Get() const { return {offset}; }
};
struct Pipe {
    uint32_t used = 0;
    void InitBuffer(Buffer &b, uint32_t bytes) { b = {used, bytes}; used += bytes; }
};
struct ConstInfo {
    uint32_t headDim = 256, headDimRope = 0, keySinkNumber = 0, gSize = 6;
    uint32_t kvHeadNum = 2, qHeadNum = 12, qSeqSize = 4, headDimAlign = 256;
    uint32_t preLoadNum = 2, kvCacheBlockSize = 128, subBlockNum = 2, s2BaseSize = 512;
    int sinkNumber = 0;
    bool ropeSplitMode = false;
};
template <bool S2 = false, uint32_t Preload = 2> struct Config {
    static constexpr bool S2_BASICSIZE_IS_1024 = S2;
    static constexpr uint32_t PRELOAD_NUM = Preload;
};
template <typename Q = bfloat16_t, typename KV = bfloat16_t, typename C = Config<>,
          bool Paged = true, FIA_LAYOUT Layout = FIA_LAYOUT::TND, FIA_LAYOUT PA = FIA_LAYOUT::BSH>
struct LayoutContract {
    using Q_T = Q; using KV_T = KV; using CFG = C;
    static constexpr bool PAGE_ATTENTION = Paged;
    static constexpr FIA_LAYOUT LAYOUT_T = Layout, PA_LAYOUT = PA;
    static constexpr uint32_t L1_Q_SIZE = CFG::S2_BASICSIZE_IS_1024 ? 128 * 192 : 256 * 192;
    static constexpr uint32_t L1_SINK_V_SIZE = 128 * 128, L1_Q_BUFCNT = 2;
    static constexpr uint32_t L1_V_BUFCNT = CFG::S2_BASICSIZE_IS_1024 ? 4 : 2;
    Buffer L1_Q, L1_V;
    LocalTensor<Q_T> qL1Tensor[L1_Q_BUFCNT];
    LocalTensor<KV_T> vL1Tensor[L1_V_BUFCNT];
#if defined(THESIS_FIA_MM2_DOUBLE_V_LAYOUT) && THESIS_FIA_MM2_DOUBLE_V_LAYOUT && (__CCE_AICORE__ == 220)
    static constexpr bool THESIS_DOUBLE_V_STATIC =
        IsSameType<Q_T, bfloat16_t>::value && IsSameType<KV_T, bfloat16_t>::value &&
        PAGE_ATTENTION && LAYOUT_T == FIA_LAYOUT::TND && PA_LAYOUT == FIA_LAYOUT::BSH &&
        !CFG::S2_BASICSIZE_IS_1024 && CFG::PRELOAD_NUM == 2;
#else
    static constexpr bool THESIS_DOUBLE_V_STATIC = false;
#endif
#if defined(THESIS_FIA_MM2_DOUBLE_V_WIDE) && THESIS_FIA_MM2_DOUBLE_V_WIDE
    static constexpr bool THESIS_DOUBLE_V_WIDE = true;
#else
    static constexpr bool THESIS_DOUBLE_V_WIDE = false;
#endif
    bool thesisDoubleVEnabled = false;
    static constexpr uint32_t THESIS_DOUBLE_SLOT_ELEMENTS = 128 * 256;
    static_assert(2 * THESIS_DOUBLE_SLOT_ELEMENTS + 2 * THESIS_DOUBLE_SLOT_ELEMENTS ==
                  2 * (256 * 192) + 2 * (128 * 128));
void allocate(const ConstInfo &constInfo, Pipe *pipe) {
    // Select the whole layout before allocation; unsupported geometry keeps its old bounds.
    thesisDoubleVEnabled = THESIS_DOUBLE_V_STATIC && constInfo.headDim == 256U &&
        constInfo.headDimRope == 0U && !constInfo.ropeSplitMode && constInfo.keySinkNumber == 0U &&
        constInfo.gSize == 6U && constInfo.kvHeadNum == 2U && constInfo.qHeadNum == 12U &&
        constInfo.qSeqSize == 4U && constInfo.headDimAlign == 256U && constInfo.preLoadNum == 2U &&
        constInfo.kvCacheBlockSize == 128U && constInfo.subBlockNum == 2U && constInfo.sinkNumber == 0 &&
        constInfo.s2BaseSize > 0U && constInfo.s2BaseSize <= 512U;
    const uint32_t qL1Stride = thesisDoubleVEnabled ? THESIS_DOUBLE_SLOT_ELEMENTS : L1_Q_SIZE;
    pipe->InitBuffer(L1_Q, qL1Stride * L1_Q_BUFCNT * sizeof(Q_T));
    for (uint32_t i = 0; i < L1_Q_BUFCNT; ++i) {
        qL1Tensor[i] = L1_Q.Get<Q_T>()[i * qL1Stride];
    }    const uint32_t vL1Stride = thesisDoubleVEnabled ? THESIS_DOUBLE_SLOT_ELEMENTS : L1_SINK_V_SIZE;
    pipe->InitBuffer(L1_V, vL1Stride * L1_V_BUFCNT * sizeof(KV_T));
    for (uint32_t i = 0; i < L1_V_BUFCNT; ++i) {
        vL1Tensor[i] = L1_V.Get<KV_T>()[i * vL1Stride];
    }
    }
};
std::vector<uint32_t> chunks(uint32_t length, bool enabled, bool sink, bool wide) {
    struct { bool isSinkTensor; } info{sink};
    const bool THESIS_DOUBLE_V_WIDE = wide, thesisDoubleVEnabled = enabled;
    const uint32_t K_SPLIT_SIZE = 64, K_BASE = 64;
    Axis k(length);
    // L0B still consumes the same ordered 64-row D256 products.
    const uint32_t kL1Split = THESIS_DOUBLE_V_WIDE && thesisDoubleVEnabled && !info.isSinkTensor
        ? 128U : K_SPLIT_SIZE;
    auto KL1Slices = k.Split(kL1Split);
    std::vector<uint32_t> result;
    for (auto &kL1 : KL1Slices) {
        assert(kL1.AlignedSize() * 256 <= (enabled ? 32768U : 16384U));
        assert(kL1.AlignedSize() * 64 <= 24576U);
        for (auto &kL0 : kL1.Split(K_BASE)) {
            assert(kL0.AlignedSize() * 256 <= 16384U);
            assert(kL0.start + kL0.sizeAct <= kL1.sizeAct);
            result.push_back(kL1.start + kL0.start);
            result.push_back(kL0.sizeAct);
            result.push_back((k.sizeAct - kL1.start - kL0.start) <= K_BASE);
            result.push_back(kL1.start == 0 && kL0.start == 0);
        }
    }
    return result;
}
template <typename L> void checkFallback() {
    L l; Pipe p; l.allocate(ConstInfo{}, &p);
    assert(!l.thesisDoubleVEnabled);
    assert(l.qL1Tensor[1].offset == L::L1_Q_SIZE * sizeof(typename L::Q_T));
    assert(l.vL1Tensor[1].offset == l.L1_V.offset + L::L1_SINK_V_SIZE * sizeof(typename L::KV_T));
}
int main() {
    using L = LayoutContract<>;
    for (uint32_t mutation = 0; mutation < 16; ++mutation) {
        ConstInfo c;
        switch (mutation) {
        case 1: c.headDim = 128; break;
        case 2: c.headDimRope = 64; break;
        case 3: c.ropeSplitMode = true; break;
        case 4: c.keySinkNumber = 1; break;
        case 5: c.gSize = 4; break;
        case 6: c.kvHeadNum = 1; break;
        case 7: c.qHeadNum = 16; break;
        case 8: c.qSeqSize = 5; break;
        case 9: c.headDimAlign = 272; break;
        case 10: c.preLoadNum = 1; break;
        case 11: c.kvCacheBlockSize = 64; break;
        case 12: c.subBlockNum = 1; break;
        case 13: c.sinkNumber = 1; break;
        case 14: c.s2BaseSize = 0; break;
        case 15: c.s2BaseSize = 513; break;
        }
        L l; Pipe p; l.allocate(c, &p);
        const bool enabled = mutation == 0;
        assert(l.thesisDoubleVEnabled == enabled);
        const uint32_t qs = enabled ? 65536U : 98304U, vs = enabled ? 65536U : 32768U;
        assert(l.L1_Q.bytes == qs * 2 && l.L1_V.bytes == vs * 2);
        assert(l.L1_V.offset == l.L1_Q.offset + qs * 2);
        for (uint32_t i = 0; i < 2; ++i) {
            assert(l.qL1Tensor[i].offset == i * qs);
            assert(l.vL1Tensor[i].offset == l.L1_V.offset + i * vs);
        }
        // The unmodified three KP slots and two K-sink slots follow Q/V.
        assert(p.used + (3 * 24576 + 2 * 16384) * 2 == 464 * 1024);
    }
    // Execute the actual compile-time gate with each predicate independently false.
    checkFallback<LayoutContract<float>>();
    checkFallback<LayoutContract<bfloat16_t, float>>();
    checkFallback<LayoutContract<bfloat16_t, bfloat16_t, Config<true>>>();
    checkFallback<LayoutContract<bfloat16_t, bfloat16_t, Config<false, 1>>>();
    checkFallback<LayoutContract<bfloat16_t, bfloat16_t, Config<>, false>>();
    checkFallback<LayoutContract<bfloat16_t, bfloat16_t, Config<>, true, FIA_LAYOUT::BSH>>();
    checkFallback<LayoutContract<bfloat16_t, bfloat16_t, Config<>, true, FIA_LAYOUT::TND, FIA_LAYOUT::NZ>>();
    // Actual Q NZ storage, including padding, both 128-column Load3D products and both slots.
    for (uint32_t actualM = 1; actualM <= 128; ++actualM) {
        const uint32_t alignedM = Align(actualM, 16U);
        L l; Pipe p; l.allocate(ConstInfo{}, &p);
        for (uint32_t slot = 0; slot < 2; ++slot) {
            const uint32_t begin = l.qL1Tensor[slot].offset;
            for (uint32_t k = 0; k < 256; ++k) {
                for (uint32_t m = 0; m < alignedM; ++m) {
                    const uint32_t nz = (k / 16) * alignedM * 16 + m * 16 + k % 16;
                    assert(nz < 32768);
                    assert(begin + (nz + 1) * 2 <= begin + 65536);
                }
            }
        }
    }
    for (uint32_t length = 0; length <= 512; ++length) {
        for (bool sink : {false, true}) {
            const auto original = chunks(length, false, sink, false);
            assert(original == chunks(length, true, sink, false));
            assert(original == chunks(length, true, sink, true));
            assert(original == chunks(length, false, sink, true));
            assert(original.size() == 4 * ((length + 63) / 64));
            uint32_t covered = 0;
            for (uint32_t i = 0; i < original.size(); i += 4) {
                assert(original[i] == covered);
                covered += original[i + 1];
                assert(original[i + 2] == (covered == length));
                assert(original[i + 3] == (i == 0));
            }
            assert(covered == length);
        }
    }
    std::puts("16 runtime layouts, 7 static fallbacks, 128 Q NZ tails x 2 slots, 1026 ordered K contracts passed");
}
