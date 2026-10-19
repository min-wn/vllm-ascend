
#include <cstdint>
#include <cassert>
#include <vector>
#include <cstdio>
#define __aicore__
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
std::vector<uint32_t> chunks(uint32_t length, bool wide, bool sink) {
    struct { uint32_t headDim = 256, headDimRope = 0; } constInfo;
    struct { bool isSinkTensor; } info{sink};
    bool THESIS_MM2_WIDE_V = wide;
    const uint32_t K_SPLIT_SIZE = 64, K_BASE = 64;
    Axis k(length);
    // Preserve the safe L0B bound and reduction order; only batch L1 transfers.
    const uint32_t kL1Split = THESIS_MM2_WIDE_V && constInfo.headDim == 256U &&
                             constInfo.headDimRope == 0U && !info.isSinkTensor
        ? 128U : K_SPLIT_SIZE;
    auto KL1Slices = k.Split(kL1Split);
    std::vector<uint32_t> result;
    for (auto &kL1 : KL1Slices) {
        const auto rows = kL1.AlignedSize();
        assert(rows * 256 <= (wide ? 32768U : 16384U));
        assert(rows * 64 <= 24576U);
        for (auto &kL0 : kL1.Split(K_BASE)) {
            assert(kL0.AlignedSize() * 256 <= 16384U);
            assert(kL0.start + kL0.sizeAct <= kL1.sizeAct);
            const bool last = ((k.sizeAct - kL1.start - kL0.start) <= K_BASE);
            const bool initialize = (kL1.start == 0) && (kL0.start == 0);
            result.push_back(kL1.start + kL0.start);
            result.push_back(kL0.sizeAct);
            result.push_back(last);
            result.push_back(initialize);
        }
    }
    return result;
}
int main() {
    for (uint32_t length = 0; length <= 512; ++length) {
        for (bool sink : {false, true}) {
            auto old = chunks(length, false, sink);
            auto newer = chunks(length, true, sink);
            assert(old == newer);
            assert(old.size() == 4 * ((length + 63) / 64));
            uint32_t covered = 0;
            for (uint32_t i = 0; i < old.size(); i += 4) {
                assert(old[i] == covered);
                covered += old[i + 1];
                assert(old[i + 2] == (covered == length));
                assert(old[i + 3] == (i == 0));
            }
            assert(covered == length);
        }
    }
    static_assert(2 * 16384 == 1 * 32768);
    static_assert((2 * 49152 + 32768 + 3 * 24576 + 2 * 16384) * 2 == 475136);
    std::puts("1026 old/new exact compute-order and tail contracts passed");
}
