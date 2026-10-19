
#include <cstdint>
#include <cassert>
#include <vector>
#include <array>
#include <cstdio>
#include <type_traits>
#include <algorithm>
#define __aicore__
#define __CCE_AICORE__ 220
#define DEBUG_DISABLE_C2 0
#define likely(x) (x)
#define unlikely(x) (x)
constexpr uint32_t BLOCK_CUBE = 16;
template <typename T> T Align(T v, T a) { return (v+a-1)/a*a; }
template <typename T> T Min(T a, T b) { return std::min(a,b); }
struct bfloat16_t { uint16_t v; };
template <typename A, typename B> using IsSameType = std::is_same<A,B>;
enum class FIA_LAYOUT { TND, BSH, NZ };
enum class CubeFormat { ND, NZ };
enum class FormatCategory { GM_KV_BNSD, OTHER };
template <int> struct GmLayoutParams { static constexpr auto CATEGORY = FormatCategory::OTHER; };
enum class HardEvent { MTE1_MTE2, MTE2_MTE1, M_MTE1, MTE1_M, FIX_M, M_FIX };
constexpr int PIPE_M = 0, PIPE_FIX = 1;
struct ConstInfo {
    static constexpr int FIA_SYNC_MODE2 = 2;
    uint32_t headDim=256, headDimRope=0, keySinkNumber=0, gSize=6, kvHeadNum=2, qHeadNum=12;
    uint32_t qSeqSize=4, headDimAlign=256, preLoadNum=2, kvCacheBlockSize=128, subBlockNum=2;
    uint32_t s2BaseSize=512;
    int sinkNumber=0;
    bool ropeSplitMode=false;
    int syncV1C2=0, syncV2C2=1, syncC2V2=2;
};
struct RunInfo {
    uint32_t actMBaseSize=24, actualSingleProcessSInnerSize=512;
    uint32_t actualSingleProcessSInnerSizeAlign=512, loop=0, bIdx=0, n2Idx=0, s2Idx=0;
    bool isSinkTensor=false, isChangeBatch=false;
};
struct MmadParams {
    uint32_t m=0,n=0,k=0,unitFlag=0;
    bool cmatrixInitVal=false,cmatrixSource=false;
};
struct Tensor { uint32_t id=0; };
struct Tile { uint32_t start=0,size=0,m=0,stride=0; };
struct Trace {
    int tokens[6][8]{};
    bool inputReady=false, outputReady=false;
    Tile kp[3],v[2],a[2],b[2];
    std::vector<std::array<uint32_t,6>> math;
    std::vector<std::array<uint32_t,5>> pcopies,vcopies;
    std::vector<char> issue;
    uint32_t crossDone=0,fixes=0;
    Trace() {
        for(int i=2;i<7;++i) tokens[0][i]=1;
        for(int i=3;i<5;++i) { tokens[2][i]=1; tokens[4][i]=1; }
    }
    void balanced() const {
        for(int e=0;e<6;++e) for(int i=0;i<8;++i) {
            const bool initial=(e==0 && i>=2 && i<7) || ((e==2 || e==4) && i>=3 && i<5);
            assert(tokens[e][i]==int(initial));
        }
    }
};
Trace *trace=nullptr;
template <HardEvent E> void WaitFlag(uint32_t id) {
    assert(id<8 && trace->tokens[int(E)][id]==1); --trace->tokens[int(E)][id];
}
template <HardEvent E> void SetFlag(uint32_t id) {
    assert(id<8 && trace->tokens[int(E)][id]==0); ++trace->tokens[int(E)][id];
}
void CrossCoreWaitFlag(int id) {
    if(id==0) { assert(!trace->inputReady); trace->inputReady=true; }
    if(id==1) { assert(trace->inputReady); trace->outputReady=true; }
}
template <int,int> void CrossCoreSetFlag(int id) {
    assert(id==2 && trace->fixes>0 && trace->outputReady); ++trace->crossDone;
}
namespace AscendC {
template <int> void PipeBarrier() { trace->issue.push_back('B'); }
}
void Mmad(Tensor,Tensor a,Tensor b,const MmadParams &p) {
    const auto x=trace->a[a.id],y=trace->b[b.id];
    assert(x.start==y.start && x.size==y.size && x.m==p.m && y.size==p.k);
    trace->math.push_back({x.start,p.k,p.m,p.n,uint32_t(p.cmatrixInitVal),p.unitFlag});
    trace->issue.push_back('M');
}
struct State {
    static constexpr uint32_t D256_HEAD_DIM=256,D256_MM2_QUERY_ROWS=64,L0B_PP_SIZE=16384;
    static constexpr uint32_t L1_KP_BUFCNT=3,L1_V_BUFCNT=2,L0AB_BUFCNT=2,L0C_BUFCNT=2;
    static constexpr uint32_t KP_EVENT0=4,V_EVENT0=2,L0AB_EVENT0=3,L0_READY_EVENT=7,L0C_EVENT0=3;
    static constexpr int KV_FORMAT=0;
    ConstInfo constInfo;
    uint32_t kpL1BufId=0,vL1BufId=0,abL0BufId=0,cL0BufId=0;
    Tensor kpL1Tensor[3]{{0},{1},{2}},vL1Tensor[2]{{0},{1}};
    Tensor aL0Tensor[2]{{0},{1}},bL0Tensor[2]{{0},{1}},cL0Tensor[2]{{0},{1}},mm2ResGm[2]{{0},{1}};
    struct { uint32_t bufCnt=0,firstBufId=0; uint64_t signature=0; } vL1Snapshot;
    void UpdateValue(uint32_t) {}
    template <uint32_t> void ResetLoad3DConfig() {}
    template <CubeFormat F> void CopyPToL1(uint32_t slot,const RunInfo &info,uint32_t stride,
                                         uint32_t ms,uint32_t m,uint32_t k,uint32_t size) {
        assert(trace->inputReady && slot<3 && !trace->tokens[0][KP_EVENT0+slot]);
        assert(size<=64 && Align(m,16U)*Align(size,16U)<=24576);
        assert(k+size<=info.actualSingleProcessSInnerSize);
        const uint32_t expected=F==CubeFormat::NZ ? Align(info.actMBaseSize,16U) : info.actualSingleProcessSInnerSizeAlign;
        assert(stride==expected);
        trace->kp[slot]={k,size,Align(m,16U),stride};
        trace->pcopies.push_back({k,size,ms,m,slot}); trace->issue.push_back('P');
    }
    void CopyVToL1(uint32_t slot,const RunInfo &info,uint32_t k,uint32_t size) {
        assert(trace->inputReady && slot<2 && !trace->tokens[0][V_EVENT0+slot]);
        assert(Align(size,16U)*constInfo.headDim<=16384 && k+size<=info.actualSingleProcessSInnerSize);
        trace->v[slot]={k,size,0,0}; trace->vcopies.push_back({k,size,slot,0,0}); trace->issue.push_back('V');
    }
    template <uint32_t> void LoadAToL0(uint32_t dst,Tensor src,uint32_t m,uint32_t startM,
                                     uint32_t sizeM,uint32_t startK,uint32_t sizeK) {
        assert(!trace->tokens[1][KP_EVENT0+src.id] && !trace->tokens[0][KP_EVENT0+src.id]);
        assert(!trace->tokens[2][L0AB_EVENT0+dst] && startM==0 && startK==0);
        assert(m==trace->kp[src.id].m && sizeM==m && sizeK==Align(trace->kp[src.id].size,16U));
        trace->a[dst]=trace->kp[src.id];
    }
    void LoadBToL0(uint32_t dst,Tensor src,uint32_t k,uint32_t startK,uint32_t sizeK,
                   uint32_t startN,uint32_t sizeN) {
        assert(!trace->tokens[1][V_EVENT0+src.id] && !trace->tokens[0][V_EVENT0+src.id]);
        assert(startK==0 && startN==0 && sizeN==constInfo.headDim);
        assert(k==Align(trace->v[src.id].size,16U) && sizeK==k);
        trace->b[dst]=trace->v[src.id];
    }
    template <CubeFormat F> void FixpipeCToGM(Tensor,uint32_t,uint32_t stride,uint32_t startM,
                                              uint32_t sizeM,uint32_t startN,uint32_t sizeN) {
        assert(trace->outputReady && startN==0 && sizeN==256 && sizeM>0);
        assert(stride==(F==CubeFormat::NZ ? sizeM+startM : 256U));
        ++trace->fixes;
    }
};
template <bool Uni=false,bool Large=false,uint32_t Preload=2> struct Config {
    static constexpr bool ENABLE_UNIFLAG=Uni,S2_BASICSIZE_IS_1024=Large;
    static constexpr uint32_t PRELOAD_NUM=Preload;
};
template <typename Q=bfloat16_t,typename KV=bfloat16_t,bool Paged=true,
          FIA_LAYOUT L=FIA_LAYOUT::TND,FIA_LAYOUT PA=FIA_LAYOUT::BSH> struct Shape {
    using queryType=Q; using kvType=KV;
    static constexpr bool paged=Paged;
    static constexpr FIA_LAYOUT layout=L,pa=PA;
};
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
template <typename FIAT,typename Config> struct Original : State {
    using CFG=Config; using Q_T=typename FIAT::queryType; using KV_T=typename FIAT::kvType;
    static constexpr bool PAGE_ATTENTION=FIAT::paged;
    static constexpr auto LAYOUT_T=FIAT::layout,PA_LAYOUT=FIAT::pa;
#if defined(THESIS_FIA_MM2_PREFETCH) && THESIS_FIA_MM2_PREFETCH && (__CCE_AICORE__ == 220)
    static constexpr bool THESIS_MM2_PREFETCH_STATIC =
        IsSameType<Q_T, bfloat16_t>::value && IsSameType<KV_T, bfloat16_t>::value &&
        PAGE_ATTENTION && LAYOUT_T == FIA_LAYOUT::TND && PA_LAYOUT == FIA_LAYOUT::BSH &&
        !CFG::S2_BASICSIZE_IS_1024 && CFG::PRELOAD_NUM == 2 && !CFG::ENABLE_UNIFLAG;
#else
    static constexpr bool THESIS_MM2_PREFETCH_STATIC = false;
#endif

    template <CubeFormat AFormat>
    __aicore__ inline void PrefetchMm2Block(const RunInfo &info, const Axis &m, const Axis &k,
                                          uint32_t kpBufId, uint32_t vBufId);

    template <CubeFormat O=CubeFormat::ND,CubeFormat A=CubeFormat::ND> void ComputeMm2(const RunInfo &info);
    bool enabled(const RunInfo &info) const {
    const bool prefetchMm2 = THESIS_MM2_PREFETCH_STATIC && constInfo.headDim == 256U &&
        constInfo.headDimRope == 0U && !constInfo.ropeSplitMode && constInfo.keySinkNumber == 0U &&
        constInfo.gSize == 6U && constInfo.kvHeadNum == 2U && constInfo.qHeadNum == 12U &&
        constInfo.qSeqSize == 4U && constInfo.headDimAlign == 256U && constInfo.preLoadNum == 2U &&
        constInfo.kvCacheBlockSize == 128U && constInfo.subBlockNum == 2U && constInfo.sinkNumber == 0 &&
        constInfo.s2BaseSize > 0U && constInfo.s2BaseSize <= 512U && !info.isSinkTensor &&
        info.actMBaseSize > 0U && info.actMBaseSize <= D256_MM2_QUERY_ROWS &&
        info.actualSingleProcessSInnerSize > 0U && info.actualSingleProcessSInnerSize <= 512U;

        return prefetchMm2;
    }
};

template <typename FIAT,typename Config> struct FiaBlockCubeNonQuantGqa : State {
    using CFG=Config; using Q_T=typename FIAT::queryType; using KV_T=typename FIAT::kvType;
    static constexpr bool PAGE_ATTENTION=FIAT::paged;
    static constexpr auto LAYOUT_T=FIAT::layout,PA_LAYOUT=FIAT::pa;
#if defined(THESIS_FIA_MM2_PREFETCH) && THESIS_FIA_MM2_PREFETCH && (__CCE_AICORE__ == 220)
    static constexpr bool THESIS_MM2_PREFETCH_STATIC =
        IsSameType<Q_T, bfloat16_t>::value && IsSameType<KV_T, bfloat16_t>::value &&
        PAGE_ATTENTION && LAYOUT_T == FIA_LAYOUT::TND && PA_LAYOUT == FIA_LAYOUT::BSH &&
        !CFG::S2_BASICSIZE_IS_1024 && CFG::PRELOAD_NUM == 2 && !CFG::ENABLE_UNIFLAG;
#else
    static constexpr bool THESIS_MM2_PREFETCH_STATIC = false;
#endif

    template <CubeFormat AFormat>
    __aicore__ inline void PrefetchMm2Block(const RunInfo &info, const Axis &m, const Axis &k,
                                          uint32_t kpBufId, uint32_t vBufId);

    template <CubeFormat O=CubeFormat::ND,CubeFormat A=CubeFormat::ND> void ComputeMm2(const RunInfo &info);
    bool enabled(const RunInfo &info) const {
    const bool prefetchMm2 = THESIS_MM2_PREFETCH_STATIC && constInfo.headDim == 256U &&
        constInfo.headDimRope == 0U && !constInfo.ropeSplitMode && constInfo.keySinkNumber == 0U &&
        constInfo.gSize == 6U && constInfo.kvHeadNum == 2U && constInfo.qHeadNum == 12U &&
        constInfo.qSeqSize == 4U && constInfo.headDimAlign == 256U && constInfo.preLoadNum == 2U &&
        constInfo.kvCacheBlockSize == 128U && constInfo.subBlockNum == 2U && constInfo.sinkNumber == 0 &&
        constInfo.s2BaseSize > 0U && constInfo.s2BaseSize <= 512U && !info.isSinkTensor &&
        info.actMBaseSize > 0U && info.actMBaseSize <= D256_MM2_QUERY_ROWS &&
        info.actualSingleProcessSInnerSize > 0U && info.actualSingleProcessSInnerSize <= 512U;

        return prefetchMm2;
    }
};
template <typename FIAT, typename Config>
template <CubeFormat OutFormat, CubeFormat AFormat>
__aicore__ inline void Original<FIAT, Config>::ComputeMm2(const RunInfo &info)
{
    if constexpr (GmLayoutParams<KV_FORMAT>::CATEGORY == FormatCategory::GM_KV_BNSD) {
        if (info.isChangeBatch) {
            UpdateValue(info.bIdx);
        }
    }

    CrossCoreWaitFlag(constInfo.syncV1C2);
#if !DEBUG_DISABLE_C2
    constexpr uint32_t M_SPLIT_SIZE = 128;
    const uint32_t mSplit = constInfo.headDim == D256_HEAD_DIM
        ? D256_MM2_QUERY_ROWS : M_SPLIT_SIZE;
    // L0B 半区容量 = L0B_PP_SIZE 元素, V tile 占用 = subK × headDim。
    // 原实现按 headDim=128 取 K 分块 128 (128×128=半区 exact-fit);
    // headDim=192 时 tile ×1.5 (8×3072=24,576 el > 16,384) 越过 L0B ping-pong 半区,
    // buf0 越界静默污染 buf1(错值) / buf1 越界写穿 L0B(0x26)。按容量推导 k 分块。
    const uint32_t kCap = (constInfo.headDim > 0) ? (L0B_PP_SIZE / constInfo.headDim) : (128U * 16U);
    const uint32_t K_SPLIT_SIZE = (kCap < 128U) ? (kCap & ~(BLOCK_CUBE - 1U)) : 128U;
    const uint32_t K_BASE = K_SPLIT_SIZE;

    Axis m(info.actMBaseSize);
    Axis n(constInfo.headDim);
    Axis k(info.actualSingleProcessSInnerSize);
    auto mmResGm = this->mm2ResGm[info.loop % CFG::PRELOAD_NUM];

    ResetLoad3DConfig<M_SPLIT_SIZE>();

    auto KL1Slices = k.Split(K_SPLIT_SIZE);

    bool canFullLoadV = (KL1Slices.size() <= L1_V_BUFCNT);
    uint64_t vCoord = ((uint64_t)info.bIdx << 48) | ((uint64_t)info.n2Idx << 32) | ((uint64_t)info.s2Idx);
    bool reuseVBuf = false;
    vL1Snapshot.bufCnt = 0;
    vL1Snapshot.firstBufId = this->vL1BufId;
    vL1Snapshot.signature = vCoord;

    for (auto &mL1 : m.Split(mSplit)) {
        WaitFlag<HardEvent::FIX_M>(L0C_EVENT0 + this->cL0BufId);

        int32_t kL1Id = -1;
        for (auto &kL1 : KL1Slices) {
            ++kL1Id;
            WaitFlag<HardEvent::MTE1_MTE2>(KP_EVENT0 + this->kpL1BufId);

            CopyPToL1<AFormat>(this->kpL1BufId,
                               info,
                               (AFormat == CubeFormat::NZ)
                                   ? m.AlignedSize()
                                   : info.actualSingleProcessSInnerSizeAlign, /* P为ND时，每行元素个数会按32对齐 */
                               mL1.start,
                               mL1.sizeAct,
                               kL1.start,
                               kL1.sizeAct);

            SetFlag<HardEvent::MTE2_MTE1>(KP_EVENT0 + this->kpL1BufId);
            WaitFlag<HardEvent::MTE2_MTE1>(KP_EVENT0 + this->kpL1BufId);

            if (unlikely(kL1Id >= vL1Snapshot.bufCnt)) {
                reuseVBuf = false;
            }

            uint32_t vBufId;
            if (unlikely(!reuseVBuf)) {
                vBufId = this->vL1BufId;
                WaitFlag<HardEvent::MTE1_MTE2>(V_EVENT0 + vBufId);
                CopyVToL1(vBufId, info, kL1.start, kL1.sizeAct);

                SetFlag<HardEvent::MTE2_MTE1>(V_EVENT0 + vBufId);
                WaitFlag<HardEvent::MTE2_MTE1>(V_EVENT0 + vBufId);
                vL1Snapshot.bufCnt += static_cast<uint32_t>(canFullLoadV); // 不能全载V时，不缓存到快照中
            } else {
                vBufId = (vL1Snapshot.firstBufId + kL1Id) % L1_V_BUFCNT;
            }

            for (auto &kL0 : kL1.Split(K_BASE)) {
                WaitFlag<HardEvent::M_MTE1>(L0AB_EVENT0 + this->abL0BufId);

                LoadAToL0<M_SPLIT_SIZE>(this->abL0BufId,
                                        kpL1Tensor[this->kpL1BufId],
                                        mL1.AlignedSize(),
                                        0,
                                        mL1.AlignedSize(),
                                        kL0.start,
                                        kL0.AlignedSize());
                LoadBToL0(this->abL0BufId,
                          vL1Tensor[vBufId],
                          kL1.AlignedSize(),
                          kL0.start,
                          kL0.AlignedSize(),
                          0,
                          n.AlignedSize());
                SetFlag<HardEvent::MTE1_M>(L0_READY_EVENT);
                WaitFlag<HardEvent::MTE1_M>(L0_READY_EVENT);

                bool isLastK = ((k.sizeAct - kL1.start - kL0.start) <= K_BASE);
                MmadParams mmadParams;
                mmadParams.m = mL1.AlignedSize();
                mmadParams.n = n.sizeAct;
                mmadParams.k = kL0.sizeAct;
                mmadParams.cmatrixInitVal = (kL1.start == 0) && (kL0.start == 0);
                mmadParams.cmatrixSource = false;
                if constexpr (CFG::ENABLE_UNIFLAG) {
                    mmadParams.unitFlag = isLastK ? 3 : 2;
                }
                Mmad(cL0Tensor[this->cL0BufId], aL0Tensor[this->abL0BufId], bL0Tensor[this->abL0BufId], mmadParams);
                if (likely(!isLastK)) {
                    AscendC::PipeBarrier<PIPE_M>();
                }
                SetFlag<HardEvent::M_MTE1>(L0AB_EVENT0 + this->abL0BufId);
                this->abL0BufId = (this->abL0BufId + 1) % L0AB_BUFCNT;
            }
            SetFlag<HardEvent::MTE1_MTE2>(KP_EVENT0 + this->kpL1BufId);
            if (likely(!info.isSinkTensor)) {
                ++this->kpL1BufId;
                if (this->kpL1BufId >= L1_KP_BUFCNT) {
                    this->kpL1BufId = 0;
                }
            }

            if (unlikely(!canFullLoadV || mL1.IsTailOf(m))) {
                // 当可以复用V_L1 buf时，它的生命周期跨越整个mL1的迭代，在mL1迭代完成释放
                SetFlag<HardEvent::MTE1_MTE2>(V_EVENT0 + vBufId);
            }

            if (unlikely(!reuseVBuf)) {
                // this->vL1BufId = (this->vL1BufId + 1) % L1_V_BUFCNT;
                if (likely(!info.isSinkTensor)) {
                    this->vL1BufId = (this->vL1BufId + 1) % L1_V_BUFCNT;
                }
                reuseVBuf = canFullLoadV;
            }
        }
        if constexpr (!CFG::ENABLE_UNIFLAG) {
            SetFlag<HardEvent::M_FIX>(L0C_EVENT0 + this->cL0BufId);
            WaitFlag<HardEvent::M_FIX>(L0C_EVENT0 + this->cL0BufId);
        }
        if (mL1.start == 0) {
            CrossCoreWaitFlag(constInfo.syncV2C2);
        }
        FixpipeCToGM<OutFormat>(mmResGm,
                                this->cL0BufId,
                                (OutFormat == CubeFormat::NZ)
                                    ? m.sizeAct
                                    : constInfo.headDimAlign, /* 输出ND时，mm2ResGm两行之间间隔元素个数按32对齐 */
                                mL1.start,
                                mL1.sizeAct,
                                0,
                                (OutFormat == CubeFormat::NZ) ? n.AlignedSize() : n.sizeAct);
        SetFlag<HardEvent::FIX_M>(L0C_EVENT0 + this->cL0BufId);
        this->cL0BufId = (this->cL0BufId + 1) % L0C_BUFCNT;
    }
#endif
    CrossCoreSetFlag<ConstInfo::FIA_SYNC_MODE2, PIPE_FIX>(constInfo.syncC2V2);
}
template <typename FIAT, typename Config>
template <CubeFormat AFormat>
__aicore__ inline void FiaBlockCubeNonQuantGqa<FIAT, Config>::PrefetchMm2Block(
    const RunInfo &info, const Axis &m, const Axis &k, uint32_t kpBufId, uint32_t vBufId)
{
    // Each slot still needs its original ownership token, including after MM1.
    WaitFlag<HardEvent::MTE1_MTE2>(KP_EVENT0 + kpBufId);
    CopyPToL1<AFormat>(kpBufId, info,
                      AFormat == CubeFormat::NZ ? m.AlignedSize() : info.actualSingleProcessSInnerSizeAlign,
                      m.start, m.sizeAct, k.start, k.sizeAct);
    SetFlag<HardEvent::MTE2_MTE1>(KP_EVENT0 + kpBufId);
    WaitFlag<HardEvent::MTE1_MTE2>(V_EVENT0 + vBufId);
    CopyVToL1(vBufId, info, k.start, k.sizeAct);
    SetFlag<HardEvent::MTE2_MTE1>(V_EVENT0 + vBufId);
}

template <typename FIAT, typename Config>
template <CubeFormat OutFormat, CubeFormat AFormat>
__aicore__ inline void FiaBlockCubeNonQuantGqa<FIAT, Config>::ComputeMm2(const RunInfo &info)
{
    if constexpr (GmLayoutParams<KV_FORMAT>::CATEGORY == FormatCategory::GM_KV_BNSD) {
        if (info.isChangeBatch) {
            UpdateValue(info.bIdx);
        }
    }

    CrossCoreWaitFlag(constInfo.syncV1C2);
#if !DEBUG_DISABLE_C2
    constexpr uint32_t M_SPLIT_SIZE = 128;
    const uint32_t mSplit = constInfo.headDim == D256_HEAD_DIM
        ? D256_MM2_QUERY_ROWS : M_SPLIT_SIZE;
    // L0B 半区容量 = L0B_PP_SIZE 元素, V tile 占用 = subK × headDim。
    // 原实现按 headDim=128 取 K 分块 128 (128×128=半区 exact-fit);
    // headDim=192 时 tile ×1.5 (8×3072=24,576 el > 16,384) 越过 L0B ping-pong 半区,
    // buf0 越界静默污染 buf1(错值) / buf1 越界写穿 L0B(0x26)。按容量推导 k 分块。
    const uint32_t kCap = (constInfo.headDim > 0) ? (L0B_PP_SIZE / constInfo.headDim) : (128U * 16U);
    const uint32_t K_SPLIT_SIZE = (kCap < 128U) ? (kCap & ~(BLOCK_CUBE - 1U)) : 128U;
    const uint32_t K_BASE = K_SPLIT_SIZE;

    Axis m(info.actMBaseSize);
    Axis n(constInfo.headDim);
    Axis k(info.actualSingleProcessSInnerSize);
    auto mmResGm = this->mm2ResGm[info.loop % CFG::PRELOAD_NUM];

    ResetLoad3DConfig<M_SPLIT_SIZE>();

    const bool prefetchMm2 = THESIS_MM2_PREFETCH_STATIC && constInfo.headDim == 256U &&
        constInfo.headDimRope == 0U && !constInfo.ropeSplitMode && constInfo.keySinkNumber == 0U &&
        constInfo.gSize == 6U && constInfo.kvHeadNum == 2U && constInfo.qHeadNum == 12U &&
        constInfo.qSeqSize == 4U && constInfo.headDimAlign == 256U && constInfo.preLoadNum == 2U &&
        constInfo.kvCacheBlockSize == 128U && constInfo.subBlockNum == 2U && constInfo.sinkNumber == 0 &&
        constInfo.s2BaseSize > 0U && constInfo.s2BaseSize <= 512U && !info.isSinkTensor &&
        info.actMBaseSize > 0U && info.actMBaseSize <= D256_MM2_QUERY_ROWS &&
        info.actualSingleProcessSInnerSize > 0U && info.actualSingleProcessSInnerSize <= 512U;

    auto KL1Slices = k.Split(K_SPLIT_SIZE);

    bool canFullLoadV = (KL1Slices.size() <= L1_V_BUFCNT);
    uint64_t vCoord = ((uint64_t)info.bIdx << 48) | ((uint64_t)info.n2Idx << 32) | ((uint64_t)info.s2Idx);
    bool reuseVBuf = false;
    vL1Snapshot.bufCnt = 0;
    vL1Snapshot.firstBufId = this->vL1BufId;
    vL1Snapshot.signature = vCoord;

    for (auto &mL1 : m.Split(mSplit)) {
        WaitFlag<HardEvent::FIX_M>(L0C_EVENT0 + this->cL0BufId);

        if (prefetchMm2) {
            // syncV1C2 above covers the complete P of this task, never the next task.
            PrefetchMm2Block<AFormat>(info, mL1, Axis(0, Min(K_SPLIT_SIZE, k.sizeAct)),
                                     this->kpL1BufId, this->vL1BufId);
        }

        int32_t kL1Id = -1;
        for (auto &kL1 : KL1Slices) {
            ++kL1Id;
            uint32_t vBufId;
            if (prefetchMm2) {
                WaitFlag<HardEvent::MTE2_MTE1>(KP_EVENT0 + this->kpL1BufId);
                WaitFlag<HardEvent::MTE2_MTE1>(V_EVENT0 + this->vL1BufId);
                vBufId = this->vL1BufId;
                reuseVBuf = false;
                vL1Snapshot.bufCnt += static_cast<uint32_t>(canFullLoadV);
            } else {
                WaitFlag<HardEvent::MTE1_MTE2>(KP_EVENT0 + this->kpL1BufId);

                CopyPToL1<AFormat>(this->kpL1BufId,
                                   info,
                                   (AFormat == CubeFormat::NZ)
                                       ? m.AlignedSize()
                                       : info.actualSingleProcessSInnerSizeAlign, /* P为ND时，每行元素个数会按32对齐 */
                                   mL1.start,
                                   mL1.sizeAct,
                                   kL1.start,
                                   kL1.sizeAct);

                SetFlag<HardEvent::MTE2_MTE1>(KP_EVENT0 + this->kpL1BufId);
                WaitFlag<HardEvent::MTE2_MTE1>(KP_EVENT0 + this->kpL1BufId);

                if (unlikely(kL1Id >= vL1Snapshot.bufCnt)) {
                    reuseVBuf = false;
                }

                if (unlikely(!reuseVBuf)) {
                    vBufId = this->vL1BufId;
                    WaitFlag<HardEvent::MTE1_MTE2>(V_EVENT0 + vBufId);
                    CopyVToL1(vBufId, info, kL1.start, kL1.sizeAct);

                    SetFlag<HardEvent::MTE2_MTE1>(V_EVENT0 + vBufId);
                    WaitFlag<HardEvent::MTE2_MTE1>(V_EVENT0 + vBufId);
                    vL1Snapshot.bufCnt += static_cast<uint32_t>(canFullLoadV); // 不能全载V时，不缓存到快照中
                } else {
                    vBufId = (vL1Snapshot.firstBufId + kL1Id) % L1_V_BUFCNT;
                }

            }

            for (auto &kL0 : kL1.Split(K_BASE)) {
                WaitFlag<HardEvent::M_MTE1>(L0AB_EVENT0 + this->abL0BufId);

                LoadAToL0<M_SPLIT_SIZE>(this->abL0BufId,
                                        kpL1Tensor[this->kpL1BufId],
                                        mL1.AlignedSize(),
                                        0,
                                        mL1.AlignedSize(),
                                        kL0.start,
                                        kL0.AlignedSize());
                LoadBToL0(this->abL0BufId,
                          vL1Tensor[vBufId],
                          kL1.AlignedSize(),
                          kL0.start,
                          kL0.AlignedSize(),
                          0,
                          n.AlignedSize());
                SetFlag<HardEvent::MTE1_M>(L0_READY_EVENT);
                WaitFlag<HardEvent::MTE1_M>(L0_READY_EVENT);

                bool isLastK = ((k.sizeAct - kL1.start - kL0.start) <= K_BASE);
                MmadParams mmadParams;
                mmadParams.m = mL1.AlignedSize();
                mmadParams.n = n.sizeAct;
                mmadParams.k = kL0.sizeAct;
                mmadParams.cmatrixInitVal = (kL1.start == 0) && (kL0.start == 0);
                mmadParams.cmatrixSource = false;
                if constexpr (CFG::ENABLE_UNIFLAG) {
                    mmadParams.unitFlag = isLastK ? 3 : 2;
                }
                Mmad(cL0Tensor[this->cL0BufId], aL0Tensor[this->abL0BufId], bL0Tensor[this->abL0BufId], mmadParams);
                if (prefetchMm2 && !isLastK) {
                    const uint32_t nextK = kL1.start + K_SPLIT_SIZE;
                    PrefetchMm2Block<AFormat>(info, mL1, Axis(nextK, Min(K_SPLIT_SIZE, k.sizeAct - nextK)),
                                             (this->kpL1BufId + 1) % L1_KP_BUFCNT,
                                             (this->vL1BufId + 1) % L1_V_BUFCNT);
                }
                if (likely(!isLastK)) {
                    AscendC::PipeBarrier<PIPE_M>();
                }
                SetFlag<HardEvent::M_MTE1>(L0AB_EVENT0 + this->abL0BufId);
                this->abL0BufId = (this->abL0BufId + 1) % L0AB_BUFCNT;
            }
            SetFlag<HardEvent::MTE1_MTE2>(KP_EVENT0 + this->kpL1BufId);
            if (likely(!info.isSinkTensor)) {
                ++this->kpL1BufId;
                if (this->kpL1BufId >= L1_KP_BUFCNT) {
                    this->kpL1BufId = 0;
                }
            }

            if (unlikely(!canFullLoadV || mL1.IsTailOf(m))) {
                // 当可以复用V_L1 buf时，它的生命周期跨越整个mL1的迭代，在mL1迭代完成释放
                SetFlag<HardEvent::MTE1_MTE2>(V_EVENT0 + vBufId);
            }

            if (unlikely(!reuseVBuf)) {
                // this->vL1BufId = (this->vL1BufId + 1) % L1_V_BUFCNT;
                if (likely(!info.isSinkTensor)) {
                    this->vL1BufId = (this->vL1BufId + 1) % L1_V_BUFCNT;
                }
                reuseVBuf = canFullLoadV;
            }
        }
        if constexpr (!CFG::ENABLE_UNIFLAG) {
            SetFlag<HardEvent::M_FIX>(L0C_EVENT0 + this->cL0BufId);
            WaitFlag<HardEvent::M_FIX>(L0C_EVENT0 + this->cL0BufId);
        }
        if (mL1.start == 0) {
            CrossCoreWaitFlag(constInfo.syncV2C2);
        }
        FixpipeCToGM<OutFormat>(mmResGm,
                                this->cL0BufId,
                                (OutFormat == CubeFormat::NZ)
                                    ? m.sizeAct
                                    : constInfo.headDimAlign, /* 输出ND时，mm2ResGm两行之间间隔元素个数按32对齐 */
                                mL1.start,
                                mL1.sizeAct,
                                0,
                                (OutFormat == CubeFormat::NZ) ? n.AlignedSize() : n.sizeAct);
        SetFlag<HardEvent::FIX_M>(L0C_EVENT0 + this->cL0BufId);
        this->cL0BufId = (this->cL0BufId + 1) % L0C_BUFCNT;
    }
#endif
    CrossCoreSetFlag<ConstInfo::FIA_SYNC_MODE2, PIPE_FIX>(constInfo.syncC2V2);
}

template <CubeFormat O,CubeFormat A> void compare(uint32_t length,uint32_t rows,uint32_t kp,
                                                uint32_t v,uint32_t ab,uint32_t c) {
    using C=FiaBlockCubeNonQuantGqa<Shape<>,Config<>>;
    C candidate; Original<Shape<>,Config<>> original;
    candidate.kpL1BufId=original.kpL1BufId=kp; candidate.vL1BufId=original.vL1BufId=v;
    candidate.abL0BufId=original.abL0BufId=ab; candidate.cL0BufId=original.cL0BufId=c;
    Trace tc,to;
    for(uint32_t call=0;call<3;++call) {
        RunInfo info; info.actualSingleProcessSInnerSize=length; info.actMBaseSize=rows;
        info.actualSingleProcessSInnerSizeAlign=Align(length,32U); info.loop=call;
        info.bIdx=call+1; info.n2Idx=call%2; info.s2Idx=call*2;
        // Model a preceding MM1 using every KP slot. MM2 must still acquire each token.
        for(Trace *t : {&tc,&to}) {
            trace=t; trace->inputReady=trace->outputReady=false;
            for(uint32_t s=0;s<3;++s) {
                WaitFlag<HardEvent::MTE1_MTE2>(4+s); SetFlag<HardEvent::MTE1_MTE2>(4+s);
            }
            trace->issue.clear();
        }
        trace=&tc; candidate.template ComputeMm2<O,A>(info); tc.balanced();
        trace=&to; original.template ComputeMm2<O,A>(info); to.balanced();
        assert(tc.math==to.math && tc.pcopies==to.pcopies && tc.vcopies==to.vcopies);
        assert(tc.fixes==to.fixes && tc.crossDone==to.crossDone);
        assert(candidate.kpL1BufId==original.kpL1BufId && candidate.vL1BufId==original.vL1BufId);
        assert(candidate.abL0BufId==original.abL0BufId && candidate.cL0BufId==original.cL0BufId);
        assert(candidate.vL1Snapshot.bufCnt==original.vL1Snapshot.bufCnt);
        assert(candidate.vL1Snapshot.firstBufId==original.vL1Snapshot.firstBufId);
        assert(candidate.vL1Snapshot.signature==original.vL1Snapshot.signature);
        uint32_t lookaheads=0;
        for(size_t i=0;i+3<tc.issue.size();++i) {
            if(tc.issue[i]=='M' && tc.issue[i+1]=='P' && tc.issue[i+2]=='V' && tc.issue[i+3]=='B') ++lookaheads;
        }
        assert(lookaheads==(candidate.enabled(info) ? (length+63)/64-1 : 0));
        tc.math.clear();to.math.clear();tc.pcopies.clear();to.pcopies.clear();tc.vcopies.clear();to.vcopies.clear();
    }
}
template <typename S=Shape<>,typename C=Config<>> void staticFallback() {
    FiaBlockCubeNonQuantGqa<S,C> x; assert(!x.enabled(RunInfo{}));
}
int main() {
    using C=FiaBlockCubeNonQuantGqa<Shape<>,Config<>>;
    for(uint32_t mutation=0;mutation<22;++mutation) {
        C x; RunInfo info;
        switch(mutation) {
        case 1:x.constInfo.headDim=128;break;
        case 2:x.constInfo.headDimRope=64;break;
        case 3:x.constInfo.ropeSplitMode=true;break;
        case 4:x.constInfo.keySinkNumber=1;break;
        case 5:x.constInfo.gSize=4;break;
        case 6:x.constInfo.kvHeadNum=1;break;
        case 7:x.constInfo.qHeadNum=16;break;
        case 8:x.constInfo.qSeqSize=5;break;
        case 9:x.constInfo.headDimAlign=272;break;
        case 10:x.constInfo.preLoadNum=1;break;
        case 11:x.constInfo.kvCacheBlockSize=64;break;
        case 12:x.constInfo.subBlockNum=1;break;
        case 13:x.constInfo.sinkNumber=1;break;
        case 14:x.constInfo.s2BaseSize=0;break;
        case 15:x.constInfo.s2BaseSize=513;break;
        case 16:info.isSinkTensor=true;break;
        case 17:info.actMBaseSize=0;break;
        case 18:info.actMBaseSize=65;break;
        case 19:info.actualSingleProcessSInnerSize=0;break;
        case 20:info.actualSingleProcessSInnerSize=513;break;
        case 21:x.constInfo.s2BaseSize=1;break;
        }
        assert(x.enabled(info)==(THESIS_FIA_MM2_PREFETCH && (mutation==0 || mutation==21)));
    }
    staticFallback<Shape<float>>(); staticFallback<Shape<bfloat16_t,float>>();
    staticFallback<Shape<bfloat16_t,bfloat16_t,false>>();
    staticFallback<Shape<bfloat16_t,bfloat16_t,true,FIA_LAYOUT::BSH>>();
    staticFallback<Shape<bfloat16_t,bfloat16_t,true,FIA_LAYOUT::TND,FIA_LAYOUT::NZ>>();
    staticFallback<Shape<>,Config<true>>(); staticFallback<Shape<>,Config<false,true>>();
    staticFallback<Shape<>,Config<false,false,1>>();
    for(uint32_t n=1;n<=512;++n) for(uint32_t m : {1U,24U,64U})
    for(uint32_t kp=0;kp<3;++kp) for(uint32_t v=0;v<2;++v)
    for(uint32_t ab=0;ab<2;++ab) for(uint32_t c=0;c<2;++c) {
        compare<CubeFormat::ND,CubeFormat::ND>(n,m,kp,v,ab,c);
        compare<CubeFormat::ND,CubeFormat::NZ>(n,m,kp,v,ab,c);
        compare<CubeFormat::NZ,CubeFormat::ND>(n,m,kp,v,ab,c);
        compare<CubeFormat::NZ,CubeFormat::NZ>(n,m,kp,v,ab,c);
    }
    // Execute the original multi-M fallback, not just its predicate.
    compare<CubeFormat::ND,CubeFormat::ND>(512,65,2,1,1,1);
    compare<CubeFormat::ND,CubeFormat::NZ>(513,24,2,1,1,1);
    std::puts("22 runtime gates, 8 static fallbacks, 147456 sequences x 3 calls; events, copies and products match");
}
