
#include <cstdint>
#include <cstdio>
#include <vector>
#include <array>
#include <cassert>
#include <type_traits>
#include <algorithm>
#define __aicore__
#define unlikely(x) (x)
#define DEBUG_MATMUL_GQA 0
constexpr uint32_t BLOCK_CUBE=16;
template<class T> T Align(T x,T a) { return (x+a-1)/a*a; }
template<class T,class U> using IsSameType=std::is_same<T,U>;
struct int4b_t {};
enum class GmFormat { PA_NZ, PA_BnBsND };
enum class L1Format { NZ };
struct Ptr { uint64_t pos=0; Ptr operator[](uint64_t v) const { return {pos+v}; } };
struct DataCopyParams { uint32_t blockCount,blockLen,dstStride,srcStride; };
void DataCopy(Ptr,Ptr,DataCopyParams) { assert(false); }
using Transfer=std::array<uint64_t,6>;
std::vector<Transfer> transfers;
void CopySingleMatrixNDToNZ(Ptr dst,Ptr src,uint32_t rows,uint32_t cols,uint64_t stride,uint32_t align) {
    transfers.push_back({dst.pos,src.pos,rows,cols,stride,align});
}
struct Parser {
    uint32_t generation=0, reads=0;
    int32_t GetBlockIdx(uint32_t b,uint32_t page) {
        ++reads; return ((page*73+b*29+generation*19)%401)+1;
    }
};
template<GmFormat> struct OffsetCalculator {
    Parser blockTableParser;
    uint64_t GetBlockSize() { return 128; }
    uint64_t GetStrideBlockSize() { return 512; }
    uint64_t GetStrideN2() { return 256; }
    uint64_t GetStrideBlockNum() { return 128*512; }
    uint64_t GetStrideD() { return 1; }
    __aicore__ inline uint64_t GetOffset(uint32_t bIdx, uint32_t n2Idx, uint32_t s2Idx, uint32_t dIdx)
    {
        uint64_t blockIdxInBatch = s2Idx / GetBlockSize(); // 获取block table上的索引
        uint64_t bsIdx = s2Idx % GetBlockSize();           // 获取在单个块上超出的行数
        int32_t blockIdx = blockTableParser.GetBlockIdx(bIdx, blockIdxInBatch);
        uint64_t offset =
            blockIdx * GetStrideBlockNum() + n2Idx * GetStrideN2() + bsIdx * GetStrideBlockSize() + dIdx * GetStrideD();
        return offset;
    }
};
template<class T,L1Format> struct FaL1Tensor { Ptr tensor; uint32_t rowCount; };
template<class T,GmFormat F> struct FaGmTensor { Ptr gmTensor; OffsetCalculator<F> offsetCalculator; };
struct GmKvCoord { uint32_t bIdx,n2Idx,s2Idx,dIdx,s2DealSize,dDealSize; };
template<class KV_T,GmFormat GM_FORMAT,L1Format L1_FORMAT=L1Format::NZ> struct Copy {
    __aicore__ inline void ProcessPageAttention(FaL1Tensor<KV_T, L1_FORMAT> &dstTensor,
                                                FaGmTensor<KV_T, GM_FORMAT> &srcTensor,
                                                GmKvCoord &gmCoord)
    {
        OffsetCalculator<GM_FORMAT> &offsetCalculator = srcTensor.offsetCalculator;
        uint32_t curS2Idx = gmCoord.s2Idx;
        uint32_t copyFinishRowCnt = 0;
        uint32_t blockElementCnt = 32 / sizeof(KV_T);
        if constexpr (IsSameType<KV_T, int4b_t>::value) {
            blockElementCnt = 64; // int4b时32B可以存64个元素
        }
        while (copyFinishRowCnt < gmCoord.s2DealSize) {
            // 获取需要拷贝的行数
            uint32_t copyRowCnt = offsetCalculator.GetBlockSize() - curS2Idx % offsetCalculator.GetBlockSize();
            if (copyFinishRowCnt + copyRowCnt > gmCoord.s2DealSize) {
                copyRowCnt = gmCoord.s2DealSize - copyFinishRowCnt; // 一个block未拷满
            }

            // 计算offset
            uint64_t gmOffset = offsetCalculator.GetOffset(gmCoord.bIdx, gmCoord.n2Idx, curS2Idx, gmCoord.dIdx);
            uint64_t l1Offset = copyFinishRowCnt * blockElementCnt;

            // 拷贝数据
            if constexpr (GM_FORMAT == GmFormat::PA_NZ) {
                DataCopyParams intriParams;
                intriParams.blockCount = gmCoord.dDealSize / blockElementCnt;
                intriParams.blockLen = copyRowCnt;
                intriParams.dstStride = dstTensor.rowCount - copyRowCnt;
                intriParams.srcStride = offsetCalculator.GetBlockSize() - copyRowCnt;
                DataCopy(dstTensor.tensor[l1Offset], srcTensor.gmTensor[gmOffset], intriParams);
            } else {
                CopySingleMatrixNDToNZ(dstTensor.tensor[l1Offset],
                                       srcTensor.gmTensor[gmOffset],
                                       copyRowCnt,
                                       gmCoord.dDealSize,
                                       offsetCalculator.GetStrideBlockSize(),
                                       dstTensor.rowCount);
            }

            // 更新完成拷贝的行数和s2Idx
            copyFinishRowCnt += copyRowCnt;
            curS2Idx += copyRowCnt;
        }
    }
    void operator()(FaL1Tensor<KV_T,L1_FORMAT> &dst,FaGmTensor<KV_T,GM_FORMAT> &src,GmKvCoord &coord) {
        ProcessPageAttention(dst,src,coord);
    }
};
struct RunInfo { uint32_t bIdx=0,n2Idx=0,s2Idx=0; bool isSinkTensor=false; };
template<class FIAT,class Config> struct FiaBlockCubeNonQuantGqa {
    using KV_T=uint16_t;
    static constexpr bool THESIS_PAGE_REUSE_STATIC=true;
    struct { bool batchContinuous=true; uint32_t s2BaseSize=512,headDim=256,kvHeadNum=2; } constInfo;
    Ptr vL1Tensor[2]{{0},{16384}};
    Ptr valueSinkGm;
    FaGmTensor<KV_T,GmFormat::PA_BnBsND> valueGmTensor;
    Copy<KV_T,GmFormat::PA_BnBsND> copyKvGmToL1;
    void CopyVToL1(uint32_t,const RunInfo&,uint32_t,uint32_t);
    __aicore__ inline void CopyVWithPageAddress(uint32_t dstBufId, const RunInfo &info,
        uint32_t subKStart, uint32_t subKSize, uint32_t &cachedPageEnd, uint64_t &cachedOffset);
};
template <typename FIAT, typename Config>
__aicore__ inline void FiaBlockCubeNonQuantGqa<FIAT, Config>::CopyVToL1(uint32_t dstBufId,
                                                                        const RunInfo &info,
                                                                        uint32_t subKStart,
                                                                        uint32_t subKSize)
{
    auto vL1Tensor = this->vL1Tensor[dstBufId];
    auto subKSizeAlign = Align(subKSize, (uint32_t)BLOCK_CUBE);

    FaL1Tensor<KV_T, L1Format::NZ> dstTensor{.tensor = vL1Tensor, .rowCount = subKSizeAlign};
    GmKvCoord gmCoord{.bIdx = constInfo.batchContinuous ? info.bIdx : 0,
                      .n2Idx = info.n2Idx,
                      .s2Idx = info.s2Idx * constInfo.s2BaseSize + subKStart,
                      .dIdx = 0,
                      .s2DealSize = subKSize,
                      .dDealSize = (uint32_t)constInfo.headDim};
    if (unlikely(info.isSinkTensor)) {
        uint64_t offset = info.n2Idx * constInfo.headDim;
        CopySingleMatrixNDToNZ(vL1Tensor,
                               valueSinkGm[offset],
                               subKSize,
                               (uint32_t)constInfo.headDim,
                               constInfo.kvHeadNum * constInfo.headDim,
                               subKSizeAlign);
    } else {
        copyKvGmToL1(dstTensor, valueGmTensor, gmCoord);
    }

#if DEBUG_MATMUL_GQA
    this->vMemSize += this->valueDSize * subKSize;
#endif
}
template <typename FIAT, typename Config>
__aicore__ inline void FiaBlockCubeNonQuantGqa<FIAT, Config>::CopyVWithPageAddress(
    uint32_t dstBufId, const RunInfo &info, uint32_t subKStart, uint32_t subKSize,
    uint32_t &cachedPageEnd, uint64_t &cachedOffset)
{
    // Only called for increasing 64-row blocks, aligned task start, page128 BSH.
    // Cache belongs to this ComputeMm2 invocation, never another task or replay.
    if constexpr (THESIS_PAGE_REUSE_STATIC) {
        constexpr uint32_t PAGE_ROWS = 128U;
        auto &offsetCalculator = valueGmTensor.offsetCalculator;
        const uint32_t s2 = info.s2Idx * constInfo.s2BaseSize + subKStart;
        if (s2 >= cachedPageEnd) {
            cachedOffset = offsetCalculator.GetOffset(
                constInfo.batchContinuous ? info.bIdx : 0, info.n2Idx, s2, 0);
            cachedPageEnd = (s2 & ~(PAGE_ROWS - 1U)) + PAGE_ROWS;
        }
        CopySingleMatrixNDToNZ(this->vL1Tensor[dstBufId], valueGmTensor.gmTensor[cachedOffset],
            subKSize, (uint32_t)constInfo.headDim, offsetCalculator.GetStrideBlockSize(),
            Align(subKSize, (uint32_t)BLOCK_CUBE));
        cachedOffset += subKSize * offsetCalculator.GetStrideBlockSize();
    }
}


int main() {
    FiaBlockCubeNonQuantGqa<int,int> old,newer;
    uint64_t cases=0,oldReads=0,newReads=0;
    for(uint32_t generation=0;generation<3;++generation)
    for(uint32_t continuous=0;continuous<2;++continuous)
    for(uint32_t batch=0;batch<3;++batch)
    for(uint32_t head=0;head<2;++head)
    for(uint32_t task=0;task<9;++task)
    for(uint32_t initialSlot=0;initialSlot<2;++initialSlot)
    for(uint32_t length=1;length<=512;++length) {
        old.constInfo.batchContinuous=newer.constInfo.batchContinuous=continuous;
        old.valueGmTensor.offsetCalculator.blockTableParser={generation,0};
        newer.valueGmTensor.offsetCalculator.blockTableParser={generation,0};
        RunInfo info{batch,head,task,false};
        transfers.clear();
        for(uint32_t k=0;k<length;k+=64)
            old.CopyVToL1((initialSlot+k/64)%2,info,k,std::min(64U,length-k));
        const auto expected=transfers;
        transfers.clear();
        uint32_t cachedPageEnd=0;
        uint64_t cachedOffset=0;
        for(uint32_t k=0;k<length;k+=64)
            newer.CopyVWithPageAddress((initialSlot+k/64)%2,info,k,std::min(64U,length-k),cachedPageEnd,cachedOffset);
        assert(expected==transfers);
        assert(old.valueGmTensor.offsetCalculator.blockTableParser.reads==(length+63)/64);
        assert(newer.valueGmTensor.offsetCalculator.blockTableParser.reads==(length+127)/128);
        oldReads+=old.valueGmTensor.offsetCalculator.blockTableParser.reads;
        newReads+=newer.valueGmTensor.offsetCalculator.blockTableParser.reads;
        ++cases;
    }
    printf("{\"cases\":%llu,\"original_queries\":%llu,\"candidate_queries\":%llu}\n",
        (unsigned long long)cases,(unsigned long long)oldReads,(unsigned long long)newReads);
}
