# SPDX-License-Identifier: Apache-2.0
"""Execute the actual address helper against native paged-copy methods on CPU.

DMA is recorded, not executed. This validates scalar addresses and extents,
not asynchronous hardware behavior or device correctness.
"""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import aclgraph_fia_sink_page_reuse_patch as patch


def extract(text, anchor):
    start = text.index(anchor)
    body = text.index("{", start)
    depth = 1
    end = body + 1
    while depth:
        depth += (text[end] == "{") - (text[end] == "}")
        end += 1
    return text[start:end]


def program(cube, copy, offset):
    patch.transformed(cube)
    page_copy = extract(copy, "    __aicore__ inline void ProcessPageAttention(FaL1Tensor<KV_T, L1_FORMAT> &dstTensor,")
    paged = offset[offset.index("struct OffsetCalculatorImpl<FORMAT, FormatCategory::GM_KV_PA_BNBD, ACTLEN_T>") :]
    getter = extract(paged, "    __aicore__ inline uint64_t GetOffset(")
    copy_v = extract(
        cube.decode(),
        "template <typename FIAT, typename Config>\n"
        "__aicore__ inline void FiaBlockCubeNonQuantGqa<FIAT, Config>::CopyVToL1(",
    )
    return (
        r"""
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
"""
        + getter
        + r"""
};
template<class T,L1Format> struct FaL1Tensor { Ptr tensor; uint32_t rowCount; };
template<class T,GmFormat F> struct FaGmTensor { Ptr gmTensor; OffsetCalculator<F> offsetCalculator; };
struct GmKvCoord { uint32_t bIdx,n2Idx,s2Idx,dIdx,s2DealSize,dDealSize; };
template<class KV_T,GmFormat GM_FORMAT,L1Format L1_FORMAT=L1Format::NZ> struct Copy {
"""
        + page_copy
        + r"""
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
"""
        + patch.DECLARATION
        + "};\n"
        + copy_v
        + "\n"
        + patch.HELPER
        + r"""
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
"""
    )


def run(native, output):
    native, output = Path(native), Path(output)
    files = {
        "cube": native / "fia_block_cube_nonquant_gqa_sink.h",
        "copy": native / "memcopy/copy_gm_to_l1.h",
        "offset": native / "memcopy/offset_calculator_v2.h",
    }
    data = {key: path.read_bytes() for key, path in files.items()}
    text = program(data["cube"], data["copy"].decode(), data["offset"].decode())
    source, binary = output / "page-contract.cpp", output / "page-contract"
    if source.exists() or binary.exists():
        raise ValueError("refusing to overwrite CPU contract")
    source.write_text(text, encoding="utf-8")
    command = ["g++", "-std=c++17", "-O2", "-Wall", "-Wextra", str(source), "-o", str(binary)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    (output / "page-contract-compile.log").write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError("native address contract failed to compile")
    result = subprocess.run([str(binary)], capture_output=True, text=True, check=True, timeout=120)
    report = dict(
        state="passed",
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        native_sha256={key: hashlib.sha256(value).hexdigest() for key, value in data.items()},
        counts=json.loads(result.stdout),
        hardware_async_validated=False,
    )
    (output / "page-contract.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.native, args.output), indent=2))
