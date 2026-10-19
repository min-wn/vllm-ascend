# SPDX-License-Identifier: Apache-2.0
"""Run the actual patched MM2 loop with native Axis and CPU event mocks.

The mocks check ownership and issue order, not device timing or asynchronous
pipeline semantics. Device correctness and latency require separate tests.
"""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import aclgraph_fia_sink_mm2_double_buffer_contract as previous
import aclgraph_fia_sink_mm2_prefetch_patch as patch

PREFIX = r"""
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
        const uint32_t expected=F==CubeFormat::NZ ? Align(info.actMBaseSize,16U) :
                                   info.actualSingleProcessSInnerSizeAlign;
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
"""
CLASS = r"""
template <typename FIAT,typename Config> struct @CLASS@ : State {
    using CFG=Config; using Q_T=typename FIAT::queryType; using KV_T=typename FIAT::kvType;
    static constexpr bool PAGE_ATTENTION=FIAT::paged;
    static constexpr auto LAYOUT_T=FIAT::layout,PA_LAYOUT=FIAT::pa;
@GUARD@
@DECL@
    template <CubeFormat O=CubeFormat::ND,CubeFormat A=CubeFormat::ND> void ComputeMm2(const RunInfo &info);
    bool enabled(const RunInfo &info) const {
@GATE@
        return prefetchMm2;
    }
};
"""
MAIN = r"""
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
"""


def program(before, axis):
    after = patch.transformed(before).decode()
    marker = "#endif // FIA_BLOCK_CUBE_NONQUANT_GQA_SINK_H"
    old = before.decode()
    old = old[old.index(patch.SIGNATURE) : old.index(marker)].replace("FiaBlockCubeNonQuantGqa", "Original")
    candidate = after[after.index(patch.HELPER) : after.index(marker)]
    classes = []
    for name in ("Original", "FiaBlockCubeNonQuantGqa"):
        classes.append(
            CLASS.replace("@CLASS@", name)
            .replace("@GUARD@", patch.GUARD)
            .replace("@DECL@", patch.DECLARATION)
            .replace("@GATE@", patch.GATE)
        )
    return PREFIX + axis.replace('#include "kernel_operator.h"', "") + "".join(classes) + old + candidate + MAIN


def run(native, output):
    native, output = Path(native).resolve(strict=True), Path(output).resolve(strict=True)
    identities = {}
    # Include the Vector producer as evidence of complete-task P readiness.
    names = [*previous.HELPERS, "fia_block_vec_nonquant_sink.h"]
    for name in names:
        data = (native / name).read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if name in previous.HELPERS and digest != previous.HELPERS[name]:
            raise ValueError("native helper drift: " + name)
        identities[name] = digest
        (output / (name.replace("/", "_") + ".txt")).write_bytes(data)
    vec = (native / "fia_block_vec_nonquant_sink.h").read_text()
    expected = (
        "    ProcessVec1SingleBuf(info);\n"
        "    CrossCoreSetFlag<ConstInfo::FIA_SYNC_MODE2, PIPE_MTE3>(constInfo.syncV1C2);"
    )
    if vec.count(expected) != 1:
        raise ValueError("Vector P readiness anchor drift")
    before = (native / patch.SOURCE.name).read_bytes()
    (output / "cube-before.h.txt").write_bytes(before)
    (output / "cube-candidate.h.txt").write_bytes(patch.transformed(before))
    source = output / "cpu-prefetch-contract.cpp"
    source.write_text(program(before, (native / "axis.h").read_text()), encoding="utf-8")
    for enabled in (0, 1):
        binary = output / ("cpu-prefetch-contract-" + str(enabled))
        with (output / (binary.name + ".log")).open("xb") as log:
            subprocess.run(
                [
                    "g++",
                    "-std=c++17",
                    "-O2",
                    "-Wall",
                    "-Wextra",
                    f"-D{patch.MACRO}={enabled}",
                    str(source),
                    "-o",
                    str(binary),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=60,
            )
            subprocess.run([str(binary)], stdout=log, stderr=subprocess.STDOUT, check=True, timeout=60)
    result = dict(
        passed=True,
        native_helpers=identities,
        actual_complete_compute_methods=True,
        actual_prefetch_helper=True,
        native_axis=True,
        macro_values=[0, 1],
        runtime_gate_cases=22,
        static_fallback_cases=8,
        paired_sequences_per_macro=147458,
        consecutive_calls_per_sequence=3,
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        scope="CPU issue order and token ownership; does not prove device async safety or gain",
    )
    (output / "cpu-prefetch-contract.json").write_text(json.dumps(result, indent=2) + "\n", encoding="ascii")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.native, args.output), indent=2))


if __name__ == "__main__":
    main()
