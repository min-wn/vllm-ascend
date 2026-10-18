# SPDX-License-Identifier: Apache-2.0
"""Pinned native patch regressions; a C++ harness runs when a compiler exists."""

import hashlib
import importlib.util
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.fixture
def patch_module():
    root = Path(__file__).resolve().parents[4]
    spec = importlib.util.spec_from_file_location(
        "fia_stage5_patch_cpu", root / "benchmarks/aclgraph_fia_sink_stage5_patch.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def corrected_source(module):
    root = Path(__file__).resolve().parents[6]
    path = root / "outputs/thesis_incremental_plan/pinned_sink_source/source" / module.METADATA
    if not path.is_file():
        pytest.skip("pinned native source archive is not present in this checkout")
    source = path.read_bytes()
    old = b'#include "../../common/aicpu/cpu_context_util.h"'
    new = b'#include "../aicpu/cpu_context_util.h"'
    assert source.count(old) == 1
    return source.replace(old, new)


def test_patch_accepts_only_exact_stage_two_source(patch_module):
    source = corrected_source(patch_module)
    assert hashlib.sha256(source).hexdigest() == patch_module.EXPECTED_STAGE2_SHA256
    patched = patch_module.transformed(source)
    assert patched.count(patch_module.HELPER.encode()) == 1
    assert patched.count(patch_module.COMPUTE_POLICY.encode()) == 1
    with pytest.raises(ValueError, match="pinned stage-two"):
        patch_module.transformed(source + b"\n")
    with pytest.raises(ValueError, match="pinned stage-two"):
        patch_module.transformed(patched)


def test_policy_uses_native_costs_and_fresh_transactional_result(patch_module):
    helper, policy = patch_module.HELPER, patch_module.COMPUTE_POLICY
    assert "CalcSplitInfo(wholeContext);" in policy
    assert "CalcCostInfo(wholeContext);" in policy
    assert "wholeContext.costInfo" in policy
    assert "costInfo.bN2CostOfEachBatch[batch]" in helper
    assert "qCost" not in helper and "kvCost" not in helper
    assert "SplitResult candidate{coreNum, result.vecCubeRatio};" in helper
    assert "result = candidate;" in helper
    assert helper.index("result = candidate;") > helper.index("if (start != totalHeads)")
    assert "if (!wholeRowBuilt)" in policy
    assert 'inputLayout_ == "TND"' in policy and 'inputLayoutKv_ == "BnBsH"' in policy
    for restriction in (
        "headDimQK_ == 256U",
        "headDimV_ == 256U",
        "ropeHeadDim_ == 0U",
        "blockSize_ == 128",
        "!batchInvariant_",
        "aicCoreNum_ <= 20U",
        "mBaseSize_ == 512U",
    ):
        assert restriction in policy
    assert "baseInfo.actualSeqS1Size[batch] != static_cast<int64_t>((batch + 1U) * 4U)" in helper
    assert "((baseInfo.actualSeqS2Size[batch] == 0) != (cost == 0))" in helper


CPP_STUBS = r"""
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <vector>
using std::uint32_t;
struct BaseInfo {
    bool batchInvariant=false, isS1G=true, isAccumSeqS1=true, isAccumSeqS2=false;
    uint32_t sparseMode=3, n2Size=2, gSize=6, s1Size=4, bSize=4, actualLenQDims=4, actualLenKvDims=4;
    int64_t sinkNumber=0,keySinkNumber=0;
    std::vector<int64_t> actualSeqS1Size{4,8,12,16},actualSeqS2Size{129,0,513,0};
};
struct CostInfo { std::vector<int64_t> bN2CostOfEachBatch; };
struct FD {
    std::vector<uint32_t> bN2IdxOfFdHead,gS1IdxOfFdHead,s2SplitNumOfFdHead,s2SplitStartIdxOfCore;
    std::vector<uint32_t> gS1SplitNumOfFdHead,gS1LastPartSizeOfFdHead,gS1IdxEndOfFdHead,gS1IdxEndOfFdHeadSplit;
    FD(uint32_t c,uint32_t r):bN2IdxOfFdHead(c),gS1IdxOfFdHead(c),s2SplitNumOfFdHead(c),
      s2SplitStartIdxOfCore(c),gS1SplitNumOfFdHead(c),gS1LastPartSizeOfFdHead(c),
      gS1IdxEndOfFdHead(c*r),gS1IdxEndOfFdHeadSplit(c*r){}
};
struct SplitResult {
    uint32_t usedCoreNum=0,vecCubeRatio=0,numOfFdHead=0,maxS2SplitNum=0,usedVecNumOfFd=0;
    int64_t maxCost=0;
    std::vector<uint32_t> bN2End,gS1End,s2End;
    FD fdRes;
    SplitResult(uint32_t c,uint32_t r):vecCubeRatio(r),bN2End(c),gS1End(c),s2End(c),fdRes(c,r){}
};
"""

CPP_MAIN = r"""
static void CheckNoFd(const SplitResult &r) {
    assert(r.numOfFdHead==0 && r.usedVecNumOfFd==0 && r.maxS2SplitNum==0);
    for(auto p:{&r.gS1End,&r.s2End,&r.fdRes.bN2IdxOfFdHead,&r.fdRes.gS1IdxOfFdHead,
        &r.fdRes.s2SplitNumOfFdHead,&r.fdRes.s2SplitStartIdxOfCore,&r.fdRes.gS1SplitNumOfFdHead,
        &r.fdRes.gS1LastPartSizeOfFdHead,&r.fdRes.gS1IdxEndOfFdHead,&r.fdRes.gS1IdxEndOfFdHeadSplit}) {
        assert(std::all_of(p->begin(),p->end(),[](uint32_t v){return v==0;}));
    }
}
int main() {
    BaseInfo b;
    CostInfo costs{{42,0,114,0}};
    SplitResult result(20,2);
    result.numOfFdHead=12;result.usedVecNumOfFd=36;
    for(auto &v:result.fdRes.s2SplitNumOfFdHead)v=2;
    assert(BuildWholeRowPlan(b,20,costs,result));
    CheckNoFd(result);
    assert(result.usedCoreNum==4 && result.bN2End[3]==8);
    uint32_t previous=0;
    for(uint32_t i=0;i<result.usedCoreNum;++i){assert(result.bN2End[i]>previous);previous=result.bN2End[i];}
    const auto saved=result.bN2End;
    b.actualSeqS1Size[2]=11;
    assert(!BuildWholeRowPlan(b,20,costs,result));assert(result.bN2End==saved);
    b.actualSeqS1Size[2]=12;
    b.actualSeqS2Size={0,0,0,0};costs.bN2CostOfEachBatch={0,0,0,0};
    assert(BuildWholeRowPlan(b,20,costs,result));CheckNoFd(result);
    assert(result.usedCoreNum==1 && result.bN2End[0]==8 && result.maxCost==0);
    b.actualSeqS2Size={129,0,513,0};costs.bN2CostOfEachBatch={42,0,114,0};
    b.actualLenQDims=3;assert(!BuildWholeRowPlan(b,20,costs,result));
    b.actualLenQDims=4;assert(!BuildWholeRowPlan(b,21,costs,result));
    costs.bN2CostOfEachBatch[1]=12;assert(!BuildWholeRowPlan(b,20,costs,result));
}
"""


def test_extracted_cpp_helper_clears_stale_fd_and_preserves_fallback(patch_module, tmp_path):
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("no local C++ compiler; run this CPU-only harness on the native build host")
    source = tmp_path / "whole_row.cpp"
    binary = tmp_path / "whole_row"
    source.write_text(
        CPP_STUBS + "\n#define THESIS_NO_FD_WHOLE_ROW\n" + patch_module.HELPER + CPP_MAIN, encoding="ascii"
    )
    subprocess.run([compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror", str(source), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)
