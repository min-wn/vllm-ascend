# SPDX-License-Identifier: Apache-2.0
"""Patch only an isolated pinned FIA source with guarded running-O residency.

The macro is enabled by the separate experimental build, never by this tool.
This changes the consumer, not the producer's plan or graph replay protocol.
"""

import argparse
import difflib
import hashlib
import json
from pathlib import Path

COMMIT = "bec19b8dea1e538e64838f3064f7e91a3fa9ee72"
SOURCE = Path("csrc/attention/fused_infer_attention_score_v2_sink/op_kernel/fia_block_vec_nonquant_sink.h")
EXPECTED_SHA256 = "4f1a8185300f1f6bbf7728b778d47ce49dc929658b629d8e330582e49866f1e0"
MACRO = "THESIS_FIA_RUNNING_O_UB"

STATE_HELPER = r"""
#if defined(THESIS_FIA_RUNNING_O_UB) && THESIS_FIA_RUNNING_O_UB && (__CCE_AICORE__ == 220)
// Vec2 consumes monotonically ordered tasks. Vec1 never touches this state.
struct ThesisRunningOState {
    bool active = false;
    bool isFirstSInnerLoop = false, isLastS2Loop = false;
    uint32_t bIdx = 0, n2Idx = 0, gS1Idx = 0, actMBaseSize = 0;
    uint32_t vecStartM = 0, elements = 0, loop = 0, s2Idx = 0;
    uint64_t workspaceOffset = 0;

    __aicore__ inline bool SameOwner(const RunInfo &info, uint32_t vecStart, uint32_t count) const
    {
        return active && !isLastS2Loop && bIdx == info.bIdx && n2Idx == info.n2Idx && gS1Idx == info.gS1Idx &&
               actMBaseSize == info.actMBaseSize && vecStartM == vecStart && elements == count &&
               loop + 1U == info.loop && s2Idx + 1U == info.s2Idx;
    }

    __aicore__ inline bool CanUse(const RunInfo &info, bool supported, uint32_t vecStart, uint32_t count) const
    {
        return supported && (info.isFirstSInnerLoop || SameOwner(info, vecStart, count));
    }

    __aicore__ inline void Remember(const RunInfo &info, uint32_t vecStart, uint32_t count, uint64_t offset)
    {
        active = true;
        isFirstSInnerLoop = info.isFirstSInnerLoop;
        isLastS2Loop = info.isLastS2Loop;
        bIdx = info.bIdx;
        n2Idx = info.n2Idx;
        gS1Idx = info.gS1Idx;
        actMBaseSize = info.actMBaseSize;
        vecStartM = vecStart;
        elements = count;
        loop = info.loop;
        s2Idx = info.s2Idx;
        workspaceOffset = offset;
    }
};
#endif
"""

MEMBERS = r"""
#if defined(THESIS_FIA_RUNNING_O_UB) && THESIS_FIA_RUNNING_O_UB && (__CCE_AICORE__ == 220)
    // Recover the 16 KiB accumulator from outputQue1; total UB stays 184 KiB.
    TBuf<> thesisRunningOBuff;
    ThesisRunningOState thesisRunningOState;
    bool thesisRunningOEnabled = false;
    __aicore__ inline bool ThesisRunningOSupported() const;
    __aicore__ inline bool ThesisPrepareRunningO(const RunInfo &info, uint32_t startRow,
                                                uint32_t dealRowCount, uint32_t columnCount);
    __aicore__ inline void ThesisSpillRunningO();
#endif
"""

ALLOCATION = r"""#if defined(THESIS_FIA_RUNNING_O_UB) && THESIS_FIA_RUNNING_O_UB && (__CCE_AICORE__ == 220)
    thesisRunningOEnabled = ThesisRunningOSupported();
    if (thesisRunningOEnabled) {
        // Q4*G6 splits 16/8 rows across the AIVs, not 12/12.
        // Vec1 BF16: 16*512*2 = 16 KiB; final cast: 16*256*2 = 8 KiB.
        pipe->InitBuffer(outputQue1, 1, ConstInfo::BUFFER_SIZE_BYTE_16K);
        pipe->InitBuffer(thesisRunningOBuff, ConstInfo::BUFFER_SIZE_BYTE_16K);
    } else {
        pipe->InitBuffer(outputQue1, 1, ConstInfo::BUFFER_SIZE_BYTE_32K);
    }
#else
    pipe->InitBuffer(outputQue1, 1, ConstInfo::BUFFER_SIZE_BYTE_32K);
#endif"""

METHODS = r"""
#if defined(THESIS_FIA_RUNNING_O_UB) && THESIS_FIA_RUNNING_O_UB && (__CCE_AICORE__ == 220)
template <typename FIAT>
__aicore__ inline bool FiaBlockVecNonQuant<FIAT>::ThesisRunningOSupported() const
{
    // qSeqSize is native metadata max-S1, not packed physical query T.
    return PAGE_ATTENTION && LAYOUT_T == FIA_LAYOUT::TND &&
           IsSameType<Q_T, bfloat16_t>::value && IsSameType<KV_T, bfloat16_t>::value &&
           IsSameType<OUT_T, bfloat16_t>::value && constInfo.qSeqSize == 4U &&
           constInfo.gSize == 6U && constInfo.qHeadNum == 12U && constInfo.kvHeadNum == 2U &&
           constInfo.headDim == 256U && constInfo.headDimAlign == 256U &&
           constInfo.preLoadNum == 2U && constInfo.s2BaseSize > 0U && constInfo.s2BaseSize <= 512U &&
           constInfo.kvCacheBlockSize == 128U && constInfo.subBlockNum == 2U &&
           constInfo.sinkNumber == 0 && constInfo.keySinkNumber == 0 && !learnableSinkFlag &&
           !constInfo.ropeSplitMode && constInfo.headDimRope == 0U;
}

template <typename FIAT>
__aicore__ inline void FiaBlockVecNonQuant<FIAT>::ThesisSpillRunningO()
{
    if (!thesisRunningOState.active) {
        return;
    }
    // Cold fallback: publish the last resident value before using legacy GM.
    // This also protects accumulator lifetime if the next task has a new owner.
    AscendC::PipeBarrier<PIPE_V>();
    event_t vToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(vToMte3);
    WaitFlag<HardEvent::V_MTE3>(vToMte3);
    DataCopy(vec2ResGm[thesisRunningOState.workspaceOffset], thesisRunningOBuff.Get<COMPUTE_T>(),
             thesisRunningOState.elements);
    event_t mte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2);
    WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2);
    event_t mte3ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
    SetFlag<HardEvent::MTE3_V>(mte3ToV);
    WaitFlag<HardEvent::MTE3_V>(mte3ToV);
    thesisRunningOState.active = false;
}

template <typename FIAT>
__aicore__ inline bool FiaBlockVecNonQuant<FIAT>::ThesisPrepareRunningO(
    const RunInfo &info, uint32_t startRow, uint32_t dealRowCount, uint32_t columnCount)
{
    const bool supported = thesisRunningOEnabled && info.actS1Size == 4U && info.actMBaseSize == 24U &&
                           !info.tndIsS2SplitCore && !info.isSinkTensor && startRow == 0U &&
                           mSplitInfo.nBufferStartM == 0U && dealRowCount == mSplitInfo.vecDealM &&
                           dealRowCount > 0U && dealRowCount <= 16U && columnCount == 256U &&
                           info.actualSingleProcessSInnerSizeAlign > 0U &&
                           info.actualSingleProcessSInnerSizeAlign <= 512U;
    const bool use = thesisRunningOState.CanUse(info, supported, mSplitInfo.vecStartM,
                                               dealRowCount * columnCount);
    if (thesisRunningOState.active && (info.isFirstSInnerLoop || !use)) {
        ThesisSpillRunningO();
    }
    return use;
}
#endif
"""

LOAD_OLD = r"""
        event_t eventIdMte2WaitMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(eventIdMte2WaitMte3);
        WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte2WaitMte3);
        LocalTensor<COMPUTE_T> bmm2ResPreUb = inputQue1.AllocTensor<COMPUTE_T>();
        uint64_t vec2ResGmOffset = ((info.loop - 1) % constInfo.preLoadNum) * constInfo.bmm2ResUbSize + inOutBaseOffset;
        DataCopy(bmm2ResPreUb, vec2ResGm[vec2ResGmOffset], vec2ComputeSize);
        inputQue1.EnQue(bmm2ResPreUb);

        inputQue1.DeQue<COMPUTE_T>();""".lstrip("\n")

LOAD_NEW = r"""        LocalTensor<COMPUTE_T> bmm2ResPreUb;
#if defined(THESIS_FIA_RUNNING_O_UB) && THESIS_FIA_RUNNING_O_UB && (__CCE_AICORE__ == 220)
        if (thesisUseRunningO) {
            // Only a preceding qualified task may make this value readable.
            AscendC::PipeBarrier<PIPE_V>();
            bmm2ResPreUb = thesisRunningOBuff.Get<COMPUTE_T>();
        } else
#endif
        {
            event_t eventIdMte2WaitMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
            SetFlag<HardEvent::MTE3_MTE2>(eventIdMte2WaitMte3);
            WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte2WaitMte3);
            bmm2ResPreUb = inputQue1.AllocTensor<COMPUTE_T>();
            uint64_t vec2ResGmOffset =
                ((info.loop - 1) % constInfo.preLoadNum) * constInfo.bmm2ResUbSize + inOutBaseOffset;
            DataCopy(bmm2ResPreUb, vec2ResGm[vec2ResGmOffset], vec2ComputeSize);
            inputQue1.EnQue(bmm2ResPreUb);
            inputQue1.DeQue<COMPUTE_T>();
        }"""

STORE_OLD = r"""        LocalTensor<COMPUTE_T> tmpBmm2Res = outputQue1.AllocTensor<COMPUTE_T>();
        DataCopy(tmpBmm2Res, bmm2ResUb, dealRowCount * columnCount);
        outputQue1.EnQue(tmpBmm2Res);
        outputQue1.DeQue<COMPUTE_T>();
        uint64_t vec2ResGmOffset = (info.loop % constInfo.preLoadNum) * constInfo.bmm2ResUbSize + inOutBaseOffset;
        DataCopy(vec2ResGm[vec2ResGmOffset], tmpBmm2Res, vec2ComputeSize);

        outputQue1.FreeTensor(tmpBmm2Res);"""

STORE_NEW = (
    r"""#if defined(THESIS_FIA_RUNNING_O_UB) && THESIS_FIA_RUNNING_O_UB && (__CCE_AICORE__ == 220)
        if (thesisUseRunningO) {
            DataCopy(thesisRunningOBuff.Get<COMPUTE_T>(), bmm2ResUb, vec2ComputeSize);
            AscendC::PipeBarrier<PIPE_V>();
            // ComputeVec2 publishes syncV2C2 on PIPE_MTE3. Preserve its V dependency.
            event_t vToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
            SetFlag<HardEvent::V_MTE3>(vToMte3);
            WaitFlag<HardEvent::V_MTE3>(vToMte3);
            const uint64_t offset = (info.loop % constInfo.preLoadNum) * constInfo.bmm2ResUbSize + inOutBaseOffset;
            thesisRunningOState.Remember(info, mSplitInfo.vecStartM, vec2ComputeSize, offset);
        } else
#endif
        {
"""
    + "\n".join("    " + line if line else line for line in STORE_OLD.splitlines())
    + r"""
        }"""
)


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"pinned source anchor changed: {old[:90]!r}")
    return text.replace(old, new, 1)


def transformed(original):
    if hashlib.sha256(original).hexdigest() != EXPECTED_SHA256:
        raise ValueError("vector source differs from the pinned unmodified source")
    text = original.decode("utf-8")
    text = replace_once(
        text,
        "template <typename FIAT> class FiaBlockVecNonQuant {",
        STATE_HELPER + "\ntemplate <typename FIAT> class FiaBlockVecNonQuant {",
    )
    text = replace_once(text, "    ConstInfo constInfo = {};", MEMBERS + "\n    ConstInfo constInfo = {};")
    text = replace_once(text, "    pipe->InitBuffer(outputQue1, 1, ConstInfo::BUFFER_SIZE_BYTE_32K);", ALLOCATION)
    marker = (
        "template <typename FIAT> __aicore__ inline void "
        "FiaBlockVecNonQuant<FIAT>::ProcessVec2SingleBuf(const RunInfo &info)"
    )
    text = replace_once(text, marker, METHODS + "\n" + marker)
    text = replace_once(
        text,
        "    uint32_t mSplitSize = BASE_BLOCK_MAX_ELEMENT_NUM / constInfo.headDimAlign;",
        "    uint32_t mSplitSize = BASE_BLOCK_MAX_ELEMENT_NUM / constInfo.headDimAlign;\n"
        "#if defined(THESIS_FIA_RUNNING_O_UB) && THESIS_FIA_RUNNING_O_UB && (__CCE_AICORE__ == 220)\n"
        "    if (thesisRunningOEnabled && mSplitSize > 16U) {\n"
        "        mSplitSize = 16U; // Every fallback FP32 output staging also fits 16 KiB.\n    }\n#endif",
    )
    anchor = "    LocalTensor<MM2_OUT_T> bmm2ResUb = inputQue1.AllocTensor<MM2_OUT_T>();"
    text = replace_once(
        text,
        anchor,
        "#if defined(THESIS_FIA_RUNNING_O_UB) && THESIS_FIA_RUNNING_O_UB && (__CCE_AICORE__ == 220)\n"
        "    const bool thesisUseRunningO = ThesisPrepareRunningO(info, startRow, dealRowCount, columnCount);\n"
        "#endif\n" + anchor,
    )
    text = replace_once(text, LOAD_OLD, LOAD_NEW)
    text = replace_once(
        text,
        "        inputQue1.FreeTensor(bmm2ResPreUb);",
        "#if defined(THESIS_FIA_RUNNING_O_UB) && THESIS_FIA_RUNNING_O_UB && (__CCE_AICORE__ == 220)\n"
        "        if (!thesisUseRunningO)\n#endif\n        {\n"
        "            inputQue1.FreeTensor(bmm2ResPreUb);\n        }",
    )
    text = replace_once(text, STORE_OLD, STORE_NEW)
    anchor = "        Bmm2ResCopyOut(info, bmm2ResUb, mStart, startRow, dealRowCount, columnCount, actualColumnCount);"
    return replace_once(
        text,
        anchor,
        anchor + "\n#if defined(THESIS_FIA_RUNNING_O_UB) && THESIS_FIA_RUNNING_O_UB && (__CCE_AICORE__ == 220)\n"
        "        thesisRunningOState.active = false;\n#endif",
    ).encode("utf-8")


def patch_tree(source, check=False):
    source = source.resolve(strict=True)
    path = (source / SOURCE).resolve(strict=True)
    if not path.is_relative_to(source) or not path.is_file():
        raise ValueError("vector source escapes the isolated source tree")
    before = path.read_bytes()
    after = transformed(before)
    manifest = {
        "source_commit": COMMIT,
        "source_file": SOURCE.as_posix(),
        "before_sha256": hashlib.sha256(before).hexdigest(),
        "after_sha256": hashlib.sha256(after).hexdigest(),
        "macro": MACRO,
        "macro_enabled_by_patch_tool": False,
        "scope": "isolated A2 BF16 paged TND Q4 GQA6 D256 consumer; plan-policy independent",
        "static_guard": "max-S1=4, Q12/KV2, D256/aligned256, page128, preload2, S2<=512, no sink/RoPE",
        "task_guard": "actual S1=4, actual M=24, !FD, !sink, one AIV16/8-row slice, contiguous same-owner S2 loops",
        "ub_bytes": {"original": 188416, "patched": 188416, "running_o": 16384, "output1": 16384},
        "max_output1_bytes": {"vec1_bf16": 16384, "fallback_fp32": 16384, "final_bf16": 8192},
        "fallback": (
            "unsupported heads use workspace; same-shape native tasks may cold-spill the saved owner with "
            "V/MTE3/MTE2 events; malformed descriptors or changing a head's actual M are outside the native contract"
        ),
        "qualification": "source patch only; requires native compilation and numerical/performance tests",
    }
    if not check:
        evidence = source / "running-o-ub-patch.json"
        diff = source / "running-o-ub-source.patch"
        if evidence.exists() or diff.exists():
            raise ValueError("running-O patch evidence already exists; refusing to overwrite")
        evidence.write_text(json.dumps(manifest, indent=2) + "\n", encoding="ascii")
        diff.write_text(
            "".join(
                difflib.unified_diff(
                    before.decode().splitlines(True),
                    after.decode().splitlines(True),
                    fromfile="a/" + SOURCE.as_posix(),
                    tofile="b/" + SOURCE.as_posix(),
                )
            ),
            encoding="utf-8",
        )
        path.write_bytes(after)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Isolated source copy containing csrc/")
    parser.add_argument("--check", action="store_true", help="Check the complete transform without writing")
    args = parser.parse_args()
    print(json.dumps(patch_tree(args.source, args.check), indent=2))


if __name__ == "__main__":
    main()
