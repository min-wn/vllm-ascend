# SPDX-License-Identifier: Apache-2.0
"""Issue the next MM2 P/V transfer before the current Cube barrier.

This isolated experiment retains the original fixed-stride buffers, events,
64-row D256 products and task boundaries. Unsupported tasks use the old loop.
"""

import argparse
import difflib
import hashlib
import json
from pathlib import Path

import aclgraph_fia_sink_mm2_transfer_patch as original

SOURCE = original.SOURCE
EXPECTED_SHA256 = original.EXPECTED_SHA256
MACRO = "THESIS_FIA_MM2_PREFETCH"
SIGNATURE = """template <typename FIAT, typename Config>
template <CubeFormat OutFormat, CubeFormat AFormat>
__aicore__ inline void FiaBlockCubeNonQuantGqa<FIAT, Config>::ComputeMm2(const RunInfo &info)
"""
GUARD = """#if defined(THESIS_FIA_MM2_PREFETCH) && THESIS_FIA_MM2_PREFETCH && (__CCE_AICORE__ == 220)
    static constexpr bool THESIS_MM2_PREFETCH_STATIC =
        IsSameType<Q_T, bfloat16_t>::value && IsSameType<KV_T, bfloat16_t>::value &&
        PAGE_ATTENTION && LAYOUT_T == FIA_LAYOUT::TND && PA_LAYOUT == FIA_LAYOUT::BSH &&
        !CFG::S2_BASICSIZE_IS_1024 && CFG::PRELOAD_NUM == 2 && !CFG::ENABLE_UNIFLAG;
#else
    static constexpr bool THESIS_MM2_PREFETCH_STATIC = false;
#endif
"""
GATE = """    const bool prefetchMm2 = THESIS_MM2_PREFETCH_STATIC && constInfo.headDim == 256U &&
        constInfo.headDimRope == 0U && !constInfo.ropeSplitMode && constInfo.keySinkNumber == 0U &&
        constInfo.gSize == 6U && constInfo.kvHeadNum == 2U && constInfo.qHeadNum == 12U &&
        constInfo.qSeqSize == 4U && constInfo.headDimAlign == 256U && constInfo.preLoadNum == 2U &&
        constInfo.kvCacheBlockSize == 128U && constInfo.subBlockNum == 2U && constInfo.sinkNumber == 0 &&
        constInfo.s2BaseSize > 0U && constInfo.s2BaseSize <= 512U && !info.isSinkTensor &&
        info.actMBaseSize > 0U && info.actMBaseSize <= D256_MM2_QUERY_ROWS &&
        info.actualSingleProcessSInnerSize > 0U && info.actualSingleProcessSInnerSize <= 512U;
"""
DECLARATION = """    template <CubeFormat AFormat>
    __aicore__ inline void PrefetchMm2Block(const RunInfo &info, const Axis &m, const Axis &k,
                                          uint32_t kpBufId, uint32_t vBufId);
"""
HELPER = """template <typename FIAT, typename Config>
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

"""
PRIME = """        if (prefetchMm2) {
            // syncV1C2 above covers the complete P of this task, never the next task.
            PrefetchMm2Block<AFormat>(info, mL1, Axis(0, k.sizeAct < K_SPLIT_SIZE ? k.sizeAct : K_SPLIT_SIZE),
                                     this->kpL1BufId, this->vL1BufId);
        }
"""
NEXT = """                if (prefetchMm2 && !isLastK) {
                    const uint32_t nextK = kL1.start + K_SPLIT_SIZE;
                    const uint32_t nextSize = k.sizeAct - nextK;
                    PrefetchMm2Block<AFormat>(info, mL1, Axis(nextK, nextSize < K_SPLIT_SIZE ? nextSize : K_SPLIT_SIZE),
                                             (this->kpL1BufId + 1) % L1_KP_BUFCNT,
                                             (this->vL1BufId + 1) % L1_V_BUFCNT);
                }
"""


def compute_edits(text):
    """Retain original copy code verbatim inside the fallback branch."""
    begin = "            WaitFlag<HardEvent::MTE1_MTE2>(KP_EVENT0 + this->kpL1BufId);"
    end = "            for (auto &kL0 : kL1.Split(K_BASE)) {"
    copies = text[text.index(begin) : text.index(end)]
    fallback = copies.replace("            uint32_t vBufId;\n", "", 1)
    replacement = (
        "            uint32_t vBufId;\n"
        "            if (prefetchMm2) {\n"
        "                WaitFlag<HardEvent::MTE2_MTE1>(KP_EVENT0 + this->kpL1BufId);\n"
        "                WaitFlag<HardEvent::MTE2_MTE1>(V_EVENT0 + this->vL1BufId);\n"
        "                vBufId = this->vL1BufId;\n"
        "                reuseVBuf = false;\n"
        "                vL1Snapshot.bufCnt += static_cast<uint32_t>(canFullLoadV);\n"
        "            } else {\n"
        + "".join("    " + line if line.strip() else line for line in fallback.splitlines(True))
        + "            }\n\n"
    )
    return (
        ("    auto KL1Slices = k.Split(K_SPLIT_SIZE);", GATE + "\n    auto KL1Slices = k.Split(K_SPLIT_SIZE);"),
        ("        int32_t kL1Id = -1;", PRIME + "\n        int32_t kL1Id = -1;"),
        (copies, replacement),
        ("                if (likely(!isLastK)) {", NEXT + "                if (likely(!isLastK)) {"),
    )


def transformed(before):
    if hashlib.sha256(before).hexdigest() != EXPECTED_SHA256:
        raise ValueError("expected the pinned, qualified D256 cube source")
    text = before.decode("utf-8")
    start = text.index(SIGNATURE)
    tail = text.index("#endif // FIA_BLOCK_CUBE_NONQUANT_GQA_SINK_H", start)
    compute = text[start:tail]
    for old, new in compute_edits(compute):
        compute = original.replace_once(compute, old, new)
    text = text[:start] + HELPER + compute + text[tail:]
    text = original.replace_once(
        text,
        "    // =================================L1 Buffer=================================",
        GUARD + "    // =================================L1 Buffer=================================",
    )
    anchor = (
        "    __aicore__ inline void CopyVToL1(uint32_t dstBufId, const RunInfo &info, "
        "uint32_t subKStart, uint32_t subKSize);"
    )
    text = original.replace_once(text, anchor, anchor + "\n" + DECLARATION)
    return text.encode("utf-8")


def patch_tree(source, check=False):
    source = Path(source).resolve(strict=True)
    path = (source / SOURCE).resolve(strict=True)
    if not path.is_relative_to(source) or not path.is_file():
        raise ValueError("cube source escapes the isolated source tree")
    before = path.read_bytes()
    after = transformed(before)
    manifest = dict(
        source_file=SOURCE.as_posix(),
        before_sha256=hashlib.sha256(before).hexdigest(),
        after_sha256=hashlib.sha256(after).hexdigest(),
        macro=MACRO,
        macro_enabled_by_patch_tool=False,
        target="A2 BF16 TND paged ND128 Q4/GQA6/D256, single M tile <=64, 0<K<=512, no sink/RoPE/uniflag",
        l1_total_bytes=475136,
        l1_kp_slots=3,
        l1_v_slots=2,
        l0_compute_k=64,
        l1_transfer_k=64,
        new_task_or_workspace_prefetch=False,
        next_transfer_before_cube_barrier=True,
        original_slot_release_position=True,
        buffer_allocations_unchanged=True,
        unchanged="MM1, copies, load parameters, Mmad order, slot release, softmax, plans, updates, graph sizes",
        fallback="original loop retained; only gated single-M tasks take the prefetch branch",
        qualification="source only; CPU event model and restricted device checks required",
        graph_split_exclusive=False,
    )
    if not check:
        evidence, diff = source / "mm2-prefetch-patch.json", source / "mm2-prefetch-source.patch"
        if evidence.exists() or diff.exists():
            raise ValueError("refusing to overwrite existing patch evidence")
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
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    print(json.dumps(patch_tree(args.source, args.check), indent=2))


if __name__ == "__main__":
    main()
