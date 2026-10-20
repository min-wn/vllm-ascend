# SPDX-License-Identifier: Apache-2.0
"""Task-local V page address reuse, without changing buffers or events."""

import argparse
import difflib
import hashlib
import json
from pathlib import Path

import aclgraph_fia_sink_mm2_prefetch_patch as prior
import aclgraph_fia_sink_mm2_transfer_patch as original

SOURCE = original.SOURCE
EXPECTED_SHA256 = original.EXPECTED_SHA256
MACRO = "THESIS_FIA_PAGE_REUSE"
SIGNATURE = prior.SIGNATURE
GUARD = prior.GUARD.replace("THESIS_FIA_MM2_PREFETCH", MACRO).replace(
    "THESIS_MM2_PREFETCH_STATIC", "THESIS_PAGE_REUSE_STATIC"
)
GATE = (
    prior.GATE.replace("prefetchMm2", "reusePageAddress")
    .replace("THESIS_MM2_PREFETCH_STATIC", "THESIS_PAGE_REUSE_STATIC")
    .replace("constInfo.s2BaseSize > 0U && constInfo.s2BaseSize <= 512U", "constInfo.s2BaseSize == 512U")
)
DECLARATION = """    __aicore__ inline void CopyVWithPageAddress(uint32_t dstBufId, const RunInfo &info,
        uint32_t subKStart, uint32_t subKSize, uint32_t &cachedPageEnd, uint64_t &cachedOffset);
"""
HELPER = """template <typename FIAT, typename Config>
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

"""
OLD_COPY = "                CopyVToL1(vBufId, info, kL1.start, kL1.sizeAct);"
NEW_COPY = """                if (reusePageAddress) {
                    CopyVWithPageAddress(vBufId, info, kL1.start, kL1.sizeAct, cachedPageEnd, cachedOffset);
                } else {
                    CopyVToL1(vBufId, info, kL1.start, kL1.sizeAct);
                }"""
LOCALS = GATE + "    uint32_t cachedPageEnd = 0;\n    uint64_t cachedOffset = 0;\n\n"


def transformed(before):
    if hashlib.sha256(before).hexdigest() != EXPECTED_SHA256:
        raise ValueError("expected the pinned, qualified D256 cube source")
    text = before.decode("utf-8")
    text = original.replace_once(text, SIGNATURE, HELPER + SIGNATURE)
    text = original.replace_once(text, OLD_COPY, NEW_COPY)
    text = original.replace_once(
        text, "    auto KL1Slices = k.Split(K_SPLIT_SIZE);", LOCALS + "    auto KL1Slices = k.Split(K_SPLIT_SIZE);"
    )
    text = original.replace_once(
        text,
        "    // =================================L1 Buffer=================================",
        GUARD + "    // =================================L1 Buffer=================================",
    )
    anchor = (
        "    __aicore__ inline void CopyVToL1(uint32_t dstBufId, const RunInfo &info, "
        "uint32_t subKStart, uint32_t subKSize);"
    )
    return original.replace_once(text, anchor, anchor + "\n" + DECLARATION).encode("utf-8")


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
        target="A2 BF16 TND BSH page128 Q4 GQA6 KV2 D256, single M<=64, task512, no sink/RoPE",
        task_local=True,
        cache_reset_each_compute=True,
        graph_split_exclusive=False,
        buffers_and_events_unchanged=True,
        l1_transfer_k=64,
        l0_compute_k=64,
        kv_data_bytes_saved=0,
        qualification="source only; native address contract and NPU checks required",
    )
    if not check:
        evidence, diff = source / "page-reuse-patch.json", source / "page-reuse-source.patch"
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
